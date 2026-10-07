"""Keep the Optuna SQLite study on local disk and copy it to/from a Databricks volume.

SQLite needs file locking and random writes, which Databricks FUSE mounts (``/Volumes``,
``/dbfs``, ``/Workspace``) do not support, so a ``sqlite:////Volumes/...`` storage fails
with ``disk I/O error``. The simple fix:

1. before the study is opened, copy ``study.db`` from the volume to ``/local_disk0``;
2. run Optuna on the local file;
3. after every trial, ``shutil.copy`` the local file back to the volume.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import optuna
from optuna.trial import TrialState

FUSE_PREFIXES = ("/dbfs", "/Volumes", "/Workspace")


def normalize_path(path: str | Path) -> Path:
    """Map ``dbfs:/x`` to the FUSE path ``/dbfs/x``; leave other paths unchanged."""
    s = str(path)
    if s.startswith("dbfs:/"):
        s = "/dbfs/" + s[len("dbfs:/"):].lstrip("/")
    return Path(s)


def is_fuse_path(path: str | Path) -> bool:
    """True for Databricks FUSE mounts where SQLite cannot run directly."""
    s = str(normalize_path(path).absolute())
    return any(s == p or s.startswith(p + "/") for p in FUSE_PREFIXES)


def default_local_dir() -> Path:
    # /local_disk0 is the driver's local SSD on Databricks.
    base = Path("/local_disk0/tmp") if Path("/local_disk0").is_dir() else Path(tempfile.gettempdir())
    return base / "optuna"


def copy_study_to_local(volume_db: str | Path, local_db: str | Path) -> str:
    """Copy the saved study from the volume to local disk; return the local storage URL.

    * volume copy exists, local missing or older -> copy volume -> local (resume)
    * local copy newer (same cluster, not restarted) -> keep local
    * no volume copy -> delete any leftover local file and start a fresh study
    """
    volume_db, local_db = normalize_path(volume_db), Path(local_db)
    local_db.parent.mkdir(parents=True, exist_ok=True)
    if volume_db.exists():
        if not local_db.exists() or local_db.stat().st_mtime < volume_db.stat().st_mtime:
            shutil.copy(volume_db, local_db)
    elif local_db.exists():
        local_db.unlink()
    return f"sqlite:///{local_db.absolute()}"


def save_study_to_volume(local_db: str | Path, volume_db: str | Path) -> None:
    """Copy the local study DB to the volume (a whole-file copy works on FUSE mounts)."""
    local_db, volume_db = Path(local_db), normalize_path(volume_db)
    if local_db.exists():
        volume_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_db, volume_db)


def load_study_from_volume(
    study_name: str,
    volume_db: str | Path,
    local_db: str | Path | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    retry_interrupted: bool = True,
) -> tuple[optuna.Study, Path]:
    """Copy a saved study from the volume to local disk and load it to continue tuning.

    The sampler is not stored in the DB, so pass it again or set ``study.sampler``
    afterwards (offset its seed by ``len(study.trials)`` to avoid repeating earlier random
    trials). Trials left RUNNING by a crashed run are marked FAIL and, if
    ``retry_interrupted``, their params are re-queued.
    Returns ``(study, local_db)``; after each trial call
    ``save_study_to_volume(local_db, volume_db)``.
    """
    volume_db = normalize_path(volume_db)
    if not volume_db.exists():
        raise FileNotFoundError(f"No saved study at {volume_db}")
    local_db = Path(local_db) if local_db is not None else default_local_dir() / f"{study_name}.db"
    storage = copy_study_to_local(volume_db, local_db)
    study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        # Private API, but the only way to close a trial whose process is gone.
        study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
        if retry_interrupted and t.params:
            study.enqueue_trial(t.params)  # no skip_if_exists: it would match the failed trial itself
    return study, local_db
