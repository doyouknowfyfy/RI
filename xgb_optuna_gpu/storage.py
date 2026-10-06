"""Optuna storage that works on Databricks (and other FUSE / network file systems).

SQLite needs POSIX file locking and random writes. Databricks FUSE mounts (``/dbfs``,
``/Volumes``, ``/Workspace``) support neither, so ``create_study(storage="sqlite:////dbfs/...")``
fails with ``sqlite3.OperationalError: disk I/O error``.

``SyncedSQLiteStorage`` keeps the live database on the driver's local disk and copies a
consistent snapshot to the persistent path after every trial. When the cluster restarts
(local disk wiped), the snapshot is copied back and the study resumes.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

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
    # /local_disk0 is the driver's fast local SSD on Databricks.
    base = Path("/local_disk0/tmp") if Path("/local_disk0").is_dir() else Path(tempfile.gettempdir())
    return base / "optuna_local_db"


class SyncedSQLiteStorage:
    """SQLite on local disk, mirrored to ``persistent_path`` with :meth:`backup`.

    The persistent copy is the source of truth: on construction it is restored to local
    disk if it exists. If it does not exist, any stale local copy is removed so a deleted
    output directory really means a fresh study.
    """

    def __init__(self, persistent_path: str | Path, local_dir: str | Path | None = None):
        self.persistent_path = normalize_path(persistent_path)
        local_dir = Path(local_dir) if local_dir is not None else default_local_dir()
        local_dir.mkdir(parents=True, exist_ok=True)
        # Unique per persistent path, so several studies can share the local dir.
        key = hashlib.sha1(str(self.persistent_path.absolute()).encode()).hexdigest()[:10]
        self.local_path = local_dir / f"{self.persistent_path.stem}-{key}.db"
        self.restore()

    @property
    def url(self) -> str:
        return f"sqlite:///{self.local_path.absolute()}"

    def restore(self) -> None:
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(f"{self.local_path}{suffix}").unlink(missing_ok=True)
        if self.persistent_path.exists():
            shutil.copyfile(self.persistent_path, self.local_path)

    def backup(self) -> None:
        """Copy a consistent snapshot of the local DB to the persistent path."""
        if not self.local_path.exists():
            return
        snapshot = self.local_path.with_suffix(".snapshot")
        src = sqlite3.connect(self.local_path)
        dst = sqlite3.connect(snapshot)
        try:
            src.backup(dst)  # consistent even while Optuna's engine holds a connection
        finally:
            dst.close()
            src.close()
        self.persistent_path.parent.mkdir(parents=True, exist_ok=True)
        # Sequential whole-file write (supported on FUSE), then an atomic rename where possible
        # so a crash mid-copy never leaves a half-written study.db.
        tmp = self.persistent_path.with_name(self.persistent_path.name + ".tmp")
        shutil.copyfile(snapshot, tmp)
        try:
            os.replace(tmp, self.persistent_path)
        except OSError:
            shutil.copyfile(snapshot, self.persistent_path)
            tmp.unlink(missing_ok=True)
        snapshot.unlink(missing_ok=True)
