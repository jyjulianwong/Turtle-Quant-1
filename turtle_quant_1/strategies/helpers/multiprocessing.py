import atexit
import contextlib
import fcntl
import hashlib
import logging
import os
import pickle
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CACHE_DIR_PATH = Path(os.path.join(tempfile.gettempdir(), "turtle-quant-1"))


class ProcessSafeCache:
    """Process-safe cache using file-based storage with proper locking."""

    REF_COUNT_FILE_PATH = CACHE_DIR_PATH / "cache_rc.txt"

    def __init__(self):
        """Initialize the cache with a directory for storage."""
        self.cache_dir_path = CACHE_DIR_PATH
        self.cache_dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Using temporary cache directory '{self.cache_dir_path}'...")

        # Use file-based locking for cross-process synchronization
        self.lock_file = self.cache_dir_path / "cache.lock"
        self._local_cache = {}  # In-memory cache for performance
        self._local_lock = threading.Lock()  # Thread-level locking

        # Increment reference count and clear if first
        with self._global_lock():
            ref_count = self._read_ref_count()
            if ref_count == 0:
                self._clear_cache_dir()
                logger.debug(f"Cleared cache directory '{self.cache_dir_path}'")
            self._write_ref_count(ref_count + 1)
            logger.debug(f"Incremented reference count to {ref_count + 1}")

        atexit.register(self._cleanup)

    def _cleanup(self):
        try:
            with self._global_lock():
                ref_count = self._read_ref_count()
                if ref_count > 0:
                    ref_count -= 1
                    self._write_ref_count(ref_count)
                    logger.debug(f"Decremented reference count to {ref_count}")
                if ref_count == 0:
                    self._clear_cache_dir()
                    logger.debug(f"Cleared cache directory '{self.cache_dir_path}'")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    # ---------------------------------
    # Global lock context manager
    # ---------------------------------

    @contextlib.contextmanager
    def _global_lock(self):
        fd = self._acquire_file_lock()
        try:
            yield
        finally:
            self._release_file_lock(fd)

    def _read_ref_count(self):
        if not self.REF_COUNT_FILE_PATH.exists():
            return 0
        try:
            return int(self.REF_COUNT_FILE_PATH.read_text().strip())
        except Exception:
            return 0

    def _write_ref_count(self, count):
        self.REF_COUNT_FILE_PATH.write_text(str(count))

    def _clear_cache_dir(self):
        size = 0
        for file in self.cache_dir_path.glob("cache_*.pkl"):
            size += file.stat().st_size
            file.unlink()
        logger.debug(
            f"Cleared cache in '{self.cache_dir_path}' ({size / 1024 / 1024:.2f} MB)"
        )

    # ---------------------------------
    # File-based lock methods
    # ---------------------------------

    def _get_cache_file_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Use hash to create safe filenames
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir_path / f"cache_{key_hash}.pkl"

    def _acquire_file_lock(self, timeout: float = 10.0):
        """Acquire a file-based lock for cross-process synchronization."""
        lock_fd = os.open(str(self.lock_file), os.O_CREAT | os.O_WRONLY)

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_fd
            except (OSError, IOError):
                time.sleep(0.01)  # Wait 10ms before retrying

        os.close(lock_fd)
        raise TimeoutError("Could not acquire file lock")

    def _release_file_lock(self, lock_fd: int):
        """Release the file-based lock."""
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        except (OSError, IOError):
            pass  # Ignore errors on release

    # ---------------------------------
    # API methods
    # ---------------------------------

    def get(self, key: str):
        """Get a value from the cache."""
        # First check local cache
        with self._local_lock:
            if key in self._local_cache:
                return self._local_cache[key]

        # Check file cache
        cache_file = self._get_cache_file_path(key)
        if cache_file.exists():
            lock_fd = None
            try:
                lock_fd = self._acquire_file_lock()
                with open(cache_file, "rb") as f:
                    value = pickle.load(f)

                # Update local cache
                with self._local_lock:
                    self._local_cache[key] = value

                return value
            except Exception:
                return None
            finally:
                if lock_fd is not None:
                    self._release_file_lock(lock_fd)

        return None

    def set(self, key: str, value: Any):
        """Set a value in the cache."""
        cache_file = self._get_cache_file_path(key)
        lock_fd = None

        try:
            lock_fd = self._acquire_file_lock()

            # Write to file
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)

            # Update local cache
            with self._local_lock:
                self._local_cache[key] = value

        except Exception as e:
            logger.error(f"Failed to cache data for key {key}: {e}")
        finally:
            if lock_fd is not None:
                self._release_file_lock(lock_fd)

    def contains(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        # Check local cache first
        with self._local_lock:
            if key in self._local_cache:
                return True

        # Check file cache
        cache_file = self._get_cache_file_path(key)
        return cache_file.exists()
