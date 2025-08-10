import fcntl
import hashlib
import logging
import os
import pickle
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ProcessSafeCache:
    """Process-safe cache using file-based storage with proper locking."""

    def __init__(self, cache_dir_path: str | None = None):
        """Initialize the cache with a directory for storage."""
        if cache_dir_path is None:
            # Use a temporary directory that's shared across processes
            cache_dir_path = os.path.join(tempfile.gettempdir(), "turtle-quant-1")
            logger.info(f"Created temporary cache directory '{cache_dir_path}'")

        self.cache_dir_path = Path(cache_dir_path)
        self.cache_dir_path.mkdir(parents=True, exist_ok=True)

        # Use file-based locking for cross-process synchronization
        self.lock_file = self.cache_dir_path / "cache.lock"
        self._local_cache = {}  # In-memory cache for performance
        self._local_lock = threading.Lock()  # Thread-level locking

    def __del__(self):
        """Delete the cache directory."""
        # if self.cache_dir_path.exists():
        #     shutil.rmtree(self.cache_dir_path)
        #     logger.info(f"Deleted cache directory '{self.cache_dir_path}'")
        pass

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

    def set(self, key: str, value):
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
