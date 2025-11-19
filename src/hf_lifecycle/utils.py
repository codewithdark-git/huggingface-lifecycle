"""
Utilities for HuggingFace Lifecycle Manager.
"""
import os
import time
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Callable, Any, TypeVar
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_file_checksum(file_path: str, algorithm: str = "sha256") -> str:
    """
    Calculate checksum of a file.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns:
        Hex digest of the file checksum.
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def get_disk_space(path: str = ".") -> dict:
    """
    Get disk space information.

    Args:
        path: Path to check disk space for.

    Returns:
        Dictionary with total, used, and free space in bytes.
    """
    stat = shutil.disk_usage(path)
    return {
        "total": stat.total,
        "used": stat.used,
        "free": stat.free,
        "percent_used": (stat.used / stat.total) * 100 if stat.total > 0 else 0,
    }


def check_disk_space(
    path: str = ".", required_bytes: int = 0, warn_threshold: float = 90.0
) -> bool:
    """
    Check if sufficient disk space is available.

    Args:
        path: Path to check.
        required_bytes: Minimum required free bytes.
        warn_threshold: Percentage usage to warn at.

    Returns:
        True if sufficient space, False otherwise.

    Raises:
        Warning if disk usage exceeds threshold.
    """
    space = get_disk_space(path)

    if space["free"] < required_bytes:
        logger.error(
            f"Insufficient disk space. Required: {required_bytes / 1e9:.2f}GB, "
            f"Available: {space['free'] / 1e9:.2f}GB"
        )
        return False

    if space["percent_used"] > warn_threshold:
        logger.warning(
            f"Disk usage at {space['percent_used']:.1f}% on {path}"
        )

    return True


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds.
        backoff_factor: Multiplier for delay after each retry.
        exceptions: Tuple of exceptions to catch and retry.

    Returns:
        Decorated function.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts"
                        )

            raise last_exception

        return wrapper

    return decorator


def get_device_info() -> dict:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device information.
    """
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_names": [],
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info["device_names"].append(torch.cuda.get_device_name(i))

    return info


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes to human-readable string.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Formatted string (e.g., "1.5 GB").
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time string.

    Args:
        seconds: Number of seconds.

    Returns:
        Formatted string (e.g., "1h 23m 45s").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {seconds:.0f}s"

    hours = minutes // 60
    minutes = minutes % 60

    return f"{hours}h {minutes}m {seconds:.0f}s"


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Operation", logger_func: Optional[Callable] = None):
        """
        Initialize timer.

        Args:
            name: Name of the operation being timed.
            logger_func: Function to log the result (defaults to logger.info).
        """
        self.name = name
        self.logger_func = logger_func or logger.info
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.logger_func(f"{self.name} took {format_time(elapsed)}")

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0
