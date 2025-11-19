import pytest
import time
import tempfile
from pathlib import Path
from hf_lifecycle.utils import (
    get_file_checksum,
    get_disk_space,
    check_disk_space,
    retry_with_backoff,
    get_device_info,
    format_bytes,
    format_time,
    Timer,
)


class TestUtils:
    def test_file_checksum(self, tmp_path):
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")
        
        # Calculate checksum
        checksum = get_file_checksum(str(test_file))
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 produces 64 hex characters

    def test_disk_space(self):
        space = get_disk_space()
        assert "total" in space
        assert "used" in space
        assert "free" in space
        assert "percent_used" in space
        assert space["total"] > 0

    def test_check_disk_space_sufficient(self):
        result = check_disk_space(required_bytes=1000)  # 1KB
        assert result is True

    def test_check_disk_space_insufficient(self):
        result = check_disk_space(required_bytes=10**15)  # 1PB
        assert result is False

    def test_retry_with_backoff_success(self):
        counter = {"calls": 0}
        
        @retry_with_backoff(max_retries=3)
        def flaky_function():
            counter["calls"] += 1
            if counter["calls"] < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert counter["calls"] == 2

    def test_retry_with_backoff_failure(self):
        @retry_with_backoff(max_retries=2, initial_delay=0.1)
        def always_fail():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fail()

    def test_get_device_info(self):
        info = get_device_info()
        assert "cuda_available" in info
        assert "cuda_device_count" in info
        assert "device_names" in info

    def test_format_bytes(self):
        assert format_bytes(500) == "500.0 B"
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"

    def test_format_time(self):
        assert "s" in format_time(30)
        assert "m" in format_time(90)
        assert "h" in format_time(3700)

    def test_timer(self):
        with Timer("test") as timer:
            time.sleep(0.1)
        
        assert timer.elapsed >= 0.1
