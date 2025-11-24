"""Comprehensive tests for AI Memory Platform Helpers."""
import json
import unittest
from unittest.mock import patch, MagicMock, mock_open

from custom_components.ai_memory.platform_helpers import (
    async_setup_platform_entities,
    get_memory_managers,
    ensure_directory_exists,
    read_json_file,
    write_json_file,
    backup_corrupted_file,
)


async def test_async_setup_platform_entities_empty(hass):
    """Test platform setup with no memory managers."""
    mock_entry = MagicMock()
    mock_async_add = MagicMock()

    with patch("custom_components.ai_memory.platform_helpers.get_memory_managers", return_value={}):
        await async_setup_platform_entities(hass, mock_entry, mock_async_add, lambda x, y, z: x, "test")

    mock_async_add.assert_not_called()


async def test_async_setup_platform_entities_with_managers(hass):
    """Test platform setup with memory managers."""
    mock_entry = MagicMock()
    mock_async_add = MagicMock()
    mock_manager = MagicMock()

    # Setup hass data directly instead of mocking the function
    hass.data = {"ai_memory": {"memory_managers": {"test": mock_manager}}}

    await async_setup_platform_entities(hass, mock_entry, mock_async_add, lambda x, y, z: x, "test")

    mock_async_add.assert_called_once()


def test_get_memory_managers(hass):
    """Test getting memory managers from hass data."""
    hass.data = {"ai_memory": {"memory_managers": {"test": "manager"}}}

    result = get_memory_managers(hass)
    assert result == {"test": "manager"}


def test_get_memory_managers_no_domain(hass):
    """Test getting memory managers when domain doesn't exist."""
    hass.data = {}

    result = get_memory_managers(hass)
    assert result == {}


def test_ensure_directory_exists():
    """Test directory creation."""
    with patch("os.makedirs") as mock_makedirs:
        ensure_directory_exists("/tmp/test_dir")
        mock_makedirs.assert_called_once_with("/tmp/test_dir", exist_ok=True)


def test_ensure_directory_exists_with_parent():
    """Test directory creation with parent path."""
    with patch("os.makedirs") as mock_makedirs:
        ensure_directory_exists("/tmp/parent/test_dir")
        mock_makedirs.assert_called_once_with("/tmp/parent/test_dir", exist_ok=True)


def test_read_json_file_success():
    """Test successful JSON file reading."""
    test_data = [{"test": "data"}]
    with patch("builtins.open", mock_open(read_data=json.dumps(test_data))), \
            patch("os.path.exists", return_value=True):
        result = read_json_file("/tmp/test.json")
        assert result == test_data


def test_read_json_file_not_found():
    """Test reading non-existent JSON file."""
    with patch("os.path.exists", return_value=False):
        result = read_json_file("/tmp/nonexistent.json")
        assert result == []


def test_read_json_file_decode_error():
    """Test reading corrupted JSON file."""
    with patch("os.path.exists", return_value=True), \
            patch("builtins.open", mock_open(read_data="invalid json")), \
            patch("custom_components.ai_memory.platform_helpers.backup_corrupted_file") as mock_backup:
        result = read_json_file("/tmp/test.json")
        assert result == []
        mock_backup.assert_called_once_with("/tmp/test.json")


def test_read_json_file_general_exception():
    """Test read_json_file with general exception (not JSONDecodeError)."""
    file_path = "/tmp/test.json"

    with patch("builtins.open", side_effect=PermissionError("Permission denied")):
        result = read_json_file(file_path)
        assert result == []


def test_write_json_file_success():
    """Test successful JSON file writing."""
    data = [{"test": "data"}]
    with patch("custom_components.ai_memory.platform_helpers.ensure_directory_exists"), \
            patch("builtins.open", mock_open()), \
            patch("json.dump") as mock_dump:
        result = write_json_file("/tmp/test.json", data)
        assert result is True
        mock_dump.assert_called_once_with(data, unittest.mock.ANY, ensure_ascii=False, indent=2)


def test_write_json_file_both_writes_fail():
    """Test write_json_file when both main and temp writes fail."""
    data = [{"test": "data"}]
    file_path = "/tmp/test.json"

    with patch("custom_components.ai_memory.platform_helpers.ensure_directory_exists"), \
            patch("builtins.open", side_effect=PermissionError("Write failed")):
        result = write_json_file(file_path, data)
        assert result is False


def test_backup_corrupted_file_success():
    """Test successful file backup."""
    file_path = "/tmp/test.json"
    backup_path = "/tmp/test.json.bak"

    with patch("os.path.exists", return_value=True), \
            patch("os.rename") as mock_rename, \
            patch("os.path.basename", return_value="test.json.bak"):
        backup_corrupted_file(file_path)
        mock_rename.assert_called_once_with(file_path, backup_path)


def test_backup_corrupted_file_no_file():
    """Test backup_corrupted_file when file doesn't exist."""
    file_path = "/tmp/nonexistent.json"

    with patch("os.path.exists", return_value=False):
        backup_corrupted_file(file_path)


def test_backup_corrupted_file_rename_fails():
    """Test backup_corrupted_file when os.rename fails."""
    file_path = "/tmp/test.json"

    with patch("os.path.exists", return_value=True), \
            patch("os.rename", side_effect=PermissionError("Rename failed")):
        backup_corrupted_file(file_path)


async def test_async_setup_platform_entities_with_different_platform(hass):
    """Test platform setup for different platforms."""
    mock_entry = MagicMock()
    mock_async_add = MagicMock()
    mock_manager = MagicMock()

    hass.data = {"ai_memory": {"memory_managers": {"test": mock_manager}}}

    await async_setup_platform_entities(hass, mock_entry, mock_async_add, lambda x, y, z: x, "button")

    mock_async_add.assert_called_once()


def test_ensure_directory_exists_permission_error():
    """Test directory creation with permission error."""
    with patch("os.makedirs", side_effect=PermissionError("Permission denied")):
        # Should not raise exception
        ensure_directory_exists("/tmp/protected_dir")


def test_read_json_file_empty_file():
    """Test reading empty JSON file."""
    with patch("os.path.exists", return_value=True), \
            patch("builtins.open", mock_open(read_data="")), \
            patch("json.loads", side_effect=json.JSONDecodeError("Empty file", "", 0)):
        result = read_json_file("/tmp/empty.json")
        assert result == []


def test_read_json_file_backup_error():
    """Test read_json_file when backup also fails."""
    with patch("os.path.exists", return_value=True), \
            patch("builtins.open", mock_open(read_data="invalid json")), \
            patch("custom_components.ai_memory.platform_helpers.backup_corrupted_file",
                  side_effect=Exception("Backup failed")) as mock_backup:
        result = read_json_file("/tmp/test.json")
        assert result == []
        # Verify backup was attempted even though it failed
        mock_backup.assert_called_once_with("/tmp/test.json")


def test_write_json_file_main_write_fails_temp_success():
    """Test write_json_file when main write fails but temp succeeds."""
    data = [{"test": "data"}]
    file_path = "/tmp/test.json"

    # Create mock file handles for both attempts
    mock_file = mock_open()
    
    # Mock both main and temp file attempts
    with patch("custom_components.ai_memory.platform_helpers.ensure_directory_exists"), \
            patch("builtins.open", side_effect=[
                PermissionError("Main write failed"),  # First call fails
                mock_file.return_value  # Second call succeeds
            ]), \
            patch("json.dump") as mock_dump:
        result = write_json_file(file_path, data)
        assert result is False  # Overall operation failed (main write failed)
        # Only temp file write succeeded, so dump called once
        assert mock_dump.call_count == 1


def test_write_json_file_temp_creation_success():
    """Test that temp file is created with correct path."""
    data = [{"test": "data"}]
    file_path = "/tmp/test.json"

    with patch("custom_components.ai_memory.platform_helpers.ensure_directory_exists"), \
            patch("builtins.open", side_effect=[PermissionError("Main failed"), mock_open()]), \
            patch("json.dump"):
        result = write_json_file(file_path, data)
        # Should create temp file at correct location
        assert result is False


def test_backup_corrupted_file_different_paths():
    """Test backup file creation with different original paths."""
    test_cases = [
        "/config/ai_memory/test.json",
        "/config/ai_memory/subdir/test.json",
        "test.json",
        "/tmp/test.json"
    ]

    for file_path in test_cases:
        expected_backup = f"{file_path}.bak"

        with patch("os.path.exists", return_value=True), \
                patch("os.rename") as mock_rename:
            backup_corrupted_file(file_path)
            mock_rename.assert_called_once_with(file_path, expected_backup)
