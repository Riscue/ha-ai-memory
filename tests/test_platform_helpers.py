"""Test platform helpers."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pytest

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
    """Test getting memory managers when domain not in hass data."""
    hass.data = {}

    result = get_memory_managers(hass)
    assert result == {}


def test_ensure_directory_exists_new():
    """Test creating new directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        new_dir = os.path.join(temp_dir, "new_test_dir")
        assert not os.path.exists(new_dir)

        result = ensure_directory_exists(new_dir)

        assert result
        assert os.path.exists(new_dir)


def test_ensure_directory_exists_existing():
    """Test ensuring existing directory exists."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = ensure_directory_exists(temp_dir)
        assert result


def test_ensure_directory_exists_failure():
    """Test directory creation failure."""
    invalid_path = "/root/nonexistent/invalid/path"
    result = ensure_directory_exists(invalid_path)
    assert not result


def test_read_json_file_not_exists():
    """Test reading non-existent file."""
    result = read_json_file("/nonexistent/file.json")
    assert result == []


def test_read_json_file_valid():
    """Test reading valid JSON file."""
    test_data = [{"key": "value"}]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name

    try:
        result = read_json_file(temp_path)
        assert result == test_data
    finally:
        os.unlink(temp_path)


def test_read_json_file_invalid_json():
    """Test reading invalid JSON file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json content")
        temp_path = f.name

    try:
        result = read_json_file(temp_path)
        assert result == []
        assert os.path.exists(f"{temp_path}.bak")
    finally:
        for ext in ['', '.bak']:
            path = f"{temp_path}{ext}"
            if os.path.exists(path):
                os.unlink(path)


def test_read_json_file_invalid_format():
    """Test reading JSON file with invalid format (not a list)."""
    test_data = {"key": "value"}  # Dict instead of list

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name

    try:
        result = read_json_file(temp_path)
        assert result == []
    finally:
        os.unlink(temp_path)


def test_write_json_file_success():
    """Test successful JSON file write."""
    test_data = [{"key": "value"}]

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        result = write_json_file(temp_path, test_data)
        assert result

        # Verify content
        with open(temp_path, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data
    finally:
        os.unlink(temp_path)


def test_write_json_file_failure_temp_fallback():
    """Test JSON file write failure with temp fallback."""
    test_data = [{"key": "value"}]

    # Use invalid path to force failure
    invalid_path = "/root/nonexistent/file.json"

    result = write_json_file(invalid_path, test_data)
    assert not result


def test_backup_corrupted_file():
    """Test backing up corrupted file."""
    test_data = "test content"

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(test_data)
        temp_path = f.name

    try:
        backup_corrupted_file(temp_path)
        assert not os.path.exists(temp_path)
        assert os.path.exists(f"{temp_path}.bak")

        # Verify backup content
        with open(f"{temp_path}.bak", 'r') as f:
            backup_content = f.read()
        assert backup_content == test_data
    finally:
        for ext in ['', '.bak']:
            path = f"{temp_path}{ext}"
            if os.path.exists(path):
                os.unlink(path)


def test_backup_corrupted_file_not_exists():
    """Test backing up non-existent file."""
    backup_corrupted_file("/nonexistent/file.json")
    # Should not raise any exception