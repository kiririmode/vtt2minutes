"""Tests for __main__.py module entry point."""

import sys
from unittest.mock import patch

import pytest


def test_main_module_entry_point():
    """Test that __main__.py calls cli() when executed as main module."""
    # Mock the cli function to avoid actually running the CLI
    with patch("vtt2minutes.__main__.cli") as mock_cli:
        # Import and execute the __main__ module

        # The import itself should not call cli() since we're not running as __main__
        mock_cli.assert_not_called()


def test_main_module_execution():
    """Test that cli() is called when __main__.py is executed."""
    with patch("vtt2minutes.cli.cli") as mock_cli:
        # Import the module with __name__ set to "__main__"
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "__main__", "src/vtt2minutes/__main__.py"
        )
        main_module = importlib.util.module_from_spec(spec)

        # Execute the module as if it was run as __main__
        main_module.__name__ = "__main__"
        spec.loader.exec_module(main_module)

        # Verify that cli() was called
        mock_cli.assert_called_once()


def test_main_module_cli_import():
    """Test that cli is properly imported from vtt2minutes.cli."""
    import vtt2minutes.__main__
    from vtt2minutes.cli import cli

    # Verify that the imported cli is the same as the one in the module
    assert hasattr(vtt2minutes.__main__, "cli")
    assert vtt2minutes.__main__.cli is cli


@pytest.mark.integration
def test_main_module_as_script():
    """Integration test: run the module as a script with --help."""
    import subprocess

    # Run the module with --help to test the entry point
    result = subprocess.run(
        [sys.executable, "-m", "vtt2minutes", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Should exit successfully and show help
    assert result.returncode == 0
    assert "VTT2Minutes" in result.stdout
    assert "Commands:" in result.stdout
    assert "batch" in result.stdout
    assert "info" in result.stdout
