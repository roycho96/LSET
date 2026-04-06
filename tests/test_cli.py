"""Tests for lset CLI entrypoint."""

import subprocess
import sys


class TestCLI:
    """CLI command routing and error handling."""

    def test_no_args_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "lset.cli"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "LSET" in result.stdout
        assert "train" in result.stdout

    def test_unknown_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "lset.cli", "foobar"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "Unknown command" in result.stdout

    def test_train_missing_config(self):
        result = subprocess.run(
            [sys.executable, "-m", "lset.cli", "train"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "--config" in result.stdout

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "lset.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "LSET" in result.stdout
