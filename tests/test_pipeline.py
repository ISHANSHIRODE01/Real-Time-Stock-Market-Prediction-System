import pytest
import os

def test_directories():
    """Verify production structure."""
    dirs = ['src/data_pipeline', 'src/models', 'api', 'dashboard', 'data/raw']
    for d in dirs:
        assert os.path.exists(d), f"Missing directory: {d}"

def test_requirements():
    """Verify dependencies are defined."""
    assert os.path.exists('requirements.txt')
