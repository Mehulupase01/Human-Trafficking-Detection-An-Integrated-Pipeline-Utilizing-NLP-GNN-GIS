import os
import shutil
import tempfile
import pytest

@pytest.fixture(autouse=True)
def temp_registry_env(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="appdata_")
    monkeypatch.setenv("APP_DATA_DIR", tmp)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)
