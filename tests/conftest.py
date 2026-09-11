import pytest

from kerr_sbi.config import Config, load_config


@pytest.fixture
def cfg() -> Config:
    return load_config()
