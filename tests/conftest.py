"""Configuração comum dos testes."""

import pytest

from src import config as cfg


def pytest_configure(config):
    config.addinivalue_line("markers", "base: precisa da base de imagens em data/raw/")


def pytest_collection_modifyitems(items):
    """Testes marcados com `base` são pulados se a base ainda não foi baixada."""
    if cfg.PASTA_BASE.exists():
        return
    pular = pytest.mark.skip(reason="base não baixada: rode `uv run invoke dados`")
    for item in items:
        if "base" in item.keywords:
            item.add_marker(pular)
