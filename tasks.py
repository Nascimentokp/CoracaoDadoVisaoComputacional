"""Tarefas do projeto executadas com o invoke.

Liste as tarefas disponíveis com `uv run invoke --list`.
"""

from invoke import task

NOTEBOOKS = [
    "01-abrir-o-lote",
    "02-filtros",
    "03-geometria-ecg",
    "04-limiar-morfologia",
    "05-cnn",
    "06-o-dado-que-mente",
    "07-mapas-de-ativacao",
    "08-segmentacao",
    "09-deteccao-yolo",
    "10-segmentacao-unet",
    "11-poucos-dados",
]


@task
def dados(c):
    """Baixa a base do Zenodo (confere o MD5) e as amostras de ECG em data/raw/."""
    from src.data.dados import baixar_base, baixar_ecg

    print("Base em", baixar_base())
    print("ECG em", baixar_ecg())


@task
def lab(c):
    """Abre o JupyterLab."""
    c.run("jupyter lab")


@task
def notebooks(c):
    """Executa todos os notebooks em ordem, regravando figuras e tabelas em reports/."""
    for nome in NOTEBOOKS:
        c.run(
            f"jupyter nbconvert --to notebook --execute --inplace "
            f"--ExecutePreprocessor.timeout=7200 notebooks/{nome}.ipynb"
        )


@task
def lint(c):
    """Verifica o código com o ruff (lint e formatação)."""
    c.run("ruff check src tests tasks.py")
    c.run("ruff format --check src tests tasks.py")


@task
def testes(c):
    """Roda os testes de aceitação."""
    c.run("pytest -q")
