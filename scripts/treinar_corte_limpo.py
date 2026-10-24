"""Treina a CNN pequena do notebook 05 no corte oficial e no corte sem paciente do teste.

O 0,802 do notebook 05 é a CNN pequena (raio-X cru, 150 épocas, paciência 15)
no corte oficial do NIH. Aqui a mesma receita é treinada também no corte em que
todo exame de treino ou validação do mesmo grupo de algum exame do teste sai
(`sem_paciente_do_teste`). O teste é o oficial nos dois cortes. Cada treino
leva ≈ 27 min na CPU; as previsões no teste vão para
`data/processed/corte_limpo.csv` a cada treino, e rodadas já gravadas são
puladas. O notebook 06 lê esse arquivo.

    uv run python scripts/treinar_corte_limpo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras
import numpy as np
import pandas as pd

from src import config
from src.data.dados import carregar_imagens, dividir, listar_imagens, sem_paciente_do_teste
from src.model import cnn

ARQUIVO = config.DATA_PROCESSED / "corte_limpo.csv"
#: O corte original com a semente do notebook 05 já existe (models/cnn_pequena_crua.keras).
RODADAS = [
    ("sem paciente do teste", config.SEMENTE),
    ("sem paciente do teste", 7),
    ("sem paciente do teste", 2026),
    ("oficial", 7),
    ("oficial", 2026),
]

catalogo = dividir(listar_imagens())
y = catalogo["classe"].to_numpy()
X = cnn.tensores(carregar_imagens(catalogo["caminho"]))
grupos = pd.read_csv(config.DATA_PROCESSED / "grupos.csv", sep=";").set_index("id")
grupos = grupos.loc[catalogo["id"], "grupo"].to_numpy()
cortes = {
    "oficial": catalogo["conjunto"],
    "sem paciente do teste": sem_paciente_do_teste(catalogo["conjunto"], grupos),
}
te = (catalogo["conjunto"] == "teste").to_numpy()

feitas = pd.read_csv(ARQUIVO, sep=";") if ARQUIVO.exists() else pd.DataFrame()
for corte, semente in RODADAS:
    if len(feitas) and ((feitas["corte"] == corte) & (feitas["semente"] == semente)).any():
        print(f"{corte} · semente {semente}: já feita, pulando", flush=True)
        continue
    conjunto = cortes[corte].to_numpy()
    tr, va = conjunto == "treino", conjunto == "validacao"
    print(f"{corte} · semente {semente}: {tr.sum()} treino, {va.sum()} validação", flush=True)
    keras.utils.set_random_seed(semente)
    modelo = cnn.construir_cnn()
    h = cnn.treinar(modelo, X[tr], y[tr], X[va], y[va], epocas=150, paciencia=15, verbose=2)
    rodada = pd.DataFrame(
        {
            "corte": corte,
            "semente": semente,
            "id": catalogo["id"].to_numpy()[te],
            "escore": cnn.prever(modelo, X[te]),
            "epocas": len(h.history["loss"]),
            "melhor_val_auc": float(np.max(h.history["val_auc"])),
        }
    )
    feitas = pd.concat([feitas, rodada], ignore_index=True)
    feitas.to_csv(ARQUIVO, sep=";", index=False)
    print(f"{corte} · semente {semente}: gravada", flush=True)
