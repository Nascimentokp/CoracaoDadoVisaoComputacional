"""Treina as U-Nets do notebook 10 fora do Jupyter, uma época por linha no terminal.

O treino leva mais de uma hora na CPU, mais que o limite por célula do
`invoke notebooks`. Rode isto antes; o notebook reaproveita os modelos salvos.
Modelos que já existem em `models/` são pulados.

    uv run python scripts/treinar_unets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.data.dados import carregar_imagens, dividir, listar_imagens
from src.model import cnn, pretreinada, unet
from src.visao import ict, segmentacao

catalogo = dividir(listar_imagens())
imagens = carregar_imagens(catalogo["caminho"])
conjunto = catalogo["conjunto"].to_numpy()
tr, va = conjunto == "treino", conjunto == "validacao"
X = cnn.tensores(imagens)


def rotulo_ict():
    medidas = [ict.medir_ict(im, **config.PARAMS_ICT) for im in imagens]
    return np.stack([segmentacao.mascara_pelo_ict(m) for m in medidas])


def rotulo_pspnet():
    prob = pretreinada.segmentar_com_cache(catalogo["id"].to_numpy(), imagens)
    return np.stack([segmentacao.maior_componente(m) for m in prob[:, 0] > 0.5])


for nome, rotulo in [("ict", rotulo_ict), ("pspnet", rotulo_pspnet)]:
    arquivo = config.MODELS / f"unet_{nome}.keras"
    if arquivo.exists():
        print(f"{arquivo.name} já existe, pulando", flush=True)
        continue
    print(f"treinando {arquivo.name}", flush=True)
    unet.treinar_ou_carregar(arquivo, X, rotulo(), tr, va, verbose=2)
    print(f"{arquivo.name} salvo", flush=True)
