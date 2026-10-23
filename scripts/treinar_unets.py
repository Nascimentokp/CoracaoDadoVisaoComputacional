"""Treina as U-Nets do notebook 10 fora do Jupyter, uma época por linha no terminal.

O treino leva mais de uma hora na CPU, mais que o limite por célula do
`invoke notebooks`. Rode isto antes; o notebook reaproveita os modelos salvos.
Modelos que já existem em `models/` são pulados.

    uv run python scripts/treinar_unets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras
import numpy as np

from src import config
from src.data.dados import carregar_imagens, dividir, listar_imagens
from src.model import cnn, pretreinada, unet
from src.utils.io import salvar_json
from src.visao import ict, segmentacao

PARAMS_ICT = dict(sigma_vertical=3, linhas=(50, 90), quantil=75)

catalogo = dividir(listar_imagens())
imagens = carregar_imagens(catalogo["caminho"])
conjunto = catalogo["conjunto"].to_numpy()
tr, va = conjunto == "treino", conjunto == "validacao"
X = cnn.tensores(imagens)


def rotulo_ict():
    medidas = [ict.medir_ict(im, **PARAMS_ICT) for im in imagens]
    return np.stack([segmentacao.mascara_pelo_ict(m) for m in medidas])


def rotulo_pspnet():
    prob = pretreinada.segmentar_com_cache(catalogo["id"].to_numpy(), imagens)
    return np.stack([segmentacao.maior_componente(m) for m in prob[:, 0] > 0.5])


for nome, rotulo in [("ict", rotulo_ict), ("pspnet", rotulo_pspnet)]:
    arquivo = config.MODELS / f"unet_{nome}.keras"
    if arquivo.exists():
        print(f"{arquivo.name} já existe, pulando", flush=True)
        continue
    M = rotulo()
    ok = M.any(axis=(1, 2))
    keras.utils.set_random_seed(config.SEMENTE)
    modelo = unet.construir_unet()
    print(f"treinando {arquivo.name}: {int((tr & ok).sum())} exames", flush=True)
    h = unet.treinar(modelo, X[tr & ok], M[tr & ok], X[va & ok], M[va & ok], verbose=2)
    modelo.save(arquivo)
    salvar_json(h.history, arquivo.with_suffix(".historico.json"))
    print(f"{arquivo.name} salvo", flush=True)
