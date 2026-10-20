"""Máscaras do coração: o contorno marcado à mão e três métodos automáticos.

Aula de segmentação: "tesoura em vez de moldura" — contorno real, não caixa
rígida — e "Dice: acordo de área, não de caixa". O gabarito é o contorno que a
equipe marcou na página de marcação (notebook 08); os métodos automáticos, do
mais simples ao mais informado:

1. **caixa fixa** — a mesma região dos notebooks 06 e 07, igual para todos os
   exames: a "moldura", a referência mais burra possível;
2. **limiar e morfologia** — Otsu calculado só na região central, o componente
   claro que contém o centro, abertura, fechamento e preenchimento de buracos;
3. **bordas do ICT** — em cada linha que o perfil do notebook 04 mediu,
   preenche da borda direita à borda esquerda do coração.

Os parâmetros foram fixados antes de comparar com o gabarito.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from src import config
from src.visao.ict import MedidaICT

LADO = config.TAMANHO


def poligono_para_mascara(pontos, lado: int = LADO) -> np.ndarray:
    """Máscara booleana do polígono (coordenadas em pixels da imagem, x e y)."""
    mascara = np.zeros((lado, lado), np.uint8)
    cv2.fillPoly(mascara, [np.round(np.asarray(pontos, float)).astype(np.int32)], 1)
    return mascara.astype(bool)


def caixa_para_mascara(caixa: tuple[int, int, int, int], lado: int = LADO) -> np.ndarray:
    x0, y0, x1, y1 = caixa
    mascara = np.zeros((lado, lado), bool)
    mascara[y0:y1, x0:x1] = True
    return mascara


def mascara_pelo_ict(medida: MedidaICT, lado: int = LADO) -> np.ndarray:
    """Em cada linha medida pelo perfil, preenche de uma borda do coração à outra."""
    mascara = np.zeros((lado, lado), bool)
    for r, _, esquerda, direita, _ in medida.bordas:
        mascara[r, esquerda : direita + 1] = True
    return mascara


def mascara_por_limiar(
    imagem: np.ndarray,
    regiao: tuple[int, int, int, int] = (24, 45, 104, 100),
    abertura: int = 7,
    fechamento: int = 9,
) -> np.ndarray:
    """Coração como o componente claro central, por limiar e morfologia.

    O Otsu é calculado só com os pixels da `regiao` (x0, y0, x1, y1): no
    raio-X inteiro o limiar separaria corpo de fundo, não coração de pulmão.
    A abertura solta o coração da coluna e das costelas; fica o componente
    que contém o centro da região (ou o maior, se o centro cair fora); o
    fechamento e o preenchimento tapam os buracos.
    """
    x0, y0, x1, y1 = regiao
    suave = cv2.GaussianBlur(imagem, (5, 5), 0)
    dentro = np.zeros(imagem.shape, bool)
    dentro[y0:y1, x0:x1] = True
    limiar, _ = cv2.threshold(
        suave[dentro].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    claro = ((suave > limiar) & dentro).astype(np.uint8)
    elipse = cv2.getStructuringElement
    claro = cv2.morphologyEx(claro, cv2.MORPH_OPEN, elipse(cv2.MORPH_ELLIPSE, (abertura,) * 2))
    n, rotulos = cv2.connectedComponents(claro)
    if n < 2:
        return np.zeros(imagem.shape, bool)
    centro = rotulos[(y0 + y1) // 2, (x0 + x1) // 2]
    alvo = centro if centro else 1 + int(np.argmax(np.bincount(rotulos.ravel())[1:]))
    mascara = (rotulos == alvo).astype(np.uint8)
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, elipse(cv2.MORPH_ELLIPSE, (fechamento,) * 2)
    )
    return ndi.binary_fill_holes(mascara)


def largura_maxima(mascara: np.ndarray) -> int:
    """Maior extensão horizontal da máscara numa mesma linha, em pixels."""
    larguras = [int(np.ptp(np.where(linha)[0])) + 1 for linha in mascara if linha.any()]
    return max(larguras, default=0)


def carregar_marcacoes(pasta: Path | None = None) -> pd.DataFrame:
    """Contornos exportados da página de marcação: uma linha por (anotador, exame).

    Cada arquivo `<id do anotador>.json` em `pasta` tem `poligonos`: {id do
    exame: [[x, y], ...]}. O id do anotador é opaco; o notebook só o usa para
    agrupar.
    """
    pasta = pasta or config.DATA_PROCESSED / "marcacoes"
    linhas = []
    for arquivo in sorted(pasta.glob("*.json")):
        conteudo = json.loads(arquivo.read_text(encoding="utf-8"))
        conteudo = conteudo.get("data", conteudo)
        for exame, pontos in conteudo.get("poligonos", {}).items():
            if len(pontos) >= 3:
                linhas.append({"anotador": arquivo.stem, "id": exame, "pontos": pontos})
    return pd.DataFrame(linhas, columns=["anotador", "id", "pontos"])
