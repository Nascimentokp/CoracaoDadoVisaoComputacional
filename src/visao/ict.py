"""Índice cardiotorácico (ICT) com limiar e morfologia, sem rede neural.

O ICT é a largura máxima do coração dividida pela largura interna máxima do
tórax, no raio-X de frente. Acima de 0,5 o coração é considerado aumentado
(no raio-X PA; no AP, feito no leito, o coração sai ampliado e o critério
superestima).

Duas tentativas, as duas com o conteúdo das aulas:

1. **Segmentar os pulmões em 2D** (`segmentar_pulmoes` + `ict_pela_mascara`) —
   a receita "contar feijão": suavização → Otsu → abertura → componentes
   conexos → fechamento → preencher buracos. Funciona mal: as costelas cortam o
   pulmão em faixas e o pulmão vaza para o fundo pela axila. O notebook 04 mostra.
2. **Perfil por linha** (`medir_ict`) — o raciocínio do perfil de Sobel do
   notebook 02. Um desfoque forte só na vertical apaga as costelas; em cada
   linha da metade de baixo do tórax, o coração é o platô claro do meio, os
   pulmões são os vales escuros e a parede torácica é a subida nas laterais. As
   bordas ficam onde a intensidade cruza a **meia altura** entre platô e vale:
   um limiar por linha, que se adapta à exposição de cada exame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi

# --------------------------------------------------------------------------- #
# Tentativa 1: segmentar os pulmões em 2D
# --------------------------------------------------------------------------- #


def _elipse(lado: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))


def segmentar_pulmoes(
    imagem: np.ndarray, abertura: int = 3, fechamento: int = 7, area_minima: float = 0.01
) -> np.ndarray:
    """Máscara (0/255) dos dois pulmões pela receita de limiar e morfologia.

    Otsu separa o escuro (ar) do claro (osso, coração, partes moles). A
    abertura solta pontes finas; dos componentes escuros, descarta os que tocam
    as laterais ou a base (fundo fora do corpo) e fica com os dois maiores. O
    fechamento e o preenchimento tapam vasos e costelas dentro de cada pulmão.
    """
    altura, largura = imagem.shape
    suave = cv2.GaussianBlur(imagem, (5, 5), 0)
    _, escuro = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    escuro = cv2.morphologyEx(escuro, cv2.MORPH_OPEN, _elipse(abertura))
    n, rotulos, estat, _ = cv2.connectedComponentsWithStats(escuro, 8)
    candidatos = []
    for i in range(1, n):
        x, y, w, h, area = estat[i]
        toca_borda = x == 0 or x + w >= largura or y + h >= altura
        if not toca_borda and area >= area_minima * altura * largura:
            candidatos.append((area, i))
    mascara = np.zeros_like(escuro)
    for _, i in sorted(candidatos, reverse=True)[:2]:
        mascara[rotulos == i] = 255
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, _elipse(fechamento))
    return np.where(ndi.binary_fill_holes(mascara > 0), 255, 0).astype(np.uint8)


def ict_pela_mascara(mascara: np.ndarray, faixa: tuple[float, float] = (0.5, 0.95)) -> float:
    """ICT a partir da máscara dos pulmões, ou NaN se não houver dois pulmões.

    Tórax = distância entre as bordas externas dos pulmões; coração = maior vão
    entre as bordas internas, na metade de baixo da altura dos pulmões.
    """
    n, rotulos, estat, centros = cv2.connectedComponentsWithStats(mascara, 8)
    maiores = sorted(range(1, n), key=lambda i: -estat[i, cv2.CC_STAT_AREA])[:2]
    if len(maiores) < 2:
        return float("nan")
    esquerdo, direito = sorted(maiores, key=lambda i: centros[i][0])
    if estat[maiores[1], cv2.CC_STAT_AREA] < 0.25 * estat[maiores[0], cv2.CC_STAT_AREA]:
        return float("nan")
    linhas = np.where(mascara.any(axis=1))[0]
    topo, base = linhas.min(), linhas.max()
    colunas = np.where(mascara.any(axis=0))[0]
    torax = colunas.max() - colunas.min()
    vaos = []
    for r in range(int(topo + faixa[0] * (base - topo)), int(topo + faixa[1] * (base - topo))):
        e, d = np.where(rotulos[r] == esquerdo)[0], np.where(rotulos[r] == direito)[0]
        if len(e) and len(d):
            vaos.append(d.min() - e.max())
    return float(max(vaos) / torax) if vaos and torax > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Tentativa 2: perfil por linha
# --------------------------------------------------------------------------- #


@dataclass
class MedidaICT:
    """Resultado de `medir_ict`. `bordas` tem uma linha por linha da imagem medida:
    (linha, parede esquerda, coração esquerdo, coração direito, parede direita)."""

    ict: float
    coracao_px: float
    torax_px: float
    bordas: np.ndarray = field(repr=False)

    @property
    def ok(self) -> bool:
        return bool(np.isfinite(self.ict))


def _vale(perfil: np.ndarray, inicio: int, fim: int, passo: int, meio: float) -> int:
    """Ponto mais escuro caminhando do coração para fora, até a parede.

    Para quando a intensidade volta a subir acima da meia altura entre o platô
    do meio e o mínimo visto até ali: é a parede torácica. Assim o fundo escuro
    fora do corpo, depois da parede, nunca é tomado pelo pulmão.
    """
    pos, minimo = inicio, perfil[inicio]
    for x in range(inicio, fim, passo):
        if perfil[x] < minimo:
            pos, minimo = x, perfil[x]
        elif perfil[x] > (meio + minimo) / 2 and meio - minimo >= 15:
            break
    return pos


def medir_linha(
    perfil: np.ndarray,
    centro: int,
    margem: int = 8,
    lateral: float = 0.04,
    pulmao_minimo: float = 0.10,
    coracao_minimo: float = 0.15,
) -> tuple[int, ...] | None:
    """Bordas de uma linha: (parede esq., coração esq., coração dir., parede dir.).

    Platô do meio = pixel mais claro a até 20 px do centro (o paciente pode
    estar rodado); vale de cada pulmão = ponto mais escuro caminhando do platô
    para fora até a parede (`_vale`). Cada borda é o primeiro pixel que cruza a
    meia altura entre platô (ou parede) e vale.

    Devolve None se a linha não tem o desenho esperado ou se a medida é
    fisicamente impossível: cada campo pulmonar precisa ocupar ao menos
    `pulmao_minimo` do tórax (descarta as linhas do diafragma, onde os pulmões
    já acabaram) e o coração ao menos `coracao_minimo` (descarta linhas em que o
    "platô" é só a coluna vertebral ou um fio).
    """
    n = len(perfil)
    inicio, fim = int(lateral * n), int((1 - lateral) * n)
    pico = centro - 20 + int(np.argmax(perfil[centro - 20 : centro + 20]))
    if pico - margem <= inicio or pico + margem >= fim:
        return None
    meio = perfil[pico]
    vale_e = _vale(perfil, pico - margem, inicio - 1, -1, meio)
    vale_d = _vale(perfil, pico + margem, fim, 1, meio)
    if meio - max(perfil[vale_e], perfil[vale_d]) < 15:
        return None

    def cruza(de, ate, passo, nivel, acima):
        for x in range(de, ate, passo):
            if (perfil[x] > nivel) if acima else (perfil[x] < nivel):
                return x
        return None

    cor_e = cruza(pico, vale_e, -1, (meio + perfil[vale_e]) / 2, acima=False)
    cor_d = cruza(pico, vale_d, 1, (meio + perfil[vale_d]) / 2, acima=False)
    parede_e = cruza(vale_e, -1, -1, (perfil[: vale_e + 1].max() + perfil[vale_e]) / 2, acima=True)
    parede_d = cruza(vale_d, n, 1, (perfil[vale_d:].max() + perfil[vale_d]) / 2, acima=True)
    if None in (cor_e, cor_d, parede_e, parede_d):
        return None
    torax = parede_d - parede_e
    if torax <= 0:
        return None
    if min(cor_e - parede_e, parede_d - cor_d) < pulmao_minimo * torax:
        return None
    if cor_d - cor_e < coracao_minimo * torax:
        return None
    return parede_e, cor_e, cor_d, parede_d


def medir_ict(
    imagem: np.ndarray,
    sigma_vertical: float = 4.0,
    linhas: tuple[int, int] = (50, 90),
    quantil: float = 50,
    minimo_linhas: int = 5,
) -> MedidaICT:
    """ICT pelo perfil horizontal das linhas `linhas` (em pixels de uma imagem 128).

    Larguras de coração e tórax são o `quantil` das larguras por linha (50 =
    mediana, robusta a linhas ruins). Com menos de `minimo_linhas` linhas
    válidas, devolve ICT NaN.
    """
    suave = cv2.GaussianBlur(imagem.astype(np.float32), (0, 0), sigmaX=1.5, sigmaY=sigma_vertical)
    centro = imagem.shape[1] // 2
    bordas = []
    for r in range(*linhas):
        medida = medir_linha(suave[r], centro)
        if medida is not None:
            bordas.append((r, *medida))
    bordas = np.array(bordas, dtype=int).reshape(-1, 5)
    if len(bordas) < minimo_linhas:
        return MedidaICT(float("nan"), float("nan"), float("nan"), bordas)
    coracao = float(np.percentile(bordas[:, 3] - bordas[:, 2], quantil))
    torax = float(np.percentile(bordas[:, 4] - bordas[:, 1], quantil))
    return MedidaICT(coracao / torax, coracao, torax, bordas)


def caixas_de_referencia(
    medidas: list[MedidaICT], folga: int = 4
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Caixas fixas (x0, y0, x1, y1) do coração e do tórax, a partir de várias medidas.

    Usa as bordas de todas as linhas medidas: coração do percentil 10 da borda
    esquerda ao 90 da direita; tórax do percentil 10 da parede esquerda ao 90 da
    direita. Na vertical, a faixa de linhas medidas (coração) e dela até o topo
    dos pulmões (tórax, que começa 35 px acima). Servem como regiões **iguais
    para todos os exames** nos notebooks 06 e 07.
    """
    bordas = np.concatenate([m.bordas for m in medidas if m.ok])
    y0, y1 = int(bordas[:, 0].min()), int(bordas[:, 0].max()) + 1
    coracao = (
        int(np.percentile(bordas[:, 2], 10)) - folga,
        y0 - folga,
        int(np.percentile(bordas[:, 3], 90)) + folga,
        y1 + folga,
    )
    torax = (
        int(np.percentile(bordas[:, 1], 10)),
        max(0, y0 - 35),
        int(np.percentile(bordas[:, 4], 90)),
        y1 + folga,
    )
    return coracao, torax
