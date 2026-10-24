"""Da tomografia ao raio-X: unidades Hounsfield, pulmão por densidade e projeção.

Exemplos de tomografia e ressonância do curso de segmentação. Na tomografia o
número do pixel tem **unidade física** (Hounsfield, HU): água ≈ 0, ar ≈ −1000,
osso > 300. Por isso um limiar fixo ("pulmão é de −950 a −400") vale em
qualquer exame. No nosso raio-X o pixel é um cinza de 0 a 255 sem unidade, e os
limiares precisam se adaptar a cada imagem (Otsu, meia altura do perfil).

Um raio-X é, fisicamente, a **soma das atenuações** ao longo do caminho do
feixe. Somando o volume da tomografia de frente para trás sai um **raio-X
sintético** (DRR, *digitally reconstructed radiograph*), e o pulmão marcado
por densidade em cada corte, projetado do mesmo jeito, vira um gabarito
físico para os métodos 2D do projeto.

Orientação: o volume está em LPS (eixo 0 cresce para a esquerda do paciente,
eixo 1 para as costas, eixo 2 para cima). A imagem sai na convenção
radiológica: cabeça em cima, lado direito do paciente à esquerda de quem olha.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from src import config


def janela(hu: np.ndarray, centro: float, largura: float) -> np.ndarray:
    """Janela de visualização: espalha [centro ± largura/2] em [0, 1]."""
    baixo = centro - largura / 2
    return np.clip((hu - baixo) / largura, 0, 1)


def pulmao_no_corte(corte: np.ndarray) -> np.ndarray:
    """Pulmão de um corte axial pela densidade: ar (−950 a −400 HU) dentro do corpo.

    O corpo é o que passa de −300 HU, com os buracos preenchidos; o ar que
    encosta na borda da imagem é o de fora do corpo e sai.
    """
    corpo = ndi.binary_fill_holes(corte > -300)
    pulmao = corpo & (corte > -950) & (corte < -400)
    pulmao = ndi.binary_opening(pulmao, iterations=1)
    rotulos, _ = ndi.label(pulmao)
    borda = np.unique(np.concatenate([rotulos[0], rotulos[-1], rotulos[:, 0], rotulos[:, -1]]))
    return pulmao & ~np.isin(rotulos, borda[borda > 0])


def ar_no_volume(volume: np.ndarray) -> np.ndarray:
    """`pulmao_no_corte` em cada corte axial (eixo 2): todo o ar dentro do corpo, gás incluído."""
    return np.stack([pulmao_no_corte(volume[:, :, z]) for z in range(volume.shape[2])], axis=2)


def pulmao_no_volume(
    volume: np.ndarray, fracao_minima: float = 0.1, ar: np.ndarray | None = None
) -> np.ndarray:
    """O ar de `ar_no_volume`, só com os pedaços 3D que são pulmão.

    Gás no estômago e no intestino também é ar dentro do corpo. Em 3D ele fica
    em pedaços separados e **não sobe** até a metade de cima do volume, aonde
    os pulmões chegam. Ficam os pedaços com pelo menos `fracao_minima` do maior
    e com algum voxel na metade de cima (eixo 2 cresce para a cabeça). Passe
    `ar` se ele já foi calculado.
    """
    ar = ar_no_volume(volume) if ar is None else ar
    rotulos, n = ndi.label(ar)
    if n == 0:
        return ar
    tamanhos = np.bincount(rotulos.ravel())[1:]
    no_alto = np.unique(rotulos[:, :, volume.shape[2] // 2 :])
    manter = [
        k + 1
        for k, t in enumerate(tamanhos)
        if t >= fracao_minima * tamanhos.max() and (k + 1) in no_alto
    ]
    return np.isin(rotulos, manter)


def ler_ct(caminho) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Volume em HU (float32) e o espaçamento (mm) de um NRRD em LPS sem eixos invertidos.

    O resto do módulo supõe eixo 0 → esquerda, 1 → costas e 2 → cabeça. Um
    volume girado ou com eixo invertido é recusado em vez de dar uma
    projeção espelhada ou larguras negativas.
    """
    import nrrd

    volume, cabecalho = nrrd.read(str(caminho))
    direcoes = np.asarray(cabecalho["space directions"], float)
    if cabecalho.get("space") != "left-posterior-superior":
        raise ValueError(f"orientação {cabecalho.get('space')!r}, esperada LPS")
    if not np.allclose(direcoes, np.diag(np.diag(direcoes))) or (np.diag(direcoes) <= 0).any():
        raise ValueError("eixos girados ou invertidos: reoriente o volume para LPS antes")
    espacamento = tuple(float(v) for v in np.linalg.norm(direcoes, axis=1))
    return volume.astype(np.float32), espacamento


def atenuacao(volume: np.ndarray) -> np.ndarray:
    """Coeficiente de atenuação relativo à água: μ/μ_água = 1 + HU/1000, nunca negativo."""
    return np.clip(1.0 + volume / 1000.0, 0, None)


def projetar(volume: np.ndarray, espacamento: tuple[float, float, float]) -> np.ndarray:
    """Soma de frente para trás (eixo 1), em mm de água equivalente.

    Sai como (z, x) na convenção radiológica.
    """
    soma = volume.sum(axis=1) * espacamento[1]  # (x, z)
    return soma.T[::-1]  # linhas = de cima para baixo, colunas = direita → esquerda do paciente


def para_128(
    imagem: np.ndarray,
    espacamento_linha: float,
    espacamento_coluna: float,
    margem_mm: float = 60.0,
    interpolacao: int = cv2.INTER_AREA,
) -> tuple[np.ndarray, float]:
    """Leva a projeção a 128 × 128 sem distorcer, e devolve o tamanho do pixel final (mm).

    O campo de visão da tomografia corta o corpo dos lados, mas o paciente faz
    o exame de braços erguidos: fora dele, nas laterais, só há ar. A imagem
    ganha `margem_mm` de ar (zero) de cada lado, como o fundo de um raio-X.
    Depois vira um quadrado em milímetros: o lado maior ocupa os 128 pixels e o
    menor é completado com ar, onde a tomografia não cobriu. Assim um pixel
    tem o mesmo tamanho na horizontal e na vertical.
    """
    margem = round(margem_mm / espacamento_coluna)
    completa = np.pad(imagem.astype(np.float32), ((0, 0), (margem, margem)))
    altura_mm = completa.shape[0] * espacamento_linha
    largura_mm = completa.shape[1] * espacamento_coluna
    mm_por_px = max(altura_mm, largura_mm) / config.TAMANHO
    altura, largura = round(altura_mm / mm_por_px), round(largura_mm / mm_por_px)
    menor = cv2.resize(completa, (largura, altura), interpolation=interpolacao)
    saida = np.zeros((config.TAMANHO, config.TAMANHO), np.float32)
    y0, x0 = (config.TAMANHO - altura) // 2, (config.TAMANHO - largura) // 2
    saida[y0 : y0 + altura, x0 : x0 + largura] = menor
    return saida, mm_por_px


def nitidez(imagem: np.ndarray, sigma: float = 8.0, forca: float = 0.8) -> np.ndarray:
    """Realce de borda (máscara de nitidez): tira uma parte do fundo borrado, só dentro do corpo.

    Tirar o fundo pode deixar o pulmão negativo. O resultado é deslocado para
    o corpo ficar todo acima de zero: zero continua sendo só o ar de fora, que
    `para_uint8` trata como fundo.
    """
    corpo = imagem > 0
    realcada = imagem - forca * cv2.GaussianBlur(imagem, (0, 0), sigma)
    if not corpo.any():
        return np.zeros_like(imagem)
    piso = realcada[corpo].min()
    return np.where(corpo, realcada - piso + 1e-3 * (realcada[corpo].max() - piso), 0)


def para_uint8(imagem: np.ndarray, gama: float = 1.0, percentis=(1, 99.5)) -> np.ndarray:
    """Janela pelos percentis do que não é fundo, com gama: o "revelado" do raio-X."""
    valores = imagem[imagem > 0]
    baixo, alto = np.percentile(valores, percentis) if valores.size else (0.0, 1.0)
    normal = np.clip((imagem - baixo) / max(alto - baixo, 1e-6), 0, 1)
    return np.round(255 * normal**gama).astype(np.uint8)
