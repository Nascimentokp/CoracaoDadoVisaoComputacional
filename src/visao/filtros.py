"""Filtros à mão: kernel, suavização, realce, Sobel, Canny e CLAHE.

Aula "Filtros à mão". O filtro é um **kernel** que passa por cada pixel e troca o
valor dele por uma soma ponderada da vizinhança. `correlacionar` faz isso com
NumPy, sem OpenCV, e o teste confere que dá o mesmo resultado do `cv2.filter2D`.

Atenção ao nome: o OpenCV aplica o kernel **sem espelhar** (correlação). A
convolução matemática espelha o kernel antes. Para kernels simétricos (média,
gaussiano, realce) não faz diferença; no Sobel, espelhar só troca o sinal.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# --------------------------------------------------------------------------- #
# Kernels escritos à mão
# --------------------------------------------------------------------------- #

MEDIA_3 = np.full((3, 3), 1 / 9, dtype=np.float32)
GAUSS_3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
REALCE = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T.copy()

KERNELS = {
    "Média 3×3": MEDIA_3,
    "Gaussiano 3×3": GAUSS_3,
    "Realce": REALCE,
    "Sobel x": SOBEL_X,
    "Sobel y": SOBEL_Y,
}


def correlacionar(imagem: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Passa o kernel sobre a imagem, sem espelhar, como o `cv2.filter2D`.

    A borda é estendida por reflexão sem repetir o pixel da borda
    (`BORDER_REFLECT_101` do OpenCV, `mode="reflect"` do NumPy). Devolve
    `float32` sem cortar a faixa, para o Sobel manter valores negativos.
    """
    kh, kw = kernel.shape
    estendida = np.pad(
        imagem.astype(np.float32), ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="reflect"
    )
    janelas = sliding_window_view(estendida, (kh, kw))
    return np.einsum("ijkl,kl->ij", janelas, kernel.astype(np.float32))


def para_uint8(imagem: np.ndarray) -> np.ndarray:
    """Corta para 0–255 e arredonda, para exibir ou passar ao OpenCV."""
    return np.clip(np.rint(imagem), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Ruído e suavização
# --------------------------------------------------------------------------- #


def ruido_gaussiano(imagem: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Soma ruído normal de desvio `sigma` (o chiado do sensor)."""
    return para_uint8(imagem.astype(np.float32) + rng.normal(0, sigma, imagem.shape))


def ruido_sal_pimenta(imagem: np.ndarray, fracao: float, rng: np.random.Generator) -> np.ndarray:
    """Troca uma `fracao` dos pixels por preto ou branco (pixel morto, poeira)."""
    saida = imagem.copy()
    sorteio = rng.random(imagem.shape)
    saida[sorteio < fracao / 2] = 0
    saida[sorteio > 1 - fracao / 2] = 255
    return saida


SUAVIZADORES = {
    "Média 3×3": lambda img: cv2.blur(img, (3, 3)),
    "Gaussiano 5×5": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
    "Mediana 3×3": lambda img: cv2.medianBlur(img, 3),
}


def psnr(referencia: np.ndarray, imagem: np.ndarray) -> float:
    """Relação sinal-ruído de pico, em dB. Quanto maior, mais perto da referência."""
    erro = np.mean((referencia.astype(np.float64) - imagem.astype(np.float64)) ** 2)
    return float("inf") if erro == 0 else float(10 * np.log10(255**2 / erro))


# --------------------------------------------------------------------------- #
# Bordas
# --------------------------------------------------------------------------- #


def sobel(imagem: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradientes horizontal e vertical e a magnitude, em `float32`."""
    gx = cv2.Sobel(imagem, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(imagem, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, np.hypot(gx, gy)


def canny_auto(imagem: np.ndarray, sigma: float = 0.33) -> tuple[np.ndarray, tuple[int, int]]:
    """Canny com os dois limiares tirados da mediana da imagem suavizada.

    Regra prática: limiares em `(1 ± sigma) × mediana`. Evita escolher na mão
    um par de limiares para cada raio-X, que varia de exposição.
    """
    suave = cv2.GaussianBlur(imagem, (5, 5), 0)
    mediana = float(np.median(suave))
    baixo = int(max(0, (1 - sigma) * mediana))
    alto = int(min(255, (1 + sigma) * mediana))
    return cv2.Canny(suave, baixo, alto), (baixo, alto)


# --------------------------------------------------------------------------- #
# Contraste
# --------------------------------------------------------------------------- #


def clahe(imagem: np.ndarray, limite: float = 2.0, grade: int = 8) -> np.ndarray:
    """Equalização adaptativa por blocos, com limite de contraste (CLAHE)."""
    return cv2.createCLAHE(clipLimit=limite, tileGridSize=(grade, grade)).apply(imagem)


def preprocessar(imagem: np.ndarray) -> np.ndarray:
    """Pré-processamento padrão do projeto: tira o chiado e equaliza o contraste.

    É a receita do exemplo de ultrassom da aula (gaussiano → CLAHE), sem a
    equalização global no fim: ela padroniza o brilho entre exames, mas lava o
    contraste dentro do tórax (notebook 02).
    """
    return clahe(cv2.GaussianBlur(imagem, (3, 3), 0))
