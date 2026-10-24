"""Download, catálogo, carga e divisão da base de cardiomegalia.

A base vem do Zenodo já dividida em treino e teste (80/20) e equilibrada
(metade com cardiomegalia). Os arquivos se chamam só `<n>.png`: **não há ID
de paciente**, então o split por paciente da aula "o dado que mente" não sai
do nome do arquivo. O notebook 01 mede quanto isso pesa procurando raio-X
quase idênticos entre treino e teste (ver `src.visao.auditoria`).

Todo modelo do projeto — regra clássica, CNN ou zero-shot — recebe o mesmo
catálogo e a mesma divisão, para que a comparação entre eles seja justa.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src import config


def _hash(caminho: Path, algoritmo: str = "md5") -> str:
    h = hashlib.new(algoritmo)
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _baixar(url: str, destino: Path) -> Path:
    """Baixa para `<destino>.part` e só renomeia no fim: um download interrompido
    não deixa no lugar um arquivo pela metade que pareça completo."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".part")
    urllib.request.urlretrieve(url, parcial)
    parcial.replace(destino)
    return destino


def _conferir(caminho: Path, esperado: str, algoritmo: str) -> None:
    """Apaga o arquivo e avisa se o hash não bate, para a próxima chamada baixar de novo."""
    if _hash(caminho, algoritmo) != esperado:
        caminho.unlink()
        raise ValueError(
            f"{algoritmo.upper()} de {caminho.name} não confere: arquivo apagado, rode de novo."
        )


def baixar_base(forcar: bool = False) -> Path:
    """Baixa o zip do Zenodo, confere o MD5 e extrai em `data/raw/`.

    Não faz nada se a pasta já estiver extraída (a menos que `forcar=True`).
    A extração vai para uma pasta temporária e só é renomeada no fim: uma
    extração interrompida não deixa uma base pela metade que pareça completa.
    """
    if config.PASTA_BASE.exists() and not forcar:
        return config.PASTA_BASE
    if forcar or not config.ZIP_BASE.exists():
        _baixar(config.URL_BASE, config.ZIP_BASE)
    _conferir(config.ZIP_BASE, config.MD5_BASE, "md5")
    temporaria = config.DATA_RAW / "_extraindo"
    shutil.rmtree(temporaria, ignore_errors=True)
    with zipfile.ZipFile(config.ZIP_BASE) as zf:
        zf.extractall(temporaria)
    shutil.rmtree(config.PASTA_BASE, ignore_errors=True)
    (temporaria / config.PASTA_BASE.name).replace(config.PASTA_BASE)
    shutil.rmtree(temporaria, ignore_errors=True)
    return config.PASTA_BASE


def baixar_ecg(arquivos: tuple[str, ...] = (config.ECG_ESCANEADO, config.ECG_FOTO)) -> Path:
    """Baixa as amostras de ECG em papel do ECG-Image-Kit que ainda faltam."""
    config.PASTA_ECG.mkdir(parents=True, exist_ok=True)
    for nome in arquivos:
        destino = config.PASTA_ECG / nome
        if not destino.exists():
            _baixar(f"{config.URL_ECG}/{nome}", destino)
    return config.PASTA_ECG


def baixar_ct() -> Path:
    """Baixa a tomografia de tórax do 3D Slicer (se faltar) e confere o SHA-256 ao baixar."""
    if not config.ARQUIVO_CT.exists():
        _baixar(config.URL_CT, config.ARQUIVO_CT)
        _conferir(config.ARQUIVO_CT, config.SHA256_CT, "sha256")
    return config.ARQUIVO_CT


def carregar_ecg(nome: str) -> np.ndarray:
    """Lê uma amostra de ECG como imagem BGR (convenção do OpenCV)."""
    imagem = cv2.imread(str(baixar_ecg((nome,)) / nome))
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível ler {nome}.")
    return imagem


def listar_imagens() -> pd.DataFrame:
    """Catálogo da base: uma linha por imagem, em ordem estável.

    Colunas: `id` (`train/true/12`), `particao` (`train`/`test`), `classe`
    (1 = cardiomegalia, 0 = normal) e `caminho`.
    """
    linhas = []
    for particao in config.PARTICOES:
        for pasta, classe in config.CLASSES.items():
            arquivos = sorted(
                (config.PASTA_BASE / particao / pasta).glob("*.png"), key=lambda f: int(f.stem)
            )
            for arquivo in arquivos:
                linhas.append(
                    {
                        "id": f"{particao}/{pasta}/{arquivo.stem}",
                        "particao": particao,
                        "classe": classe,
                        "caminho": arquivo,
                    }
                )
    if not linhas:
        raise FileNotFoundError(
            f"Nenhuma imagem em {config.PASTA_BASE}. Rode `uv run invoke dados` primeiro."
        )
    return pd.DataFrame(linhas)


def carregar_imagens(caminhos) -> np.ndarray:
    """Lê os PNGs como matriz `uint8` de forma (n, 128, 128), um canal."""
    return np.stack([np.asarray(Image.open(c).convert("L"), dtype=np.uint8) for c in caminhos])


def dividir(catalogo: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a coluna `conjunto` (treino / validacao / teste).

    O teste é o da própria base. A validação sai do treino original,
    estratificada pela classe e com semente fixa. O teste só é tocado na
    avaliação final.
    """
    catalogo = catalogo.copy()
    catalogo["conjunto"] = catalogo["particao"].map({"train": "treino", "test": "teste"})
    treino = catalogo[catalogo["particao"] == "train"]
    _, validacao = train_test_split(
        treino.index,
        test_size=config.PROPORCAO_VALIDACAO,
        stratify=treino["classe"],
        random_state=config.SEMENTE,
    )
    catalogo.loc[validacao, "conjunto"] = "validacao"
    return catalogo


def dividir_por_grupo(
    catalogo: pd.DataFrame, grupos: np.ndarray, semente: int = config.SEMENTE
) -> pd.Series:
    """Nova divisão 60/20/20 da base inteira em que nenhum grupo fica partido.

    `grupos` vem do notebook 01 (`data/processed/grupos.csv`): raio-X quase
    idênticos, provavelmente do mesmo paciente, têm o mesmo grupo. Cada grupo
    cai inteiro em treino, validação ou teste, e as classes ficam equilibradas
    (StratifiedGroupKFold em 5 dobras: 3 treino, 1 validação, 1 teste).
    """
    dobras = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semente)
    dobra = np.empty(len(catalogo), dtype=int)
    for k, (_, idx) in enumerate(dobras.split(catalogo, catalogo["classe"], grupos)):
        dobra[idx] = k
    nomes = np.array(["treino", "treino", "treino", "validacao", "teste"])
    return pd.Series(nomes[dobra], index=catalogo.index, name="conjunto")


def dividir_aleatorio(catalogo: pd.DataFrame, semente: int = config.SEMENTE) -> pd.Series:
    """Divisão 60/20/20 da base inteira estratificada pela classe, ignorando grupos.

    É a referência "ingênua" contra `dividir_por_grupo`: mesmas proporções,
    mas dois exames do mesmo paciente podem cair em lados opostos.
    """
    resto, teste = train_test_split(
        catalogo.index, test_size=0.2, stratify=catalogo["classe"], random_state=semente
    )
    treino, validacao = train_test_split(
        resto, test_size=0.25, stratify=catalogo.loc[resto, "classe"], random_state=semente
    )
    conjunto = pd.Series("treino", index=catalogo.index, name="conjunto")
    conjunto[validacao] = "validacao"
    conjunto[teste] = "teste"
    return conjunto
