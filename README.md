# Coração de Dados — Visão Computacional
# EQUIPE: Adriano, Leonardo e João Paulo

Visão computacional para **detectar cardiomegalia** (coração aumentado) em raio-X de tórax.
É o terceiro projeto da série Coração de Dados: o Tema 6 agrupava pacientes pelo perfil
clínico, o projeto de Redes Neurais previa doença cardiovascular a partir desse perfil, e
aqui o mesmo coração passa a ser visto **pela imagem**.

O trabalho segue a ordem das aulas: abrir o lote, filtros, geometria, limiar e morfologia,
CNN, o dado que mente, mapas de ativação e segmentação. A visão clássica **mede** o coração
(índice cardiotorácico), a CNN **aprende** sozinha, e a auditoria mostra onde cada uma engana.

## Dados

**Cardiomegaly Chest X-Ray Image Dataset** ([Zenodo 17937122](https://zenodo.org/records/17937122),
CC BY 4.0), recorte do ChestX-ray14 do NIH Clinical Center.

| | Treino | Teste | Total |
|:--|--:|--:|--:|
| Cardiomegalia | 2.219 | 557 | 2.776 |
| Normal | 2.219 | 557 | 2.776 |
| **Total** | **4.438** | **1.114** | **5.552** |

Raio-X em tons de cinza, 128 × 128. O teste é o da própria base; 20 % do treino é separado,
estratificado, para validação. A base não vai para o git: `uv run invoke dados` baixa o zip,
confere o MD5 e extrai em `data/raw/`.

O notebook 03 usa duas amostras de ECG em papel do
[ECG-Image-Kit](https://github.com/alphanumericslab/ecg-image-kit) (licença BSD-3), do mesmo grupo
que montou a ECG-Image-Database: um escaneado limpo e uma foto de celular. Também são baixadas por
`uv run invoke dados`.

O notebook 12 usa a tomografia de tórax CTChest, amostra pública do
[3D Slicer](https://github.com/Slicer/SlicerTestingData) (a mesma dos exemplos do curso de segmentação),
baixada por `uv run invoke dados` e conferida pelo SHA-256.

O notebook 10 usa a PSPNet de segmentação anatômica do
[torchxrayvision](https://github.com/mlmed/torchxrayvision) (licença Apache-2.0), treinada no
[ChestX-Det](https://github.com/Deepwise-AILab/ChestX-Det-Dataset); o notebook 11, a DenseNet121 do
mesmo pacote treinada no [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/). Os pesos
são baixados na primeira execução.

**Limitações conhecidas** (medidas no notebook 01):

* os arquivos se chamam só `<n>.png`, sem ID de paciente. 76 raio-X de teste (6,8 %) têm um
  quase gêmeo no treino (correlação ≥ 0,92), em geral a mesma pessoa em outro exame;
* os rótulos vêm de laudos lidos por processamento de texto: mesmo entre pares quase idênticos,
  só 60–70 % têm o mesmo rótulo.

## Resultados até aqui

| Abordagem | AUC no teste | Observação |
|:--|--:|:--|
| DenseNet121 do CheXpert congelada + regressão logística (notebook 11) | **0,826** (0,801–0,848) | +0,031 sobre a MobileNetV2 com ajuste fino (IC 0,005–0,057); base treinada em raio-X de outro hospital |
| CNN pequena, raio-X pré-processado (notebook 05) | 0,809 (0,783–0,833) | 27 mil pesos; empata com a crua |
| CNN pequena, raio-X cru (notebook 05) | 0,802 (0,776–0,827) | +0,064 sobre o ICT (IC 0,030–0,097, bootstrap pareado) |
| ICT clássico (perfil por linha, notebook 04) | 0,739 (0,710–0,769) | Corte clínico 0,5: sensibilidade 54 %, especificidade 82 % |
| MobileNetV2 ImageNet + ajuste fino (notebook 05) | 0,795 (0,768–0,819) | Empata com a CNN pequena, com 80× mais pesos |
| ICT pela segmentação: coração da U-Net / pulmões da PSPNet (notebook 10) | 0,803 (0,776–0,830) | +0,064 sobre o ICT do perfil (IC 0,025–0,100, pareado); medida interpretável |

**O que a rede usa** (notebooks 06 e 07): com a região do coração tapada a AUC cai só de 0,695 para
0,657; o Grad-CAM mostra a borda esquerda do coração nas cardiomegalias, mas também atenção fora do
tórax. O vazamento por paciente não inflou a AUC (split aleatório 0,680 × por grupo 0,694).

**Segmentação** (notebook 08): contra 29 contornos marcados à mão, as bordas do ICT dão Dice 0,64
(IC 0,59–0,69), o limiar e morfologia 0,58 e a caixa fixa 0,51. Os contornos foram feitos na
[página de marcação](https://claude.ai/artifact/BgKdoMehNApFxrYfeobmE1) e exportados para
`data/processed/marcacoes/`.

## Roteiro

| Notebook | Aula | O que faz | Situação |
|:--|:--|:--|:--|
| `01-abrir-o-lote` | A imagem como dado | Shape, dtype, histograma por classe, brilho/contraste/nitidez, duplicatas e quase duplicatas entre treino e teste | ✅ |
| `02-filtros` | Filtros à mão | Kernel na mão conferido com o OpenCV, ruído × suavização (PSNR), Sobel no perfil do coração, Canny, CLAHE | ✅ |
| `03-geometria-ecg` | Geometria | ECG em papel: canais de cor e HSV, régua da grade (px/mm), frequência cardíaca na imagem, foto torta → Otsu + morfologia + 4 cantos + homografia, interpolações | ✅ |
| `04-limiar-morfologia` | Limiar e morfologia | **Índice cardiotorácico** sem rede: a receita Otsu + morfologia + componentes (e por que falha) contra o perfil por linha com limiar à meia altura. AUC 0,739 no teste | ✅ |
| `05-cnn` | CNN | CNN pequena em Keras (cru × pré-processado) contra MobileNetV2, filtros aprendidos comparados ao Sobel, mapas de ativação, teste pareado contra o ICT | ✅ |
| `06-o-dado-que-mente` | O dado que mente | Exames com gêmeo no treino, split aleatório × por grupo (3 sementes), coração tapado × só o coração | ✅ |
| `07-mapas-de-ativacao` | Mapa de ativação | Grad-CAM da classe prevista em acertos e erros; fração da atenção no coração, no tórax e fora dele | ✅ |
| `08-segmentacao` | Segmentação | Contorno do coração à mão (página de marcação) contra caixa fixa, limiar e morfologia e bordas do ICT: Dice, IoU, ICT do contorno e Grad-CAM dentro do coração | ✅ (1 anotador) |
| `09-deteccao-yolo` | Detecção (curso YOLO) | YOLOv8n treinado com **pseudo-rótulos** tirados do ICT; formato YOLO, confiança e NMS, IoU e mAP50 contra as caixas marcadas à mão, ICT da caixa | ✅ |
| `10-segmentacao-unet` | Segmentação com rede | PSPNet pré-treinada (torchxrayvision) e U-Net em Keras com perda Dice, treinada com pseudo-rótulos de dois professores; Dice, IoU, distância de borda e Hausdorff 95; melhores e piores casos; ICT do coração recortado | ✅ |
| `11-poucos-dados` | Transfer learning com poucas imagens (exercício) | Curva de aprendizado de 10 a 1.775 exames por classe: MobileNetV2, EfficientNetB0 e DenseNet121 do CheXpert congeladas contra a CNN do zero e os ICTs sem rótulo; a validação de 20 imagens; o efeito do espelhamento | ✅ |
| `12-raio-x-sintetico` | Apêndice: tomografia e ressonância (curso de segmentação) | Hounsfield, janelas e pulmão por densidade numa tomografia real; raio-X sintético somando o volume; largura do tórax com gabarito físico; o mesmo tórax em cinco revelados | ✅ |

**Detecção** (notebook 09): o YOLOv8n treinado com pseudo-rótulos do ICT reproduz o método clássico
(mAP50 0,91 contra os pseudo-rótulos), mas não o supera contra as caixas à mão (IoU 0,497 × 0,488;
mAP50 0,36). O ICT medido pela caixa separa pior que o do perfil (AUC 0,702 × 0,739).

**Segmentação com rede** (notebook 10): a U-Net treinada com pseudo-rótulos da PSPNet chega a Dice 0,69
(IC 0,61–0,76) e Hausdorff 95 de 11 px contra os contornos à mão (bordas do ICT: 0,64 e 16 px), e empata
com o próprio professor: o rótulo limita mais que a arquitetura. O ICT do coração recortado sobre os
pulmões segmentados dá AUC 0,803. A PSPNet foi treinada no ChestX-Det, outro recorte do ChestX-ray14:
parte do nosso teste pode ter estado no treino dela, então o Dice pode estar otimista.

**Poucos dados** (notebook 11): com 40 exames por classe, como no exercício, a DenseNet do CheXpert
congelada dá AUC 0,705 e as bases ImageNet 0,63; a CNN do zero fica perto do acaso até 100 por classe.
Nenhum aprendiz alcança o ICT da segmentação (0,803, sem rótulos) antes do treino inteiro. A acurácia
numa validação de 20 imagens varia de 25 % a 95 % entre sorteios da mesma receita. Espelhar o raio-X
tira 0,010 de AUC da DenseNet do CheXpert, que sabe de que lado fica o coração, e quase não mexe nas
bases ImageNet.

**Raio-X sintético** (notebook 12, apêndice): somando a tomografia CTChest sai um raio-X com largura
do tórax conhecida (29,9 cm). Mudando só o "revelado", a PSPNet erra −3 % a −5 % em quatro de cinco
versões e falha numa (−84 %); os métodos clássicos medem em duas ou três, com erros de −4 % a −67 %. O
pixel sem unidade faz os limiares dependerem do processamento do aparelho. O coração não sai por
densidade, e não há gabarito para ele.

Do curso de YOLO do professor, **vídeo e rastreamento** (aulas 2 e 3) ficam de fora: raio-X de tórax é
imagem parada. Detecção em imagem e treino com dataset próprio (aulas 1 e 4) estão no notebook 09.

## Como rodar

```bash
uv sync                    # ambiente (Python 3.12)
uv run invoke dados        # baixa e confere a base
uv run invoke testes       # critérios de aceitação (specs/01-cardiomegalia.md)
uv run python scripts/treinar_unets.py  # U-Nets do notebook 10 (≈ 1 h cada na CPU)
uv run invoke notebooks    # executa os notebooks e regrava reports/
uv run invoke lab          # JupyterLab
```

## Estrutura

```
src/
  config.py          caminhos, fonte dos dados, semente, paleta
  data/dados.py      download, catálogo, carga e divisão
  visao/auditoria.py qualidade, duplicatas e quase duplicatas
  visao/filtros.py   kernels à mão, ruído, Sobel, Canny, CLAHE e pré-processamento
  visao/geometria.py achar a folha, homografia, régua da grade, traçado e frequência cardíaca
  visao/metricas.py  Dice, IoU e distâncias de borda (média, Hausdorff 95)
  visao/ict.py       índice cardiotorácico: segmentação 2D e perfil por linha
  visao/segmentacao.py máscaras do coração, maior componente e leitura das marcações
  visao/deteccao.py  caixas, formato YOLO, IoU e pseudo-rótulos
  visao/tomografia.py Hounsfield, pulmão por densidade e raio-X sintético
  model/cnn.py       CNN pequena, MobileNetV2, aumento sem espelhar, treino
  model/avaliacao.py AUC com bootstrap, comparação pareada, limiar de Youden
  model/gradcam.py   Grad-CAM e atenção por região
  model/unet.py      U-Net em Keras e perda Dice
  model/pretreinada.py PSPNet do torchxrayvision (coração e pulmões), com cache
  model/transferencia.py bases congeladas (ImageNet e CheXpert), sorteio balanceado e cabeça
  utils/io.py        gravação padronizada de figuras e tabelas
notebooks/           um notebook por aula
reports/             figuras e as tabelas que as originaram
specs/               critérios de aceitação verificados pelos testes
tests/
```
