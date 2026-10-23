# Especificação: cardiomegalia em raio-X de tórax

## Objetivo e escopo
Mostrar, com o conteúdo das aulas de Visão Computacional, como medir e detectar
cardiomegalia em raio-X de tórax 128 × 128. É um trabalho de disciplina: não faz
diagnóstico e não substitui a leitura do radiologista.

## Critérios de aceitação

### Dados
- **DADOS-01**: o catálogo tem 5.552 imagens (4.438 treino, 1.114 teste), classes
  equilibradas em treino, validação e teste; a validação sai só do treino, com semente
  fixa; o teste da base fica intocado. (`tests/test_dados.py`)

### Auditoria do lote
- **AUD-01**: brilho, contraste e nitidez são calculados por imagem; duplicatas exatas
  são achadas por hash; quase duplicatas por correlação das imagens reduzidas, que não
  depende de brilho e contraste; imagens parecidas em cadeia caem no mesmo grupo.
  (`tests/test_auditoria.py`)

### Filtros
- **FIL-01**: a correlação escrita em NumPy reproduz o `cv2.filter2D` (borda refletida) para
  kernels simétricos e assimétricos; suavizações preservam imagem lisa; a mediana limpa sal e
  pimenta melhor que a média; o Sobel só responde na direção da borda; o Canny automático
  devolve imagem binária com limiares ordenados; o CLAHE aumenta o contraste de imagem apagada.
  (`tests/test_filtros.py`)

### Geometria (ECG)
- **GEO-01**: em papel milimetrado sintético de régua conhecida, o passo da grade é recuperado
  (±0,05 px/mm); a frequência cardíaca da tira de ritmo sai a ±3 % para 60, 75 e 110 bpm; a
  máscara do traçado ignora grade vermelha e sombra; numa foto simulada em perspectiva os 4 cantos
  são achados a menos de 6 px e, depois da homografia, a régua volta ao valor original.
  (`tests/test_geometria.py`)

### Métricas
- **MET-01**: Dice e IoU valem 1 para máscaras idênticas (ou ambas vazias), 0 para disjuntas,
  batem com a conta à mão num caso de meia sobreposição, e Dice ≥ IoU. (`tests/test_metricas.py`)

### Índice cardiotorácico
- **ICT-01**: num tórax sintético de medidas conhecidas, o perfil por linha recupera o ICT a ±0,03
  com ou sem costelas; imagem sem estrutura não é medida; a segmentação 2D acha os dois pulmões
  quando não há costelas; máscara com um pulmão só não é medida. (`tests/test_ict.py`)

### Modelos e avaliação
- **MOD-01**: os tensores têm formato (n, 128, 128, 1) em [0, 1]; o aumento de dados **não espelha**
  (o coração não muda de lado); a CNN pequena aprende um problema de brinquedo (AUC > 0,9) e expõe
  `conv1`–`conv3`; o Grad-CAM tem o tamanho da imagem, fica em [0, 1] e concentra a atenção na região
  que decide, inclusive no modelo de transfer learning. (`tests/test_modelos.py`)
- **MOD-02**: a AUC com bootstrap contém a AUC pontual; a comparação **pareada** detecta o escore
  melhor (IC acima de zero) e dá diferença zero para escores iguais; o limiar de Youden separa um
  caso perfeito; exames sem escore ficam de fora. (`tests/test_modelos.py`)

### Divisões e auditoria de atalho
- **DADOS-02**: a divisão por grupo nunca parte um grupo e mantém 60/20/20 com classes equilibradas;
  a divisão aleatória mantém 60/20/20; tapar e isolar uma caixa são complementares e pintam com a
  média da própria imagem; as caixas de referência cobrem coração e tórax de um tórax sintético.
  (`tests/test_dados.py`, `tests/test_auditoria.py`, `tests/test_ict.py`)

### Segmentação
- **SEG-01**: um polígono vira máscara com a área certa; a largura máxima de uma máscara é a maior
  extensão numa linha; as bordas do ICT e o limiar com morfologia acham o coração de um tórax
  sintético (Dice > 0,7); as marcações exportadas são lidas com e sem o envelope `data`, e contornos
  com menos de 3 pontos ficam de fora; o maior componente conexo descarta ilhas e tapa buracos.
  (`tests/test_segmentacao.py`)
- **SEG-02**: o Dice suave da U-Net é igual ao Dice em máscaras binárias e a perda some quando a
  previsão é o rótulo; a U-Net devolve um mapa 128 × 128 em [0, 1] com os blocos `desce`, `fundo` e
  `sobe` nomeados; num problema de brinquedo (disco em posição sorteada) ela aprende o recorte
  (Dice > 0,8). (`tests/test_unet.py`)

### Detecção
- **DET-01**: caixa → formato YOLO → caixa volta igual; IoU de caixas idênticas é 1, disjuntas 0 e
  meia sobreposição 1/3; a caixa do polígono é a menor que o contém; o pseudo-rótulo do ICT cobre o
  coração de um tórax sintético (IoU > 0,8) e não existe sem medida; o dataset grava imagem, rótulo
  (vazio quando não há objeto) e o YAML de uma classe. (`tests/test_deteccao.py`)
