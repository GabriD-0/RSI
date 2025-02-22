# Recomendação de Similaridade de Imagem com Transfer Learning

**AVISO:** Este projeto ainda está em processo de desenvolvimento e melhorias contínuas. Novos recursos, otimizações e ajustes podem ser implementados a qualquer momento.

Este projeto utiliza o dataset [Clothing Dataset Full](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full) do Kaggle para criar um pipeline de classificação de imagens utilizando técnicas de Transfer Learning. Além do treinamento do modelo para classificação, o projeto também contempla um sistema de recomendação de imagens baseado na similaridade visual.

O repositório está dividido em três partes principais:

- **Organização do Dataset:**  
  Script para ler o arquivo CSV e mover as imagens para pastas organizadas por categoria.

- **Treinamento do Modelo:**  
  Script que utiliza TensorFlow e TensorFlow Hub para treinar um modelo de classificação a partir das imagens organizadas.

- **API de Recomendação (Flask):**  
  Serviço web que integra a extração de features com indexação (usando Annoy) e fornece recomendações de imagens similares para um produto de interesse.

---

## Estrutura do Repositório

- **OrganizaDataset.py**  
  Responsável por:  
  - Ler o arquivo `images.csv` do dataset.
  - Verificar a existência de cada imagem na pasta de origem.
  - Criar pastas de destino baseadas nas labels e mover as imagens correspondentes.

- **Treino.py**  
  Responsável por:  
  - Carregar os dados organizados a partir da estrutura de pastas.
  - Criar geradores de dados para treinamento (80%) e validação (20%).
  - Construir e compilar o modelo utilizando o módulo BiT do TensorFlow Hub para Transfer Learning.
  - Treinar o modelo, exibir gráficos de loss e acurácia e salvar os modelos treinados (modelo completo e extrator de features).

- **API de Recomendação (Flask)**  
  Esta parte do projeto foi desenvolvida para permitir a recomendação de imagens similares com base na aparência. A aplicação Flask, comentada detalhadamente no código-fonte, realiza as seguintes funções:
  
  - **Upload e Processamento de Imagens Base:**  
    Um endpoint recebe um arquivo ZIP contendo imagens de referência (base). O sistema:
    - Extrai as imagens do ZIP.
    - Processa cada imagem para extrair seus vetores de features utilizando o modelo pré-treinado (BiT).
    - Armazena os vetores extraídos e os indexa com a biblioteca Annoy, facilitando a busca por similaridade.

  - **Upload e Teste de Imagens:**  
    Outro endpoint recebe imagens (em arquivo ZIP) que servirão como consulta. Para cada imagem de teste, o sistema:
    - Extrai as features.
    - Compara o vetor extraído com o índice previamente construído.
    - Retorna os itens mais similares, de acordo com um limiar de distância definido.

  Essa estrutura possibilita que, a partir de uma imagem de entrada, o sistema indique produtos com aparência similar, permitindo a criação de um mecanismo de recomendação visual robusto.

---

## Pré-requisitos

- **Python 3.10 ou superior**
- **Bibliotecas necessárias:**
  - tensorflow==2.13.*
  - tensorflow_hub==0.14.*
  - numpy
  - matplotlib
  - pandas
  - Pillow
  - tqdm
  - scipy
  - annoy
  - flask

Para instalar as dependências, você pode utilizar o seguinte comando:

```bash
pip install tensorflow==2.13.* tensorflow_hub==0.14.* numpy matplotlib pandas Pillow tqdm scipy annoy flask
```

---

## Dataset

Baixe o dataset a partir do Kaggle:

[Clothing Dataset Full](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full)

**Atenção:**  
Após o download, verifique se os caminhos definidos nos scripts (`csv_file` e `origem` no `OrganizaDataset.py`) correspondem à localização correta dos arquivos no seu ambiente.

---

## Como Utilizar

### 1. Organizando o Dataset

O script `OrganizaDataset.py` lê o arquivo CSV e organiza as imagens em pastas baseadas nas labels.  
Para executar:

```bash
python OrganizaDataset.py
```

### 2. Treinando o Modelo

O script `Treino.py` realiza as seguintes tarefas:
- Configura os geradores de dados a partir da estrutura de pastas (./dataset/Categories).
- Define os hiperparâmetros (tamanho da imagem, batch size, data augmentation, etc.).
- Constrói e compila o modelo utilizando o módulo BiT para Transfer Learning.
- Executa o treinamento, plota os gráficos de loss e acurácia, e salva os modelos treinados.

Para executar:

```bash
python Treino.py
```

**Nota:**  
- Caso o dataset esteja organizado em uma estrutura diferente, ajuste a variável `data_dir` no `Treino.py`.  
- Se desejar utilizar data augmentation, altere a variável `do_data_augmentation` para `True`.

### 3. Utilizando a API de Recomendação

A aplicação Flask integra a funcionalidade de recomendação de imagens por similaridade. Embora o código-fonte da API esteja presente no repositório (com comentários detalhados para facilitar o entendimento), a sua funcionalidade pode ser resumida em dois pontos principais:

- **Upload de Imagens Base:**  
  Permite que um arquivo ZIP com imagens de referência seja enviado. O sistema processa essas imagens, extrai as features e constrói um índice (usando Annoy) para buscas futuras.

- **Upload de Imagens para Consulta:**  
  Permite que imagens de teste sejam enviadas para consulta. Para cada imagem, o sistema extrai as features e realiza uma busca no índice, retornando os itens com maior similaridade (dentro de um limiar definido).

Essa API facilita a integração do mecanismo de recomendação em sistemas web ou mobile, permitindo que os usuários recebam sugestões de produtos visualmente similares a partir de uma imagem de entrada.

---

## Contribuições

Contribuições são bem-vindas! Se você tiver sugestões, melhorias ou encontrar algum problema, sinta-se à vontade para abrir uma issue ou enviar um pull request.

