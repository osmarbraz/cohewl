# COHEWL - Coherence at Word Level 

Avaliando a coerência semântica em textos curtos a nível de palavras.

## **Resumo**

Colocar o resumo aqui.

## **Diretórios**

Relação e descrição dos principais diretórios do COHEWL:
* **dataset** - Diretório com os conjuntos de dados.
* **notebooks** - Diretório com o notebooks dos experimentos.
* **projecao** - Diretório com os arquivos de projeções.

## **Instalação**

**Requisitos**

* Python 3.6.9
* [Transformer Huggingface 4.5.1](https://huggingface.co/transformers/)
* [PyTorch 1.8.1](https://pytorch.org/)
* [Spacy 3.7.13](https://spacy.io/)

**Download - Clone**

```
!git clone https://github.com/osmarbraz/cohewl.git
```

## 1. Dataset

A pasta **"dataset"** contêm os arquivos dos conjuntos de dados utilizados em português e inglês.

## 2. Notebooks

A pasta **"notebooks"** contêm os arquivos dos notebooks utilizados em nossos experimentos. Os arquivos estão divididos para cada conjuntos de dados e seu idioma.

## 3. Projeções de Embeddings

Projeções de *embeddings* sentenças, palavras e sentenças e palavras gerados pelo **BERT** tamanho **large** utilizando a ferramenta **Embedding Projector** (https://projector.tensorflow.org/).

Os arquivos utilizados pelo projetor estão na pasta **"projecao"** e divididos em três pastas: **"sentenca"**, **"token"** e **"token_sentenca"**. As pastas indicam se foi utilizado *embeddings* consolidados das **sentenças**, *embeddings* de **tokens** ou combinados. 
Cada pasta **"sentenca"**, **"token"** e **"token_sentenca"**,  possui os arquivos para os conjuntos de dados. As projeções ocorrem para as palavras e sentenças dos documentos originais(**DO**) e modificados (**DM**).

### 3.1 Projeções dos *embeddings* de sentenças

Projeções dos *embeddings* das sentenças consolidados pela média dos *embeddings* dos tokens. Os *embeddings* dos tokens são das 4 últimas camadas do **BERTimbau Large** em português ou **BERT Large** em inglês e concatenados.

As projeções dos *embeddings* de sentenças estão relacionados com os seguintes **metadados**:
- Sentença
- Id (Id do documento)
- Origem (Id do documento de origem)
- Classe (1 - Original e 0 - Modificado)

#### As pastas **"sentenca/cohquad_cohinc"**, **"sentenca/d_cohquad_coh"**, **"sentenca/d_cohquad_inc"**, **"sentenca/d_faquad"** e **"sentenca/d_squard2"**  possuem projeções dos *embeddings* separam os arquivos de dados dos conjuntos de dados.


**Links** para os arquivos de configuração do **Embedding Projector** para os conjuntos de dados:

- **CohQuAD Coh+Inc (pt-br):** *config_cohquad_cohinc_ptbr_sentenca.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/sentenca/config_cohquad_cohinc_ptbr_sentenca.json

- **CohQuAD Coh+Inc (en):** *config_cohquad_cohinc_ptbr_sentenca.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/sentenca/config_cohquad_cohinc_en_sentenca.json

- **D(CohQuAD Coh, 20) (pt-br):** *config_d_cohquad_coh_ptbr_20_sentenca.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/sentenca/config_d_cohquad_coh_ptbr_20_sentenca.json

- **D(CohQuAD Coh, 20) (en):** *config_d_cohquad_coh_en_20_sentenca.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/sentenca/config_d_cohquad_coh_en_20_sentenca.json

- **D(CohQuAD Inc, 20) (pt-br):** *config_d_cohquad_inc_ptbr_20_sentenca.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/sentenca/config_d_cohquad_inc_ptbr_20_sentenca.json

- **D(CohQuAD Inc, 20) (en):** *config_d_cohquad_inc_en_20_sentenca.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/sentenca/config_d_cohquad_inc_en_20_sentenca.json

- **20 D(20 DO FAQuAD, 20) (pt-br):** *config_d_faquad_ptbr_20_sentenca.json*:
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/sentenca/config_d_faquad_ptbr_20_sentenca.json

- **20 D(20 DO SQuAD2, 20) (pt-br):** *config_d_squad2_ptbr_20_sentenca.json*:
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/sentenca/config_d_squad2_ptbr_20_sentenca.json

- **D(20 DO SQuAD2, 20) (en):** *config_d_squad2_en_20_sentenca.json*:
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/sentenca/config_d_squad2_en_20_sentenca.json

### 3.2 Projeções dos *embeddings* de palavras
Projeções de *embeddings* das palavras consolidados pela média dos *embeddings* dos tokens para as palavras que estão fora do vocabulário do modelo. Os *embeddings* dos tokens são recuperados das 4 últimas camadas do **BERTimbau Large** em português ou **BERT Large** em inglês e concatenados.

As projeções dos *embeddings* de palavras podem se relacionar com os seguintes **metadados**:
- Token
- POS-Tag 
- OOV (1 - Não existe no vocabulário do **BERT** e combina os *embeddings* dos tokens para formar a palavra e 0 - Existe no vocabulário do **BERT**)
- Id (Id do documento)
- Origem (Id do documento de origem)
- Classe (1 - Original e 0 - Modificado)
- Perturbada (1 - Modificada, 0 - Original)
- Index (Índice da palavra na sentença)
- Próximo token da sentença
- Sentença

#### As pastas **"token/cohquad_cohinc"**, **"token/d_cohquad_coh"**, **"token/d_cohquad_inc"**, **"token/d_faquad"** e **"token/d_squard2"**  possuem projeções dos *embeddings* separam os arquivos de dados dos conjuntos de dados.

**Links** para os arquivos de configuração do **Embedding Projector** dos conjuntos de dados:

- **CohQuAD Coh+Inc (pt-br)**:
	- **Sem linhas** *config_cohquad_cohinc_ptbr_token.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_cohquad_cohinc_ptbr_token.json
	- **Com linhas** *config_cohquad_cohinc_ptbr_token_next.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_cohquad_cohinc_ptbr_token_next.json

- **CohQuAD Coh+Inc (en)**:
	- **Sem linhas** *config_cohquad_cohinc_en_token.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_cohquad_cohinc_en_token.json
	- **Com linhas** *config_cohquad_cohinc_en_token_next.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_cohquad_cohinc_en_token_next.json

- **D(CohQuAD Coh, 20) (pt-br)**:
	- **Sem linhas** *config_d_cohquad_coh_ptbr_20_token.json* sem linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_cohquad_coh_ptbr_20_token.json
	- **Com linhas** *config_d_cohquad_coh_ptbr_20_token_next.json* com linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_cohquad_coh_ptbr_20_token_next.json

- **D(CohQuAD Coh, 20) (en)**:
	- **Sem linhas** *config_d_cohquad_coh_en_20_token.json* sem linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_cohquad_coh_en_20_token.json
	- **Com linhas** *config_d_cohquad_coh_en_20_token_next.json* com linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_cohquad_coh_en_20_token_next.json

- **D(CohQuAD Inc, 20) (pt-br)**:
	- **Sem linhas** *config_d_cohquad_inc_ptbr_20_token.json* sem linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_cohquad_inc_ptbr_20_token.json
	- **Com linhas** *config_d_cohquad_inc_ptbr_20_token_next.json* com linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_cohquad_inc_ptbr_20_token_next.json

- **D(CohQuAD Inc, 20) (en)**:
	- **Sem linhas** *config_d_cohquad_inc_en_20_token.json* sem linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_cohquad_inc_en_20_token.json
	- **Com linhas** *config_d_cohquad_inc_en_20_token_next.json* com linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_cohquad_inc_en_20_token_next.json

- **D(20 DO FaQuAD, 20)**:
	- **Sem linhas** *config_d_faquad_ptbr_20_token.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_faquad_ptbr_20_token.json
	- **Com linhas** *config_d_faquad_ptbr_20_token_next.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_faquad_ptbr_20_token_next.json

- **D(20 DO SQuAD2, 20) (pt-br)**:
	- **Sem linhas** *config_d_squad2_ptbr_20_token.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_squad2_ptbr_20_token.json
	- **Com linhas** *config_d_squad2_ptbr_20_token_next.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_squad2_ptbr_20_token_next.json

- **D(20 DO SQuAD2, 20) (en)**:
	- **Sem linhas** *config_d_squad2_en_20_token.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_squad2_en_20_token.json
	- **Com linhas** *config_d_squad2_en_20_token_next.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token/config_d_squad2_en_20_token_next.json

### 3.3 Projeções dos *embeddings* de palavras e sentenças

Projeções dos *embeddings* das palavras e sentenças. Os *embeddings* das palavras são consolidados pela média dos *embeddings* dos tokens para as palavras que estão foram do vocabulário do modelo. Os *embeddings* das sentenças são consolidados pela média dos *embeddings* dos tokens de todas para as palavras. Todos os *embeddings* são recuperados das 4 últimas camadas do **BERTimbau Large** em português ou **BERT Large** em inglês e concatenados.

As projeções dos *embeddings* de palavras e sentenças podem se relacionar com os seguintes **metadados**:
- Token (Exibe a palavra ou o número do documento)
- POS-Tag 
- OOV (1 - Não existe no vocabulário do **BERT** e combina os *embeddings* dos tokens para formar a palavra e 0 - Existe no vocabulário do **BERT**)
- Id (Id do documento)
- Origem (Id do documento de origem)
- Classe (1 - Original e 0 - Modificado)
- Perturbada (1 - Modificado, 0 - Original)
- Index (Índice da palavra na sentença)
- Próximo token da sentença
- Granularidade (0 - Token, 1 - Sentença)
- Tipo Texto (0 - Palavra modificada, 1 Palavra Original, 2 - Sentença modificada, 3 - Sentença original)
- Sentença

#### As pastas **"token_sentenca/cohquad_cohinc"**, **"token_sentenca/d_cohquad_coh"**, **"token_sentenca/d_cohquad_inc"**, **"token_sentenca/d_faquad"** e **"token_sentenca/d_squard2"**  possuem projeções dos *embeddings* separam os arquivos de dados dos conjuntos de dados.

**Links** para os arquivos de configuração do **Embedding Projector** dos conjuntos de dados:

- **CohQuAD Coh+Inc (pt-br)**:
	- **Sem linhas** *config_cohquad_cohinc_ptbr_token_sentenca.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_cohquad_cohinc_ptbr_token_sentenca.json
	- **Com linhas** *config_cohquad_cohinc_ptbr_token_sentenca_next.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_cohquad_cohinc_ptbr_token_sentenca_next.json

- **CohQuAD Coh+Inc (en)**:
	- **Sem linhas** *config_cohquad_cohinc_en_token_sentenca.json* sem linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_cohquad_cohinc_en_token_sentenca.json
	- **Com linhas** *config_cohquad_cohinc_en_token_sentenca_next.json* com linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_cohquad_cohinc_en_token_sentenca_next.json

- **D(CohQuAD Coh, 20) (pt-br)**:
	- **Sem linhas** *config_d_cohquad_coh_ptbr_20_token_sentenca.json* sem linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_cohquad_coh_ptbr_20_token_sentenca.json
	- **Com linhas** *config_d_cohquad_coh_ptbr_20_token_sentenca_next.json* com linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_cohquad_coh_ptbr_20_token_sentenca_next.json

- **D(CohQuAD Coh, 20) (en)**:
	- **Sem linhas** *config_d_cohquad_coh_en_20_token_sentenca.json* sem linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_cohquad_coh_en_20_token_sentenca.json
	- **Com linhas** *config_d_cohquad_coh_en_20_token_sentenca_next.json* com linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_cohquad_coh_en_20_token_sentenca_next.json

- **D(CohQuAD Inc, 20) (pt-br)**:
	- **Sem linhas** *config_d_cohquad_inc_ptbr_20_token_sentenca.json* sem linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_cohquad_inc_ptbr_20_token_sentenca.json
	- **Com linhas** *config_d_cohquad_inc_ptbr_20_token_sentenca_next.json* com linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_cohquad_inc_ptbr_20_token_sentenca_next.json

- **D(CohQuAD Inc, 20) (en)**:
	- **Sem linhas** *config_d_cohquad_inc_en_20_token_sentenca.json* sem linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_cohquad_inc_en_20_token_sentenca.json
	- **Com linhas** *config_d_cohquad_inc_en_20_token_sentenca_next.json* com linhas: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_cohquad_inc_en_20_token_sentenca_next.json

- **D(20 DO FaQuAD, 20)**:
	- **Sem linhas** *config_d_faquad_ptbr_20_token_sentenca.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_faquad_ptbr_20_token_sentenca.json
	- **Com linhas** *config_d_faquad_ptbr_20_token_sentenca_next.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_faquad_ptbr_20_token_sentenca_next.json

- **D(20 DO SQuAD2, 20) (pt-br)**:
	- **Sem linhas** *config_d_squad2_ptbr_20_token_sentenca.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_squad2_ptbr_20_token_sentenca.json
	- **Com linhas** *config_d_squad2_ptbr_20_token_sentenca_next.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_squad2_ptbr_20_token_sentenca_next.json

- **D(20 DO SQuAD2, 20) (en)**:
	- **Sem linhas** *config_d_squad2_en_20_token_sentenca.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_squad2_en_20_token_sentenca.json
	- **Com linhas** *config_d_squad2_en_20_token_sentenca_next.json*: 
https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/osmarbraz/cohewl/main/projecao/token_sentenca/config_d_squad2_en_20_token_sentenca_next.json

## Referências

**BERTimbau** : Souza, F., Nogueira, R., Lotufo, R., 2020. Bertimbau: Pretrained bert models for brazilian portuguese, in: Brazilian Conference on Intelligent Systems, Springer. Springer, Rio Grande, Brazil. pp. 403–417. https://link.springer.com/chapter/10.1007/978-3-030-61377-8_28

**Embedding Projector** : Smilkov, D., Thorat, N., Nicholson, C., Reif, E., Vi ́egas, F.B., Wattenberg, M., 2016. Embedding projector: Interactive visualization and interpretation of embeddings. arXiv preprint arXiv:1611.05469. https://arxiv.org/pdf/1611.05469.pdf
