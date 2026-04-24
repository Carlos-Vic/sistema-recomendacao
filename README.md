# Sistema de Recomendação de Perfumes Nacionais

Projeto 1 — Introdução à Inteligência Artificial · UnB 2026/1

Sistema de recomendação **híbrido** para uma loja virtual de perfumes nacionais, combinando:
- **TF-IDF** — gera candidatos com base nas preferências declaradas pelo usuário no formulário
- **SVD** — reordena os candidatos prevendo a nota que o usuário daria a cada perfume, treinado na matriz de utilidade de 500 usuários

---

## Estrutura do projeto

```
sistema-recomendacao/
├── README.md
├── app.py                      # interface Gradio
├── scripts/
│   ├── gerar_dados.py          # gera os CSVs de produtos e matriz de utilidade
│   └── sistema_recomendacao.py # código principal do notebook (células # %%)
├── notebook/
│   └── sistema_recomendacao.ipynb  # notebook exportado pelo VS Code
├── modelos/                    # gerado ao executar o notebook
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   └── svd_pred_matrix.pkl
├── imagens/                    # imagens dos perfumes
└── dados/                      # gerado automaticamente pelo gerar_dados.py
    ├── produtos.csv
    ├── produtos_processados.csv
    ├── matriz_utilidade.csv
    └── feedback.csv            # gerado pelo app ao receber avaliações
```

---

## Como rodar

### Pré-requisitos

- Python 3.10 ou superior instalado
- Git instalado

### 0. Clonar o repositório

```bash
git clone https://github.com/Carlos-Vic/sistema-recomendacao.git
cd sistema-recomendacao
```

### 1. Instalar dependências

```bash
pip install pandas scikit-learn scipy seaborn matplotlib gradio joblib
```

### 2. Gerar os dados

```bash
python scripts/gerar_dados.py
```

Isso cria a pasta `dados/` com dois arquivos:
- `produtos.csv` — catálogo com 50 perfumes nacionais
- `matriz_utilidade.csv` — 500 usuários × 50 perfumes com avaliações 1–5

### 3. Rodar a interface

```bash
python app.py
```

O Gradio abrirá automaticamente no navegador em `http://127.0.0.1:7860`.

> Os modelos já estão pré-treinados na pasta `modelos/`. O notebook `sistema_recomendacao.ipynb` documenta o processo de EDA e treinamento, mas não precisa ser executado para usar o sistema.

### 4. Testar o sistema

1. Acesse a aba **Portal do Cliente** e crie uma conta informando e-mail, senha, nome e preferências olfativas
2. Acesse **Minha Vitrine** e clique em **Gerar Recomendações**
3. Selecione os perfumes desejados e clique em **Fazer Pedido**
4. Acesse **Feedback**, carregue seus pedidos e avalie os perfumes com nota 1–5
5. Para alterar suas preferências, volte ao **Portal do Cliente** e use o accordion **Editar Preferências**

---

## Sobre o `feedback.csv`

Gerado automaticamente em `dados/feedback.csv` quando o usuário avalia perfumes na aba **Feedback** do app. Cada linha registra o nome do perfume e a nota atribuída (1–5):

```
perfume,nota,review
Malbec Tradicional,5, ótimo perfume
Quasar,3, perfume bem mais ou menos
```

O arquivo cresce a cada avaliação e persiste entre sessões. Em uma versão futura de produção, essas notas poderiam ser incorporadas à matriz de utilidade para retreinar o SVD periodicamente, tornando as recomendações mais precisas com o tempo.

---

## Sobre o `gerar_dados.py`

O script gera os dados sintéticos do projeto em duas etapas:

**Catálogo de produtos** — escreve `produtos.csv` com 50 perfumes nacionais (O Boticário, Natura, Eudora, Avon, Granado, Phebo e Mahogany), cada um com família olfativa, notas olfativas, ocasião de uso e preço pesquisado nas fontes oficiais.

**Matriz de utilidade** — gera `matriz_utilidade.csv` com 500 usuários simulados. A geração não é puramente aleatória: cada usuário recebe uma *persona* com famílias olfativas favoritas, ocasião preferida e preferência de gênero. A nota de cada perfume é calculada com base no grau de compatibilidade entre a persona e o produto, com ruído gaussiano (σ = 0,45) para simular variabilidade natural. Cerca de 12% dos usuários são *silenciosos* — compraram mas nunca avaliaram nenhum perfume, comportamento comum em lojas reais.

A execução é reproduzível: `seed=42` garante que a matriz gerada seja sempre idêntica.
