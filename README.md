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
├── scripts/
│   ├── gerar_dados.py          # gera os CSVs de produtos e matriz de utilidade
│   └── sistema_recomendacao.py # código principal do notebook (células # %%)
├── notebook/
│   └── sistema_recomendacao.ipynb  # notebook exportado pelo VS Code
├── imagens/                    # imagens dos perfumes (baixar via Drive compartilhado)
└── dados/                      # gerado automaticamente pelo gerar_dados.py
    ├── produtos.csv
    └── matriz_utilidade.csv
```

---

## Como rodar

### 1. Instalar dependências

```bash
pip install pandas scikit-learn seaborn matplotlib gradio scikit-surprise
```

### 2. Gerar os dados

```bash
python scripts/gerar_dados.py
```

Isso cria a pasta `dados/` com dois arquivos:
- `produtos.csv` — catálogo com 50 perfumes nacionais
- `matriz_utilidade.csv` — 500 usuários × 50 perfumes com avaliações 1–5

### 3. Baixar as imagens

Acesse o Drive compartilhado e copie o conteúdo para a pasta `imagens/`.

### 4. Abrir o notebook

Abra `scripts/sistema_recomendacao.py` no VS Code com a extensão **Jupyter** instalada. As células são delimitadas por `# %%` e podem ser executadas diretamente pelo VS Code, sem precisar abrir o `.ipynb`. O arquivo `notebook/sistema_recomendacao.ipynb` é a versão exportada do notebook.

---

## Sobre o `gerar_dados.py`

O script gera os dados sintéticos do projeto em duas etapas:

**Catálogo de produtos** — escreve `produtos.csv` com 50 perfumes nacionais (O Boticário, Natura, Eudora, Avon, Granado, Phebo e Mahogany), cada um com família olfativa, notas olfativas, ocasião de uso e preço pesquisado nas fontes oficiais.

**Matriz de utilidade** — gera `matriz_utilidade.csv` com 500 usuários simulados. A geração não é puramente aleatória: cada usuário recebe uma *persona* com famílias olfativas favoritas, ocasião preferida e preferência de gênero. A nota de cada perfume é calculada com base no grau de compatibilidade entre a persona e o produto, com ruído gaussiano (σ = 0,45) para simular variabilidade natural. Cerca de 12% dos usuários são *silenciosos* — compraram mas nunca avaliaram nenhum perfume, comportamento comum em lojas reais.

A execução é reproduzível: `seed=42` garante que a matriz gerada seja sempre idêntica.
