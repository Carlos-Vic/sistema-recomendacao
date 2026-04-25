"""
Gera o arquivo produtos_processados.csv a partir de produtos.csv
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
DADOS = BASE / "dados"

# Carrega o CSV de produtos
df_produtos = pd.read_csv(DADOS / "produtos.csv")

# Salva como produtos_processados.csv (a mesma estrutura)
df_produtos.to_csv(DADOS / "produtos_processados.csv", index=False, encoding="utf-8")

print(f"✅ {DADOS / 'produtos_processados.csv'} criado com sucesso!")
