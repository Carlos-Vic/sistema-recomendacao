# %% [markdown]
# # Sistema de Recomendação de Perfumes Nacionais
# **Projeto 1 · Introdução a Inteligência Artificial · UnB 2026/1**
#
# Sistema de recomendação para uma loja virtual de perfumes nacionais, combinando dois modelos:
#
# - **TF-IDF** - gera candidatos com base nas preferências declaradas pelo usuário no formulário (família olfativa, ocasião, faixa de preço)
# - **SVD** - reordena os candidatos usando a matriz de utilidade de 500 usuários, priorizando perfumes bem avaliados por usuários com perfil similar
#
# Essa abordagem híbrida resolve o problema de novos usuários sem histórico, onde eles vão receber recomendações pelo TF-IDF, que são então refinadas pelo SVD.

# %% [markdown]
# ## 1. Carregamento dos dados

# %%
import pandas as pd

# carregando o csv
df_produtos = pd.read_csv("../dados/produtos.csv")
df_produtos.head()

# %%

df_matriz = pd.read_csv('../dados/matriz_utilidade.csv', index_col='usuario_id')
df_matriz.head()

# %% [markdown]
# ## 2. Análise Exploratória dos Dados (EDA)
#
# ### 2.1 Visão geral dos datasets

# %%

print(df_produtos.info())
print(f'\n#########')
print(df_produtos.describe())

# %% [markdown]
# O catálogo possui 50 perfumes com 9 colunas: id, nome, marca, gênero, família olfativa, notas olfativas, ocasião, preço e caminho da imagem.
# Não há valores nulos. Os preços variam de R$69,90 a R$319,90, com média em torno de R$156.

# %%

print(df_matriz.info())
print(f'\n#########')
print(df_matriz.describe())

# %% [markdown]
# A matriz possui 500 usuários × 50 perfumes. A maioria das células é 0 (não avaliado).
# As notas reais (1–5) serão analisadas mais a frente.

# %% [markdown]
# ### 2.2 Distribuição do catálogo de produtos

# %%
# agrupando os perfumes por marca para plotar
perfume_por_marca = df_produtos.groupby(['marca']).agg(total=('nome','count')).sort_values('total', ascending=False).reset_index()
perfume_por_marca

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(x="total", y="marca", data=perfume_por_marca)
plt.title("Perfumes por Marca")
plt.xlabel("Quantidade")
plt.ylabel("Marca")
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# %% [markdown]
# Boticário e Natura dominam aparecem mais vezes no catálogo.
# Avon contribui com 7 perfumes, Eudora com 4, e Mahogany, Granado e Phebo com 1–2 cada.

# %%
# agrupando perfume por familia olfativa para plotar
perfume_por_familia = df_produtos.groupby(['familia_olfativa']).agg(total=('nome','count')).sort_values('total', ascending=False).reset_index()
perfume_por_familia

# %%

sns.barplot(x="total", y="familia_olfativa", data=perfume_por_familia)
plt.title("Perfumes por Família Olfativa")
plt.xlabel("Quantidade")
plt.ylabel("Família Olfativa")
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# %% [markdown]
# O catálogo é bastante diversificado: 22 famílias olfativas diferentes. As mais representadas são
# amadeirado aromático e floral. A maioria das famílias porém aparece apenas uma vez

# %%
# agrupando perfume por genero para plotar
perfume_por_genero = df_produtos.groupby(['genero']).agg(total=('nome','count')).sort_values('total', ascending=False).reset_index()
perfume_por_genero

# %%

sns.barplot(x="total", y="genero", data=perfume_por_genero)
plt.title("Perfumes por Gênero")
plt.xlabel("Quantidade")
plt.ylabel("Gênero")
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# %% [markdown]
# O catálogo é equilibrado entre feminino e masculino, mas tem apenas 1 perfume unissex.

# %%

sns.histplot(data=df_produtos, x="preco", kde=True)
plt.title("Distribuição de Preços")
plt.xlabel("Preço (R$)")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

# %% [markdown]
# A distribuição de preços é assimétrica a direita. A maioria dos perfumes se concentra entre R$120 e R$170,
# com uma cauda de produtos premium acima de R$250.

# %% [markdown]
# ### 2.3 Análise da matriz de utilidade
#
# Antes de analisar as notas é verificado quantas células não foram avaliadas.
# Isso é esperado em sistemas de recomendação reais: usuários avaliam apenas uma pequena fração dos produtos disponíveis.
# Os zeros serão excluídos nas análises seguintes pois representam ausência de avaliação, não uma nota ruim.

# %%

total = df_matriz.size # total de combinaçoes da matriz
nao_avaliados = (df_matriz == 0).sum().sum() # conta o total de celulas zeradas na matriz
avaliados = total - nao_avaliados # calcula as combinacoes que tem nota de 1 a 5 
media_por_usuario = avaliados / df_matriz.shape[0] # calcula a media de perfumes avaliados por usuario
silenciosos = (df_matriz.sum(axis=1) == 0).sum() # conta os usuarios que nao avaliaram nenhum perfume

print(f"Total de combinações possíveis (usuários x perfumes): {total}")
print(f"Combinações avaliadas (nota 1-5):  {avaliados} ({avaliados/total*100:.1f}%)")
print(f"Combinações não avaliadas (nota 0): {nao_avaliados} ({nao_avaliados/total*100:.1f}%)")
print(f"Média de perfumes avaliados por usuário: {media_por_usuario:.1f}")
print(f"Usuários que nunca avaliaram nenhum perfume: {silenciosos} ({silenciosos/df_matriz.shape[0]*100:.1f}%)")

# %% [markdown]

# A maioria dos usuários avaliou apenas uma pequena fração dos 50 perfumes disponíveis, o que é esperado em lojas reais.
# Uma parcela dos usuários não avaliou nenhum produto. Os dados foram gerados dessa forma para simular
# compradores reais que nunca deixam avaliação, tornando a matriz mais próxima de um cenário real.
# %% [markdown]
# ### 2.3 (continuação) — Distribuição das notas

# %%
# Histograma das notas 1–5 (excluindo zeros = não avaliados)
notas = df_matriz.values.flatten()
notas_validas = notas[notas > 0]

sns.histplot(notas_validas, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], discrete=True)
plt.title("Distribuição das Notas (1–5)")
plt.xlabel("Nota")
plt.ylabel("Frequência")
plt.xticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.show()

# %% [markdown]
# As notas se concentram entre 3 e 4, com formato aproximadamente normal centrado em 3.
# Notas extremas (1 e 5) são menos frequentes, o que indica que o modelo de personas gera
# avaliações realistas, sem polarização excessiva.

# %%
# Distribuição de avaliações por usuário (quantos perfumes cada um avaliou)
avaliacoes_por_usuario = (df_matriz > 0).sum(axis=1)

sns.histplot(avaliacoes_por_usuario, bins=20, kde=True)
plt.title("Perfumes Avaliados por Usuário")
plt.xlabel("Quantidade de perfumes avaliados")
plt.ylabel("Número de usuários")
plt.tight_layout()
plt.show()

# %% [markdown]
# A maioria dos usuários avaliou entre 8 e 16 perfumes. Há um pico em 0 correspondente
# aos ~12% de usuários silenciosos que nunca avaliaram nenhum produto.

# %%
# Perfumes mais e menos avaliados
avaliacoes_por_perfume = (df_matriz > 0).sum(axis=0).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Top 10 mais avaliados
top10 = avaliacoes_por_perfume.head(10)
nomes_top = [df_produtos.loc[df_produtos['id'] == int(pid[1:]), 'nome'].values[0] for pid in top10.index]
axes[0].barh(nomes_top[::-1], top10.values[::-1], color='steelblue')
axes[0].set_title("Top 10 — Mais avaliados")
axes[0].set_xlabel("Avaliações")

# Top 10 menos avaliados
bot10 = avaliacoes_por_perfume.tail(10)
nomes_bot = [df_produtos.loc[df_produtos['id'] == int(pid[1:]), 'nome'].values[0] for pid in bot10.index]
axes[1].barh(nomes_bot[::-1], bot10.values[::-1], color='salmon')
axes[1].set_title("Top 10 — Menos avaliados")
axes[1].set_xlabel("Avaliações")

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 3. Modelo híbrido
#
# ### 3.1 TF-IDF — Filtragem por conteúdo

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cria a coluna corpus combinando as 3 características textuais
df_produtos['corpus'] = (
    df_produtos['familia_olfativa'] + ' ' +
    df_produtos['notas_olfativas'] + ' ' +
    df_produtos['ocasiao']
)

# Vetorização TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_produtos['corpus'])

# Matriz de similaridade de cosseno (50×50)
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"Vocabulário TF-IDF: {len(tfidf.vocabulary_)} termos")
print(f"Matriz TF-IDF: {tfidf_matrix.shape}")
print(f"Matriz de similaridade: {cos_sim.shape}")

# %%
# Exemplo: perfumes mais similares ao Malbec Tradicional (id=1, índice 0)
exemplo_idx = 0
sim_scores = list(enumerate(cos_sim[exemplo_idx]))
sim_scores.sort(key=lambda x: x[1], reverse=True)

print(f"Perfumes mais similares a '{df_produtos.iloc[exemplo_idx]['nome']}':\n")
for i, (idx, score) in enumerate(sim_scores[1:6], 1):
    row = df_produtos.iloc[idx]
    print(f"  {i}. {row['nome']} ({row['marca']}) — similaridade: {score:.3f}")

# %% [markdown]
# ### 3.2 SVD — Filtragem colaborativa
#
# Implementação manual com `numpy.linalg.svd` (sem dependência do scikit-surprise).
# O SVD decompõe a matriz de utilidade em fatores latentes que capturam padrões
# de preferência ocultos (ex: "quem gosta de amadeirado tende a gostar de oriental").

# %%
import numpy as np

# Converte a matriz para float; 0 → NaN (não avaliado)
mat = df_matriz.values.astype(float)
mat[mat == 0] = np.nan

# Média por usuário (ignorando NaN)
user_means = np.nanmean(mat, axis=1)

# Imputação: substitui NaN pela média do usuário
mat_filled = mat.copy()
for i in range(mat_filled.shape[0]):
    mask = np.isnan(mat_filled[i])
    mat_filled[i, mask] = user_means[i] if not np.isnan(user_means[i]) else 3.0

# Centraliza (subtrai média de cada usuário)
mat_centered = mat_filled - user_means[:, np.newaxis]

# SVD completo, truncado para k=15 fatores latentes
U, sigma, Vt = np.linalg.svd(mat_centered, full_matrices=False)
k = 15
U_k = U[:, :k]
S_k = np.diag(sigma[:k])
Vt_k = Vt[:k, :]

# Reconstrução: predições de nota para todos os pares (usuário, perfume)
pred_matrix = U_k @ S_k @ Vt_k + user_means[:, np.newaxis]
pred_matrix = np.clip(pred_matrix, 1.0, 5.0)

# Variância explicada pelos k fatores
var_total = np.sum(sigma ** 2)
var_k = np.sum(sigma[:k] ** 2)
print(f"SVD: k={k} fatores latentes")
print(f"Variância explicada: {var_k / var_total * 100:.1f}%")
print(f"Matriz de predições: {pred_matrix.shape}")

# %% [markdown]
# ### 3.3 Pipeline de recomendação híbrida
#
# 1. O usuário informa suas preferências (família olfativa, ocasião, faixa de preço)
# 2. **TF-IDF** gera os 20 perfumes candidatos mais similares ao perfil
# 3. **SVD** reordena os candidatos usando as notas previstas de usuários similares
# 4. Retorna os **top-5** finais com peso 70% TF-IDF + 30% SVD

# %%
def recomendar_hibrido(familias_pref, ocasiao_pref, genero_pref, faixa_preco, top_n=5):
    """Recomenda perfumes usando pipeline TF-IDF + SVD."""

    # TF-IDF: gera score de conteúdo para o perfil do usuário
    perfil = " ".join(familias_pref) + " " + ocasiao_pref
    perfil_vec = tfidf.transform([perfil])
    scores_tfidf = cosine_similarity(perfil_vec, tfidf_matrix).flatten()

    # Filtros de gênero e preço
    mask = df_produtos['genero'].isin([genero_pref, 'unissex'])
    if faixa_preco == 'ate_100':
        mask &= df_produtos['preco'] <= 100
    elif faixa_preco == '100_200':
        mask &= (df_produtos['preco'] > 100) & (df_produtos['preco'] <= 200)
    elif faixa_preco == '200_300':
        mask &= (df_produtos['preco'] > 200) & (df_produtos['preco'] <= 300)
    else:
        mask &= df_produtos['preco'] > 300

    if mask.sum() == 0:
        mask = df_produtos['genero'].isin([genero_pref, 'unissex'])

    # Top-20 candidatos TF-IDF
    indices = df_produtos[mask].index.tolist()
    candidatos = sorted([(i, scores_tfidf[i]) for i in indices],
                        key=lambda x: x[1], reverse=True)[:20]
    top20 = [i for i, _ in candidatos]

    # SVD: média das predições de todos os usuários para cada candidato
    svd_scores = pred_matrix[:, top20].mean(axis=0)

    # Combinação: 70% TF-IDF + 30% SVD
    resultado = []
    for rank, idx in enumerate(top20):
        tfidf_s = scores_tfidf[idx]
        svd_s = (svd_scores[rank] - 1) / 4  # normaliza 1-5 → 0-1
        final = 0.7 * tfidf_s + 0.3 * svd_s
        resultado.append((idx, final, svd_scores[rank]))

    resultado.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print(f"Recomendações — {', '.join(familias_pref)} | {ocasiao_pref} | {genero_pref}")
    print(f"{'='*60}")
    for i, (idx, score, nota) in enumerate(resultado[:top_n], 1):
        row = df_produtos.iloc[idx]
        print(f"  {i}. {row['nome']} ({row['marca']}) — R${row['preco']:.2f}")
        print(f"     Família: {row['familia_olfativa']} | Nota SVD: {nota:.1f} | Score: {score:.3f}")

    return resultado[:top_n]

# Teste do pipeline
recomendar_hibrido(['amadeirado aromatico'], 'noturno-formal', 'masculino', '100_200')

# %% [markdown]
# ---
# ## 4. Interface com Gradio
#
# A interface Gradio está implementada em `app.py` na raiz do projeto.
# Para abrir no navegador:
# ```bash
# python app.py
# ```