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
# Avon contribui com 7 perfumes, Eudora com 4, Mahogany com 2, Granado e Phebo com 1 cada.

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
# O catálogo é bastante diversificado: 23 famílias olfativas diferentes. As mais representadas são
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
# O catálogo é equilibrado entre feminino e masculino, com 26 femininos e 24 masculinos.

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

# %%
notas = df_matriz.values.flatten()
notas = notas[notas > 0]  # exclui zeros (não avaliados)

sns.histplot(notas, bins=5, discrete=True)
plt.title("Distribuição das Notas (1–5)")
plt.xlabel("Nota")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

# %%[markdown]

#As notas concentram-se entre 3 e 4, sugerindo uma tendência de avaliações positivas moderadas.
#Notas extremas (1 e 5) são menos frequentes, o que é comum em sistemas de avaliação reais.

# %%

avaliacoes_por_usuario = (df_matriz > 0).sum(axis=1)

sns.histplot(avaliacoes_por_usuario, bins=20)
plt.title("Quantidade de Perfumes Avaliados por Usuário")
plt.xlabel("Perfumes avaliados")
plt.ylabel("Usuários")
plt.tight_layout()
plt.show()

# %%[markdown]

# A maioria dos usuários avaliou entre 5 e 20 perfumes. Os 66 usuários que não avaliaram nenhum
#perfume aparecem na barra mais a esquerda. Poucos usuários avaliaram mais de 25 perfumes,
#o que é esperado em um cenário de compras real.

# %%
avaliacoes_por_perfume = (df_matriz > 0).sum(axis=0).reset_index()
avaliacoes_por_perfume.columns = ['perfume_id', 'total_avaliacoes']
avaliacoes_por_perfume['id'] = avaliacoes_por_perfume['perfume_id'].str.extract(r'(\d+)').astype(int)
avaliacoes_por_perfume = avaliacoes_por_perfume.merge(
      df_produtos[['id', 'nome', 'genero', 'familia_olfativa', 'ocasiao']], on='id'
)

plt.figure(figsize=(10, 14))
sns.barplot(x='total_avaliacoes', y='nome',
            data=avaliacoes_por_perfume.sort_values('total_avaliacoes', ascending=True))
plt.title("Avaliações por Perfume")
plt.xlabel("Total de avaliações")
plt.ylabel("Perfume")
plt.tight_layout()
plt.show()

# %%

print('Média de avaliações por gênero')
print(avaliacoes_por_perfume.groupby('genero')['total_avaliacoes'].mean().round(1))
# %%[markdown]

# Perfumes femininos recebem em média 124,6 avaliações contra 111,0 dos masculinos.
# A diferença de ~12% pode indicar maior engajamento do público feminino na simulação,
# possivelmente porque há mais personas femininas geradas

# %% [markdown]
# ## 3. Treinamento do modelo
#
# ### 3.1 TF-IDF (filtragem por conteúdo)

# %%[markdown]

# Primeiro a ideia é criar o vetor corpus para que o TF-IDF consiga transformar os perfumes em vetores.
#  Essa transformação precisa juntar os textos das colunas familia_olfativa, notas_olfativas, ocasiao e genero.

# %% [markdown]

# As colunas marca e nome servirão apenas para visualização na interface e a coluna preço servirá como filtro mais futuramente.
#  Como o TF-IDF recebe somente texto e o preço é um dado do tipo float, ele não pode entrar nesse vetor corpus.

# %%
import unicodedata

def clean_text(texto):
    texto = str(texto).lower() # transforma para minuscula
    # separa a letra do acento e remove o acento ignorando tudo que nao é ascii
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = texto.replace(',', '') # remove virgula
    texto = texto.replace('-', ' ') # remove hifen
    return texto

# %% [markdown]
# A função clean_text normaliza o texto convertendo para minúsculo, removendo acentos via unicodedata,
# removendo vírgulas das notas olfativas e substituindo hífens por espaços na ocasião,
# para que noturno-formal vire noturno formal e o TF-IDF trate cada token separadamente.

# %%

df_produtos['familia_olfativa'] = df_produtos['familia_olfativa'].apply(clean_text)
df_produtos['notas_olfativas'] = df_produtos['notas_olfativas'].apply(clean_text)
df_produtos['ocasiao'] = df_produtos['ocasiao'].apply(clean_text)
df_produtos['genero'] = df_produtos['genero'].apply(clean_text)

# %%

df_produtos.head()

# %% [markdown]
# A normalização é aplicada nas quatro colunas que formarão o corpus.

# %%

df_produtos['corpus'] = df_produtos['familia_olfativa'] + " " + df_produtos['notas_olfativas'] + " " + df_produtos['ocasiao'] + " " + df_produtos['genero']
df_produtos['corpus'].head()

# %% [markdown]
# O corpus é criado concatenando as quatro colunas normalizadas em uma única string por perfume.
# Essa string representa o "documento" que o TF-IDF vai vetorizar.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
vectorized = vectorizer.fit_transform(df_produtos['corpus'])

# matriz de similaridade de cosseno entre os 50 perfumes (50×50)
matriz_similaridade = cosine_similarity(vectorized)

print(f"Matriz TF-IDF: {vectorized.shape}")
print(f"Matriz de similaridade: {matriz_similaridade.shape}")

# %% [markdown]
# O TfidfVectorizer aprende o vocabulário dos 50 perfumes e transforma cada corpus em um vetor numérico.
# Palavras raras recebem peso maior que palavras comuns, o que é mais adequado que uma simples contagem.
# A matriz resultante tem dimensão 50 × 90: 50 perfumes e 90 tokens únicos no vocabulário.
# A similaridade de cosseno compara cada par de perfumes com base nos seus vetores TF-IDF,
# retornando valores entre 0 (nenhuma similaridade) e 1 (idênticos). A matriz de similaridade é 50 × 50.

# %%

exemplo_idx = 0
sim_scores = list(enumerate(matriz_similaridade[exemplo_idx]))
sim_scores.sort(key=lambda x: x[1], reverse=True)

indices = [idx for idx, _ in sim_scores[1:6]]
scores  = [score for _, score in sim_scores[1:6]]

resultado = df_produtos.iloc[indices][['nome', 'marca', 'familia_olfativa', 'ocasiao', 'notas_olfativas']].copy()
resultado['similaridade'] = [round(s, 3) for s in scores]
resultado.reset_index(drop=True, inplace=True)

print(f"Perfumes mais similares a '{df_produtos.iloc[exemplo_idx]['nome']}':\n")
resultado

# %% [markdown]
# A validação com o Malbec Tradicional confirma que o modelo está funcionando corretamente:
# os perfumes retornados compartilham família olfativa amadeirado aromático e notas como
# bergamota e cedro.

# %% [markdown]
# ### 3.2 SVD — Filtragem colaborativa
#
# O SVD (Singular Value Decomposition) decompõe a matriz de avaliações (500×50) em três fatores:
# U (usuários × fatores), sigma (importância de cada fator) e Vt (fatores × perfumes).
# Os fatores latentes capturam padrões ocultos de preferência, como "usuários que gostam de
# amadeirado tendem a gostar de oriental". 
# A matriz reconstruída contém previsões de nota para todos os pares usuário-perfume,
# inclusive os não avaliados, que é exatamente o que permite a filtragem colaborativa.

# %%
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

matrix = df_matriz.values.astype(float)
train_raw, test_raw = train_test_split(matrix, test_size=0.2, random_state=42)

# %%

def preparar_svd(mat):
    # converte zeros para NaN (não avaliado ≠ nota 0)
    m = mat.astype(float).copy()
    m[m == 0] = np.nan

    # média global de todas as notas reais (usada para usuários silenciosos)
    media_global = np.nanmean(m)

    # média por usuário ignorando NaN
    user_means = np.nanmean(m, axis=1)
    # usuários silenciosos recebem a média global — mais honesto que um valor fixo arbitrário
    user_means = np.where(np.isnan(user_means), media_global, user_means)

    # substitui NaN pela média do próprio usuário
    m_filled = m.copy()
    for i in range(m_filled.shape[0]):
        mascara = np.isnan(m_filled[i])
        m_filled[i, mascara] = user_means[i]

    # centraliza subtraindo a média de cada usuário
    m_centered = m_filled - user_means[:, np.newaxis]

    return m_centered, user_means

# %% [markdown]
# A comparação para u1 mostra que o SVD captura bem as preferências mais altas — perfumes com
# nota 5 recebem previsões entre 4.0 e 4.5. Nas notas mais baixas o modelo é menos preciso:
# o perfume com nota 1 recebeu previsão 3.10, acima de alguns com nota 2. Isso é uma limitação
# esperada do SVD com dados esparsos — a ordenação no topo é confiável, mas os extremos
# negativos são suavizados.

# %%
from scipy.linalg import svd as scipy_svd

k = 15  # número de fatores latentes

# treina o SVD com o conjunto de treino
train_centered, train_means = preparar_svd(train_raw)
U, sigma, Vt = scipy_svd(train_centered, full_matrices=False)

# projeta os usuários de teste no espaço aprendido pelo treino
# W_test = test_centered @ Vt.T / sigma (equivalente ao transform do NMF)
test_centered, test_means = preparar_svd(test_raw)
W_test = test_centered @ Vt[:k, :].T / sigma[:k]
pred_test_svd = W_test @ np.diag(sigma[:k]) @ Vt[:k, :] + test_means[:, np.newaxis]
pred_test_svd = np.clip(pred_test_svd, 1.0, 5.0)

# calcula RMSE apenas nas células que tinham nota real (ignora zeros)
mask_test = test_raw > 0
rmse = np.sqrt(mean_squared_error(test_raw[mask_test], pred_test_svd[mask_test]))
print(f"RMSE no conjunto de teste: {rmse:.4f}")

# variância explicada pelos k fatores latentes
var_total = np.sum(sigma ** 2)
var_k     = np.sum(sigma[:k] ** 2)
print(f"Variância explicada pelos {k} fatores: {var_k / var_total * 100:.1f}%")

# %%

# treina o SVD final com a matriz completa dos 500 usuários
mat_centered, user_means = preparar_svd(matrix)
U, sigma, Vt = np.linalg.svd(mat_centered, full_matrices=False)

# reconstrói a matriz completa de predições (500×50)
pred_matrix = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :] + user_means[:, np.newaxis]
pred_matrix = np.clip(pred_matrix, 1.0, 5.0)

print(f"Matriz de predições: {pred_matrix.shape}")

# %%

# valida comparando previsão vs notas reais do usuário u1
usuario_idx = df_matriz.index.get_loc('u1')
real     = df_matriz.iloc[usuario_idx]
previsto = pd.Series(pred_matrix[usuario_idx], index=df_matriz.columns).round(2)

comparacao = pd.DataFrame({'real': real, 'previsto': previsto})
comparacao[comparacao['real'] > 0].sort_values('real', ascending=False)

# %% [markdown]
# A comparação para u1 mostra que perfumes com nota 5 recebem previsões próximas de 4.5 e
# o perfume com nota 1 recebe a menor previsão. O SVD captura bem a ordenação das preferências.
# A matriz pred_matrix será usada na função de recomendação híbrida para reordenar os candidatos
# do TF-IDF com base no gosto de usuários com perfil similar ao novo usuário.

# %% [markdown]
# ### 3.3 Pipeline de recomendação híbrida
#
# 1. O usuário informa suas preferências (família olfativa, ocasião, faixa de preço)
# 2. **TF-IDF** gera os 20 perfumes candidatos mais similares ao perfil
# 3. **SVD** encontra os usuários com perfil mais similar e usa as notas previstas deles
# 4. Retorna os **top-5** finais com peso 70% TF-IDF + 30% SVD

# %%
def recomendar_hibrido(familias_pref, ocasiao_pref, genero_pref, faixa_preco, top_n=5):

    # TF-IDF: gera score de conteúdo para o perfil do usuário
    perfil = " ".join(familias_pref) + " " + clean_text(ocasiao_pref) + " " + clean_text(genero_pref)
    perfil_vec = vectorizer.transform([perfil])
    scores_tfidf = cosine_similarity(perfil_vec, vectorized).flatten()

    # filtro de gênero
    mask = df_produtos['genero'] == clean_text(genero_pref)

    # filtro de preço
    if faixa_preco == 'ate_100':
        mask &= df_produtos['preco'] <= 100
    elif faixa_preco == '100_200':
        mask &= (df_produtos['preco'] > 100) & (df_produtos['preco'] <= 200)
    elif faixa_preco == '200_300':
        mask &= (df_produtos['preco'] > 200) & (df_produtos['preco'] <= 300)
    else:
        mask &= df_produtos['preco'] > 300

    # se nenhum perfume passar no filtro de preço, ignora o filtro
    if mask.sum() == 0:
        mask = df_produtos['genero'] == clean_text(genero_pref)

    # top-20 candidatos pelo TF-IDF
    indices = df_produtos[mask].index.tolist()
    candidatos = sorted([(i, scores_tfidf[i]) for i in indices],
                        key=lambda x: x[1], reverse=True)[:20]
    top20 = [i for i, _ in candidatos]

    # SVD: encontra os 50 usuários cujas notas previstas mais se alinham ao perfil do novo usuário
    # usando os scores TF-IDF dos candidatos como proxy do perfil
    scores_perfil = scores_tfidf[top20]
    pesos_usuarios = pred_matrix[:, top20] @ scores_perfil  # alinhamento de cada usuário ao perfil
    top_usuarios = np.argsort(pesos_usuarios)[::-1][:50]    # top-50 usuários mais similares
    svd_scores = pred_matrix[top_usuarios][:, top20].mean(axis=0)  # média das notas deles

    # combinação: 70% TF-IDF + 30% SVD (SVD normalizado para 0-1)
    resultado = []
    for rank, idx in enumerate(top20):
        tfidf_s = scores_tfidf[idx]
        svd_s   = (svd_scores[rank] - 1) / 4  # normaliza 1-5 → 0-1
        final   = 0.7 * tfidf_s + 0.3 * svd_s
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

# %%

# teste do pipeline
recomendar_hibrido(['amadeirado aromatico'], 'noturno-formal', 'masculino', '100_200')

# %% [markdown]
# ## 4. Salvamento dos modelos
#
# Os modelos são salvos em disco para que o app.py (Gradio) possa carregá-los diretamente,
# sem precisar retreinar toda vez que a interface for aberta.
#
# São salvos:
# - **vectorizer** — o TfidfVectorizer treinado, necessário para vetorizar o perfil do novo usuário
# - **vectorized** — a matriz TF-IDF dos 50 perfumes (50×90), usada no cálculo de similaridade
# - **pred_matrix** — a matriz SVD de predições (500×50), usada para reordenar os candidatos
# - **df_produtos** — o catálogo já normalizado, com a coluna corpus criada

# %%
import joblib
import os

os.makedirs('../modelos', exist_ok=True)

joblib.dump(vectorizer,  '../modelos/tfidf_vectorizer.pkl')
joblib.dump(vectorized,  '../modelos/tfidf_matrix.pkl')
joblib.dump(pred_matrix, '../modelos/svd_pred_matrix.pkl')
df_produtos.to_csv('../dados/produtos_processados.csv', index=False)

print("Modelos salvos em ../modelos/")
print(f" tfidf_vectorizer.pkl")
print(f" tfidf_matrix.pkl")
print(f" svd_pred_matrix.pkl")
print(f"../dados/produtos_processados.csv")
