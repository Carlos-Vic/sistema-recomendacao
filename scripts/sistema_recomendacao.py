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

phebo = df_produtos[df_produtos['nome'] == 'Phebo Agua de Folhas de Figo']
phebo

# %%

print('Média de avaliações por gênero')
print(avaliacoes_por_perfume.groupby('genero')['total_avaliacoes'].mean().round(1))
# %%[markdown]

#O perfume com mais avaliações é o Phebo Água de Folhas de Figo, único unissex do catálogo.
#Por não ter barreira de gênero, todos os 500 usuários simulados têm chance de avaliá-lo,

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

vectorizer = TfidfVectorizer()
vectorized = vectorizer.fit_transform(df_produtos['corpus'])

# %%
from sklearn.metrics.pairwise import cosine_similarity

matriz_similaridade = cosine_similarity(vectorized)
print(matriz_similaridade)

# %%

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

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

matrix = df_matriz.values.astype(float)

# separação treino/teste
train_raw, test_raw = train_test_split(matrix, test_size=0.2, random_state=42)

# %%
from sklearn.decomposition import NMF

nmf = NMF(n_components=20, random_state=42, max_iter=500)
W_train = nmf.fit_transform(train_raw)
W_test  = nmf.transform(test_raw)

pred_test= np.dot(W_test, nmf.components_)
rmse = np.sqrt(mean_squared_error(test_raw, pred_test))
print(f"RMSE com zeros: {rmse:.4f}")

# %%

W_full = nmf.transform(matrix)
pred_full = np.dot(W_full, nmf.components_)
df_pred = pd.DataFrame(pred_full, columns=df_matriz.columns, index=df_matriz.index)

usuario = 'u1'
comparacao = pd.DataFrame({
      'real': df_matriz.loc[usuario],
      'previsto': df_pred.loc[usuario].round(2)
})
comparacao[comparacao_zeros['real'] > 0].sort_values('real', ascending=False)

# %%

def preencher_media(mat):
      m = mat.copy()
      contagem = (m != 0).sum(axis=1)
      soma = m.sum(axis=1)
      means = np.divide(soma, contagem, out=np.zeros_like(soma), where=contagem != 0)
      for i, mean in enumerate(means):
          m[i][m[i] == 0] = mean
      return m

train_filled = preencher_media(train_raw)
test_filled  = preencher_media(test_raw)

nmf_final = NMF(n_components=20, random_state=42)
W_train = nmf_final.fit_transform(train_filled)
W_test  = nmf_final.transform(test_filled)

pred_test_filled = np.dot(W_test, nmf_final.components_)
rmse_filled = np.sqrt(mean_squared_error(test_filled, pred_test_filled))
print(f"RMSE com zeros substituídos: {rmse_filled:.4f}")

# %% 
matrix_filled = preencher_media(matrix)
W_full = nmf_final.transform(matrix_filled)
predicted_full = np.dot(W_full, nmf_final.components_)
df_predicted = pd.DataFrame(predicted_full, columns=df_matriz.columns, index=df_matriz.index)

comparacao_filled = pd.DataFrame({
      'real': df_matriz.loc[usuario],
      'previsto': df_predicted.loc[usuario].round(2)
})
comparacao_filled[comparacao_filled['real'] > 0].sort_values('real', ascending=False)

# %%
