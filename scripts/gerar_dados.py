"""
Geração dos dados do Sistema de Recomendação de Perfumes Nacionais.

Este script produz três arquivos a partir de dados fixos (catálogo) e geração
sintética controlada (matriz de utilidade):

    dados/produtos.csv         — 50 perfumes nacionais com 3 características
                                  (família olfativa, notas olfativas, ocasião)
                                  + preço pesquisado nas fontes oficiais
    dados/matriz_utilidade.csv — 500 usuários x 50 perfumes, notas 1-5
                                  (0 = produto não avaliado; ~12% dos usuários
                                  são silenciosos: compraram mas nunca avaliaram)

Reprodutibilidade: seed=42 garante que a matriz gerada seja sempre idêntica.

Execução:
    python gerar_dados.py
"""

import csv
import random
import re
import unicodedata
from pathlib import Path

# Seed fixa: garante que a matriz de utilidade seja reproduzível —
# toda execução do script gera exatamente os mesmos dados.
SEED = 42
random.seed(SEED)

# Caminhos base do projeto (relativo à localização deste script)
BASE = Path(__file__).parent
DADOS = BASE / "dados"
IMAGENS = BASE / "imagens"
DADOS.mkdir(exist_ok=True)
IMAGENS.mkdir(exist_ok=True)
(IMAGENS / ".gitkeep").touch()  # mantém a pasta no git mesmo sem imagens


def slug(s: str) -> str:
    """Converte uma string em slug ASCII para uso como nome de arquivo.

    Ex: "Egeo Cherry Blast" → "egeo_cherry_blast"
    Remove acentos, caracteres especiais e substitui espaços por underscore.
    """
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    return re.sub(r"[-\s]+", "_", s)


# ---------------------------------------------------------------------------
# 1) Catálogo de 50 perfumes nacionais
#
#    Cada tupla contém: (nome, marca, gênero, família_olfativa, notas, ocasião)
#
#    As 3 características usadas pelo modelo TF-IDF são:
#      - família_olfativa : categoria olfativa principal (ex: "floral amadeirado")
#      - notas_olfativas  : pirâmide olfativa simplificada em 5 notas-chave
#      - ocasião          : contexto de uso (ex: "noturno-formal", "diurno-verao")
#
#    Fonte das notas e famílias: sites oficiais das marcas e Fragrantica.
# ---------------------------------------------------------------------------

PERFUMES = [
    # ---- O Boticário --------------------------------------------------------
    ("Malbec Tradicional", "O Boticário", "masculino", "amadeirado aromatico",
     "bergamota, lavanda, cedro, vetiver, patchouli", "noturno-formal"),
    ("Malbec Black", "O Boticário", "masculino", "amadeirado especiado",
     "canela, pimenta rosa, cedro, couro, whisky", "noturno-inverno"),
    ("Lily Essence", "O Boticário", "feminino", "floral",
     "lirio, rosa, iris, ambar, baunilha", "noturno-formal"),
    ("Lily Absolu", "O Boticário", "feminino", "floral oriental",
     "lirio, jasmim, patchouli, baunilha, ambar", "noturno-inverno"),
    ("Coffee Man Duo", "O Boticário", "masculino", "amadeirado aromatico",
     "cafe, bergamota, gengibre, patchouli, sandalo", "diurno-casual"),
    ("Coffee Woman Seduction", "O Boticário", "feminino", "gourmand",
     "cafe, flor de laranjeira, chocolate, baunilha, ambar", "noturno-casual"),
    ("Floratta Rosas", "O Boticário", "feminino", "floral",
     "rosa, pessego, jasmim, almiscar, ambar", "diurno-casual"),
    ("Floratta Blue", "O Boticário", "feminino", "floral aquatico",
     "violeta, jasmim, iris, almiscar, sandalo", "diurno-casual"),
    ("Egeo Dolce", "O Boticário", "feminino", "gourmand",
     "pistache, chantilly, baunilha, sandalo, almiscar", "diurno-casual"),
    ("Egeo Cherry Blast", "O Boticário", "feminino", "floral oriental frutal",
     "cereja, caramelo, baunilha, framboesa, bergamota", "diurno-casual"),
    ("Quasar", "O Boticário", "masculino", "amadeirado aromatico",
     "bergamota, lavanda, salvia, cedro, almiscar", "diurno-casual"),
    ("Zaad Classic", "O Boticário", "masculino", "amadeirado oriental",
     "pimenta rosa, cardamomo, cedro, ambar, almiscar", "noturno-formal"),
    ("Zaad Intenso", "O Boticário", "masculino", "amadeirado oriental",
     "cardamomo, pimenta, ebano, couro, ambar", "noturno-inverno"),
    ("Uomini", "O Boticário", "masculino", "amadeirado aromatico",
     "bergamota, manjericao, geranio, cedro, vetiver", "diurno-casual"),
    ("Glamour Secrets Black", "O Boticário", "feminino", "floral frutal",
     "pera, maca, jasmim, rosa, patchouli", "diurno-casual"),

    # ---- Natura -------------------------------------------------------------
    ("Essencial Masculino", "Natura", "masculino", "amadeirado",
     "breu branco, mate, cumaru, almiscar, pau-rosa", "noturno-formal"),
    ("Essencial Exclusivo Feminino", "Natura", "feminino", "floral amadeirado",
     "rosa, jasmim, breu, ambar, patchouli", "noturno-formal"),
    ("Kaiak Masculino", "Natura", "masculino", "aromatico",
     "bergamota, limao, geranio, lavanda, almiscar", "diurno-casual"),
    ("Kaiak Feminino", "Natura", "feminino", "citrico floral",
     "limao, laranja, jasmim, rosa, almiscar", "diurno-verao"),
    ("Humor Feminino", "Natura", "feminino", "floral frutal",
     "pessego, framboesa, jasmim, magnolia, almiscar", "diurno-casual"),
    ("Una Deo Parfum", "Natura", "feminino", "floral oriental",
     "tuberosa, jasmim, baunilha, patchouli, ambar", "noturno-formal"),
    ("Una Artisan", "Natura", "feminino", "chipre floral",
     "rosa, iris, patchouli, musgo, ambar", "noturno-formal"),
    ("Kriska Drama", "Natura", "feminino", "oriental baunilha",
     "baunilha, caramelo, patchouli, frutas vermelhas, ambar", "noturno-casual"),
    ("Biografia Masculino", "Natura", "masculino", "amadeirado",
     "pimenta, gengibre, cumaru, sandalo, ambar", "noturno-casual"),
    ("Homem Natura", "Natura", "masculino", "amadeirado aromatico",
     "bergamota, lavanda, cedro, pau-rosa, almiscar", "diurno-formal"),
    ("Ilia", "Natura", "feminino", "floral",
     "flor de laranjeira, iris, ambar, almiscar, sandalo", "diurno-casual"),
    ("Todo Dia Cereja e Avela", "Natura", "feminino", "floral frutal gourmand",
     "cereja, avela, baunilha, jasmim, chocolate", "diurno-casual"),

    # ---- Eudora -------------------------------------------------------------
    ("Niina Secrets Bloom", "Eudora", "feminino", "floral",
     "rosa, peonia, jasmim, almiscar, madeira", "diurno-casual"),
    ("Eudora Rose", "Eudora", "feminino", "chipre frutal",
     "champanhe rose, pera, maca, rosa, patchouli", "noturno-formal"),
    ("Eudora H Flow", "Eudora", "masculino", "aromatico fougere aquatico",
     "bergamota, notas aquosas, pimenta rosa, sal, vetiver", "diurno-verao"),
    # Malbec Gold/Elegant/Icon: variantes premium confirmadas em boticario.com.br
    ("Malbec Gold", "O Boticário", "masculino", "amadeirado oriental",
     "cardamomo, cumaru, baunilha, ambar, cedro", "noturno-inverno"),
    ("Malbec Elegant", "O Boticário", "masculino", "amadeirado aromatico",
     "bergamota, pimenta, vetiver, cedro, ambar", "noturno-formal"),
    ("Lyra Happy", "Eudora", "feminino", "chipre frutal",
     "framboesa, bergamota, magnolia, rosa, baunilha", "diurno-casual"),
    ("Malbec Icon", "O Boticário", "masculino", "amadeirado aromatico",
     "bergamota, limao, cardamomo, patchouli, cedro", "noturno-casual"),
    # Kaiak Oceano: confirmado em natura.com.br (busca anterior)
    ("Kaiak Oceano Masculino", "Natura", "masculino", "aromatico aquatico",
     "algas marinhas, pataqueira, sal, ambar, madeira", "diurno-verao"),

    # ---- Avon ---------------------------------------------------------------
    ("Little Black Dress", "Avon", "feminino", "floral chipre",
     "groselha, pimenta, jasmim, patchouli, almiscar", "noturno-formal"),
    ("Far Away", "Avon", "feminino", "oriental floral",
     "tangerina, freesia, jasmim, sandalo, baunilha", "noturno-formal"),
    ("Luck for Him", "Avon", "masculino", "amadeirado",
     "bergamota, gengibre, cedro, couro, ambar", "diurno-formal"),
    ("Luck for Her", "Avon", "feminino", "floral oriental",
     "groselha, peonia, jasmim, patchouli, almiscar", "noturno-casual"),
    ("Today", "Avon", "feminino", "floral",
     "freesia, lirio, rosa, sandalo, almiscar", "diurno-formal"),
    ("Full Speed", "Avon", "masculino", "aromatico",
     "bergamota, cardamomo, lavanda, cedro, almiscar", "diurno-casual"),
    ("Derek", "Avon", "masculino", "amadeirado aromatico",
     "bergamota, lavanda, salvia, vetiver, almiscar", "diurno-casual"),

    # ---- Kaiak (variantes confirmadas em natura.com.br) ---------------------
    ("Kaiak Aventura Masculino", "Natura", "masculino", "citrico aromatico",
     "bergamota, pimenta preta, cardamomo, sandalo, almiscar", "diurno-casual"),
    ("Kaiak Urbe Masculino", "Natura", "masculino", "aromatico aquatico",
     "limao, bergamota, copaiba, ambar, cedro", "diurno-casual"),
    ("Kaiak Pulso Masculino", "Natura", "masculino", "aromatico",
     "bergamota, gengibre, pimenta, cedro, ambar", "diurno-casual"),
    ("Kaiak O2 Masculino", "Natura", "masculino", "aromatico aquatico",
     "notas aquosas, bergamota, ambar marinho, sandalo, almiscar", "diurno-verao"),

    # ---- Granado / Phebo / Mahogany -----------------------------------------
    ("Granado Pink", "Granado", "feminino", "floral",
     "rosa, peonia, violeta, almiscar, madeira", "diurno-casual"),
    ("Phebo Agua de Folhas de Figo", "Phebo", "unissex", "aromatico verde",
     "folha de figo, bergamota, notas aquosas, vetiver, patchouli", "diurno-casual"),
    ("Mahogany Flor de Cerejeira", "Mahogany", "feminino", "floral",
     "flor de cerejeira, rosa, violeta, pessego, almiscar", "diurno-casual"),
    ("Mahogany Eau Intense", "Mahogany", "masculino", "amadeirado",
     "bergamota, pimenta, cedro, couro, ambar", "noturno-formal"),
]

assert len(PERFUMES) == 50, f"Esperado 50 perfumes, obtido {len(PERFUMES)}"

# ---------------------------------------------------------------------------
# Preços pesquisados nas fontes oficiais e Mercado Livre (abril 2026, 100ml).
# A lista está na mesma ordem que PERFUMES para ser usada com zip().
# ---------------------------------------------------------------------------

PRECOS = [
    189.90,  # 01 Malbec Tradicional       — boticario.com.br
    179.90,  # 02 Malbec Black             — boticario.com.br
    159.90,  # 03 Lily Essence             — linha Lily
    319.90,  # 04 Lily Absolu              — boticario.com.br (EDP 75ml)
    149.90,  # 05 Coffee Man Duo           — linha Coffee
    149.90,  # 06 Coffee Woman Seduction   — linha Coffee
    164.90,  # 07 Floratta Rosas           — boticario.com.br
    164.90,  # 08 Floratta Blue            — boticario.com.br
    149.90,  # 09 Egeo Dolce              — linha Egeo
    154.90,  # 10 Egeo Cherry Blast        — linha Egeo
    139.90,  # 11 Quasar                  — boticario.com.br
    229.90,  # 12 Zaad Classic             — boticario.com.br (EDP)
    299.90,  # 13 Zaad Intenso             — boticario.com.br (EDP 95ml)
    139.90,  # 14 Uomini                  — boticario.com.br
    169.90,  # 15 Glamour Secrets Black    — linha Glamour
    149.90,  # 16 Essencial Masculino      — natura.com.br
    169.90,  # 17 Essencial Exclusivo Fem  — natura.com.br
    129.90,  # 18 Kaiak Masculino          — natura.com.br
    129.90,  # 19 Kaiak Feminino           — natura.com.br
    129.90,  # 20 Humor Feminino           — natura.com.br
    219.90,  # 21 Una Deo Parfum           — natura.com.br (Deo Parfum)
    299.90,  # 22 Una Artisan              — natura.com.br (EDP 75ml)
    119.90,  # 23 Kriska Drama             — mercadolivre.com.br
    139.90,  # 24 Biografia Masculino      — natura.com.br
    139.90,  # 25 Homem Natura             — natura.com.br
    159.90,  # 26 Ilia                    — natura.com.br
     79.90,  # 27 Todo Dia Cereja e Avela  — linha Todo Dia (acessível)
    179.90,  # 28 Niina Secrets Bloom      — eudora.com.br
    219.90,  # 29 Eudora Rose              — eudora.com.br (2024)
    179.90,  # 30 Eudora H Flow            — eudora.com.br (2024)
    219.90,  # 31 Malbec Gold              — boticario.com.br
    209.90,  # 32 Malbec Elegant           — boticario.com.br
    189.90,  # 33 Lyra Happy               — eudora.com.br (2025)
    199.90,  # 34 Malbec Icon              — boticario.com.br
    139.90,  # 35 Kaiak Oceano Masculino   — natura.com.br
     99.90,  # 36 Little Black Dress       — avon.com.br
     89.90,  # 37 Far Away                — avon.com.br
     79.90,  # 38 Luck for Him             — avon.com.br
     79.90,  # 39 Luck for Her             — avon.com.br
     79.90,  # 40 Today                   — avon.com.br
     79.90,  # 41 Full Speed              — avon.com.br
     79.90,  # 42 Derek                   — avon.com.br
    139.90,  # 43 Kaiak Aventura Masculino — natura.com.br
    139.90,  # 44 Kaiak Urbe Masculino     — natura.com.br
    139.90,  # 45 Kaiak Pulso Masculino    — natura.com.br
    139.90,  # 46 Kaiak O2 Masculino       — natura.com.br
     69.90,  # 47 Granado Pink             — granado.com.br
     99.90,  # 48 Phebo Agua Folhas de Figo— phebo.com.br (50ml)
    159.90,  # 49 Mahogany Flor Cerejeira  — mercadolivre.com.br
    149.90,  # 50 Mahogany Eau Intense     — mahogany.com.br
]

assert len(PRECOS) == 50, f"Esperado 50 preços, obtido {len(PRECOS)}"

# ---------------------------------------------------------------------------
# Escrita de produtos.csv e checklist_imagens.md
# ---------------------------------------------------------------------------

produtos_path = DADOS / "produtos.csv"

with produtos_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "nome", "marca", "genero",
                "familia_olfativa", "notas_olfativas", "ocasiao", "preco", "imagem_path"])

    for i, ((nome, marca, gen, fam, notas, ocas), preco) in enumerate(
            zip(PERFUMES, PRECOS), start=1):
        base = f"{i:02d}_{slug(nome)}"
        caminho = f"imagens/{base}"
        w.writerow([i, nome, marca, gen, fam, notas, ocas, preco, caminho])

print(f"[OK] {produtos_path} — {len(PERFUMES)} perfumes")

# ---------------------------------------------------------------------------
# Geração da matriz de utilidade (sintética, mas internamente consistente)
#
# Modelo de personas:
#   Cada usuário recebe uma "persona" com:
#     - familias_pref : lista de 1-2 famílias olfativas favoritas
#     - ocasiao_pref  : ocasião de uso preferida (diurno/noturno, etc.)
#     - genero_pref   : conjunto de gêneros aceitáveis (masc, fem ou ambos)
#
# A nota de cada perfume é calculada assim:
#   nota_base = 2.5 (neutro)
#   + até 1.6 se a família olfativa do perfume coincide com a preferida
#   + até 0.6 se a ocasião coincide
#   - 2.2 se o gênero do perfume não é aceito pela persona
#   + ruído gaussiano N(0, 0.45) para simular variabilidade natural
#
# A probabilidade de um usuário avaliar cada perfume depende do match:
#   - Gênero incompatível → 4% de chance (raramente compra)
#   - Família favorita    → 65% de chance (muito provável que já usou)
#   - Outros             → 22% de chance (descoberta casual)
#
# Além disso, ~12% dos usuários são "silenciosos": têm persona definida mas
# nunca avaliaram nenhum perfume — comportamento comum em lojas reais.
# ---------------------------------------------------------------------------

N_USUARIOS = 500      # atende o requisito de "pelo menos 100 linhas" do enunciado
PROB_SILENCIOSO = 0.12  # ~12% dos usuários compram mas nunca avaliam nada

# extrai os valores únicos de família e ocasião do catálogo
FAMILIAS = sorted(set(p[3] for p in PERFUMES))
OCASIOES = sorted(set(p[5] for p in PERFUMES))


def gerar_persona():
    """Cria um perfil aleatório de preferências para um usuário."""
    n_favs = random.choice([1, 2])  # 1 ou 2 famílias favoritas
    familias_pref = random.sample(FAMILIAS, n_favs)
    ocasiao_pref = random.choice(OCASIOES)

    # distribuição de gênero: 45% masc, 45% fem, 10% aceita ambos
    r = random.random()
    if r < 0.45:
        genero_pref = {"masculino", "unissex"}
    elif r < 0.90:
        genero_pref = {"feminino", "unissex"}
    else:
        genero_pref = {"masculino", "feminino", "unissex"}

    return familias_pref, ocasiao_pref, genero_pref


def score(perfume, persona):
    """Calcula a nota esperada (1-5) de um perfume para uma persona.

    O cálculo usa sobreposição de tokens entre os atributos do perfume
    e as preferências da persona. Ex: família "floral amadeirado" tem
    sobreposição parcial com preferência "amadeirado aromatico".
    """
    _, _, gen, fam, _, ocas = perfume
    familias_pref, ocasiao_pref, genero_pref = persona

    s = 2.5  # ponto de partida neutro

    # match de família olfativa (maior peso: ±1.6)
    tokens_prod_fam = set(fam.split())
    for f_pref in familias_pref:
        tokens_pref = set(f_pref.split())
        inter = tokens_pref & tokens_prod_fam
        if inter:
            # sobreposição proporcional: "floral" dentro de "floral oriental" = 0.5
            s += 1.6 * len(inter) / len(tokens_pref)
            break  # considera apenas a família favorita de melhor match

    # match de ocasião (peso moderado: ±0.6)
    toks_pref_ocas = set(ocasiao_pref.split("-"))
    toks_prod_ocas = set(ocas.split("-"))
    inter = toks_pref_ocas & toks_prod_ocas
    if inter:
        s += 0.6 * len(inter) / len(toks_pref_ocas)

    # penalidade forte se gênero é incompatível (-2.2)
    if gen not in genero_pref:
        s -= 2.2

    # ruído gaussiano: simula variabilidade nas preferências individuais
    s += random.gauss(0, 0.45)

    # garante que a nota fique entre 1 e 5
    return max(1.0, min(5.0, s))


def prob_avaliar(perfume, persona):
    """Retorna a probabilidade de um usuário ter avaliado um perfume.

    Usuários tendem a comprar e avaliar perfumes compatíveis com seu perfil.
    Perfumes de gênero incompatível raramente são avaliados (4%).
    """
    _, _, gen, fam, _, _ = perfume
    familias_pref, _, genero_pref = persona

    if gen not in genero_pref:
        return 0.04  # quase nunca compra perfume de gênero errado

    for f_pref in familias_pref:
        if set(f_pref.split()) & set(fam.split()):
            return 0.65  # família favorita: alta chance de ter experimentado

    return 0.22  # outros: descoberta casual (amigo indicou, promoção, etc.)


# escreve a matriz: linhas = usuários, colunas = perfumes (p1..p50)
# 0 indica que o usuário não avaliou aquele perfume
matriz_path = DADOS / "matriz_utilidade.csv"
total = 0

with matriz_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["usuario_id"] + [f"p{i+1}" for i in range(len(PERFUMES))])

    for u in range(1, N_USUARIOS + 1):
        persona = gerar_persona()
        linha = [f"u{u}"]

        # usuários silenciosos: compraram mas nunca deixaram avaliação
        if random.random() < PROB_SILENCIOSO:
            linha.extend([0] * len(PERFUMES))
        else:
            for perf in PERFUMES:
                if random.random() < prob_avaliar(perf, persona):
                    linha.append(int(round(score(perf, persona))))
                    total += 1
                else:
                    linha.append(0)

        w.writerow(linha)

densidade = total / (N_USUARIOS * len(PERFUMES)) * 100
print(f"[OK] {matriz_path} — {N_USUARIOS} usuários x {len(PERFUMES)} perfumes")
print(f"     {total} avaliações (~{total / N_USUARIOS:.1f} por usuário, "
      f"densidade {densidade:.1f}%)")
