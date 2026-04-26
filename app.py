import warnings
warnings.filterwarnings("ignore")

import csv
import json
import numpy as np
import pandas as pd
import joblib
import base64
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

BASE = Path(__file__).parent
DADOS    = BASE / "dados"
MODELOS  = BASE / "modelos"
USUARIOS_JSON = BASE / "usuarios.json"
FEEDBACK_CSV  = DADOS / "feedback.csv"

# carrega modelos treinados pelo notebook — sem retreino
df_produtos = pd.read_csv(DADOS / "produtos_processados.csv")
vectorizer  = joblib.load(MODELOS / "tfidf_vectorizer.pkl")
vectorized  = joblib.load(MODELOS / "tfidf_matrix.pkl")
pred_matrix = joblib.load(MODELOS / "svd_pred_matrix.pkl")

FAMILIAS = sorted(df_produtos["familia_olfativa"].unique().tolist())
OCASIOES = sorted(df_produtos["ocasiao"].unique().tolist())
GENEROS  = ["feminino", "masculino"]


def obter_top5_mockados(genero_pref):
    """
    Retorna os 5 perfumes 'mais vendidos' mockados para cada gênero.
    Estes são perfumes bem-avaliados e populares do catálogo.
    """
    # Mapeamento de top-5 mockados por gênero
    top5_por_genero = {
        "feminino": [
            (18, 4.8, 4.8),   # Kaiak Feminino - citrico floral, diurno verão
            (2, 4.6, 4.6),    # Lily Essence - floral, noturno formal
            (5, 4.7, 4.7),    # Coffee Woman Seduction - gourmand
            (16, 4.5, 4.5),   # Essencial Exclusivo Feminino - floral amadeirado
            (6, 4.6, 4.6),    # Floratta Rosas - floral clássico
        ],
        "masculino": [
            (10, 4.7, 4.7),   # Quasar - amadeirado aromatico, diurno
            (0, 4.8, 4.8),    # Malbec Tradicional - amadeirado, noturno
            (4, 4.6, 4.6),    # Coffee Man Duo - amadeirado, diurno
            (17, 4.7, 4.7),   # Kaiak Masculino - aromatico, diurno
            (15, 4.5, 4.5),   # Essencial Masculino - amadeirado, noturno
        ]
    }
    return top5_por_genero.get(genero_pref, top5_por_genero["feminino"])


def validar_inputs(familias_pref, ocasiao_pref, genero_pref, preco_max):
    """
    Valida e normaliza os inputs do usuário.
    Fornece valores padrão se campos estiverem None ou vazios.
    """
    # Validar e normalizar familias_pref
    if not familias_pref:
        familias_pref = ["floral"]  # Padrão
    
    # Validar e normalizar ocasiao_pref
    if not ocasiao_pref or ocasiao_pref == "null":
        ocasiao_pref = "diurno casual"  # Padrão (com espaço)
    
    # Validar e normalizar genero_pref
    if not genero_pref or genero_pref not in ["feminino", "masculino"]:
        genero_pref = "feminino"  # Padrão
    
    # Validar preco_max
    if preco_max is None or preco_max <= 0:
        preco_max = 200  # Padrão
    
    return familias_pref, ocasiao_pref, genero_pref, preco_max


def recomendar_hibrido(familias_pref, ocasiao_pref, genero_pref, preco_max, top_n=5):
    # Validar inputs
    familias_pref, ocasiao_pref, genero_pref, preco_max = validar_inputs(
        familias_pref, ocasiao_pref, genero_pref, preco_max
    )
    
    perfil = " ".join(familias_pref) + " " + ocasiao_pref + " " + genero_pref
    perfil_vec = vectorizer.transform([perfil])
    scores_tfidf = cosine_similarity(perfil_vec, vectorized).flatten()

    # Nível 0: Respeita SEMPRE o filtro de preço - não remove restrições
    mask = (df_produtos["genero"] == genero_pref) & (df_produtos["preco"] <= preco_max)

    indices = df_produtos[mask].index.tolist()
    candidatos = sorted([(i, scores_tfidf[i]) for i in indices], key=lambda x: x[1], reverse=True)[:20]

    # Se não houver nenhum candidato (ex: preço muito baixo), retorna vazio
    # O fallback cuidará de oferecer alternativas
    if not candidatos:
        return [], False

    top20 = [i for i, _ in candidatos]

    # encontra os 50 usuários com perfil mais similar e usa as notas previstas deles
    scores_perfil = scores_tfidf[top20]
    pesos_usuarios = pred_matrix[:, top20] @ scores_perfil
    top_usuarios = np.argsort(pesos_usuarios)[::-1][:50]
    svd_scores = pred_matrix[top_usuarios][:, top20].mean(axis=0)

    resultado = []
    for rank, idx in enumerate(top20):
        tfidf_s = scores_tfidf[idx]
        svd_s   = (svd_scores[rank] - 1) / 4
        final   = 0.7 * tfidf_s + 0.3 * svd_s
        resultado.append((idx, final, svd_scores[rank]))

    resultado.sort(key=lambda x: x[1], reverse=True)
    return resultado[:top_n], False


def recomendar_com_fallback(familias_pref, ocasiao_pref, genero_pref, preco_max, top_n=5):
    """
    Sistema de recomendação com fallback em cascata.
    
    Níveis de fallback:
    0. Nível 0: TF-IDF + SVD com todas as restrições (família + ocasião + gênero + preço)
    1. Nível 1: Remove restrição de ocasião (família + gênero + preço)
    2. Nível 2: Remove restrição de gênero (família + ocasião + preço)
    3. Nível 3: Retorna "Top-5 Mais Vendidos" mockados por gênero
    
    Retorna: (resultado, nivel_fallback, mensagem_fallback)
    """
    
    # Validar inputs
    familias_pref, ocasiao_pref, genero_pref, preco_max = validar_inputs(
        familias_pref, ocasiao_pref, genero_pref, preco_max
    )
    
    SCORE_MINIMO = 0.50
    nivel_fallback = 0
    mensagem = ""
    
    # ===== NÍVEL 0: Tentativa normal =====
    resultado, aviso_exp = recomendar_hibrido(familias_pref, ocasiao_pref, genero_pref, preco_max, top_n)
    
    if resultado and resultado[0][1] >= SCORE_MINIMO:
        # Validação adicional: verifica se há pelo menos 1 resultado com a primeira família solicitada
        # (a mais importante, já que é a primeira na lista de preferências do usuário)
        tem_familia_correta = False
        familia_principal = familias_pref[0].lower()
        
        for idx, _, _ in resultado:
            familia_produto = df_produtos.iloc[idx]["familia_olfativa"].lower()
            # Verifica se a família principal está no produto
            if familia_principal in familia_produto:
                tem_familia_correta = True
                break
        
        if tem_familia_correta:
            return resultado, nivel_fallback, "✅ Recomendação personalizada com excelente match!"
    
    # ===== NÍVEL 1: Remove restrição de ocasião =====
    nivel_fallback = 1
    mensagem = "⚠️ Poucas opções para essa ocasião. Mostrando recomendações de outras ocasiões."
    perfil = " ".join(familias_pref) + " " + genero_pref
    perfil_vec = vectorizer.transform([perfil])
    scores_tfidf = cosine_similarity(perfil_vec, vectorized).flatten()
    
    # Mantém filtro de preço neste nível
    mask = (df_produtos["genero"] == genero_pref) & (df_produtos["preco"] <= preco_max)
    if mask.sum() == 0:
        # Se não houver nem no preço original, passa para próximo nível
        pass
    else:
        indices = df_produtos[mask].index.tolist()
        candidatos = sorted([(i, scores_tfidf[i]) for i in indices], key=lambda x: x[1], reverse=True)[:20]
        
        if candidatos:
            top20 = [i for i, _ in candidatos]
            scores_perfil = scores_tfidf[top20]
            pesos_usuarios = pred_matrix[:, top20] @ scores_perfil
            top_usuarios = np.argsort(pesos_usuarios)[::-1][:50]
            svd_scores = pred_matrix[top_usuarios][:, top20].mean(axis=0)
            
            resultado = []
            for rank, idx in enumerate(top20):
                tfidf_s = scores_tfidf[idx]
                svd_s = (svd_scores[rank] - 1) / 4
                final = 0.7 * tfidf_s + 0.3 * svd_s
                resultado.append((idx, final, svd_scores[rank]))
            
            resultado.sort(key=lambda x: x[1], reverse=True)
            resultado = resultado[:top_n]
            
            if resultado and resultado[0][1] >= SCORE_MINIMO:
                return resultado, nivel_fallback, mensagem
    
    # ===== NÍVEL 2: Remove restrição de gênero =====
    nivel_fallback = 2
    mensagem = "⚠️ Poucos perfumes no seu gênero. Mostrando recomendações de outras categorias."
    
    perfil = " ".join(familias_pref) + " " + ocasiao_pref
    perfil_vec = vectorizer.transform([perfil])
    scores_tfidf = cosine_similarity(perfil_vec, vectorized).flatten()
    
    # Mantém filtro de preço neste nível (não remove gênero ainda)
    mask = df_produtos["preco"] <= preco_max
    if mask.sum() == 0:
        # Se não houver nem no preço original, passa para próximo nível
        pass
    else:
        indices = df_produtos[mask].index.tolist()
        candidatos = sorted([(i, scores_tfidf[i]) for i in indices], key=lambda x: x[1], reverse=True)[:20]
        
        if candidatos:
            top20 = [i for i, _ in candidatos]
            scores_perfil = scores_tfidf[top20]
            pesos_usuarios = pred_matrix[:, top20] @ scores_perfil
            top_usuarios = np.argsort(pesos_usuarios)[::-1][:50]
            svd_scores = pred_matrix[top_usuarios][:, top20].mean(axis=0)
            
            resultado = []
            for rank, idx in enumerate(top20):
                tfidf_s = scores_tfidf[idx]
                svd_s = (svd_scores[rank] - 1) / 4
                final = 0.7 * tfidf_s + 0.3 * svd_s
                resultado.append((idx, final, svd_scores[rank]))
            
            resultado.sort(key=lambda x: x[1], reverse=True)
            resultado = resultado[:top_n]
            
            if resultado and resultado[0][1] >= SCORE_MINIMO:
                return resultado, nivel_fallback, mensagem
    
    # ===== NÍVEL 3: Top-5 Mais Vendidos (Mockados) =====
    nivel_fallback = 3
    mensagem = "📌 Mostrando os 5 perfumes mais vendidos da comunidade!"
    resultado = obter_top5_mockados(genero_pref)[:top_n]
    
    return resultado, nivel_fallback, mensagem


def carregar_usuarios():
    if USUARIOS_JSON.exists():
        with open(USUARIOS_JSON, "r", encoding="utf-8") as f:
            conteudo = f.read().strip()
            if conteudo:
                return json.loads(conteudo)
    return {}


def salvar_usuarios(dados):
    with open(USUARIOS_JSON, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


def faixa_from_slider(valor):
    if valor <= 100:
        return "ate_100"
    elif valor <= 200:
        return "100_200"
    elif valor <= 300:
        return "200_300"
    return "acima_300"


def auth_output_payload(status_msg, msg_bem_vindo, auth_visible, perfil_visible,
                        email, nome, genero, familia, ocasiao, preco_max):
    return (
        status_msg,
        msg_bem_vindo,
        gr.update(visible=auth_visible),
        gr.update(visible=perfil_visible),
        gr.update(visible=perfil_visible), # vitrine_auth
        gr.update(visible=auth_visible),   # vitrine_anon
        gr.update(visible=perfil_visible), # feedback_auth
        gr.update(visible=auth_visible),   # feedback_anon
        email, nome, genero, familia, ocasiao, preco_max
    )


def cadastrar(email, senha, nome, genero, familia, ocasiao, preco_max):
    if not email or not senha or not nome:
        return auth_output_payload("⚠️ Preencha e-mail, senha e nome.", "", True, False, None, None, None, [], None, 200)
    
    if not familia or len(familia) == 0:
        return auth_output_payload("⚠️ Selecione pelo menos uma família olfativa.", "", True, False, None, None, None, [], None, 200)
    
    if not ocasiao:
        return auth_output_payload("⚠️ Selecione uma ocasião de uso.", "", True, False, None, None, None, [], None, 200)
    
    if not genero:
        return auth_output_payload("⚠️ Selecione um gênero de perfume.", "", True, False, None, None, None, [], None, 200)
    usuarios = carregar_usuarios()
    if email in usuarios:
        return auth_output_payload("⚠️ E-mail já cadastrado. Faça login.", "", True, False, None, None, None, [], None, 200)
        
    usuarios[email] = {
        "senha": senha,
        "nome": nome, "genero": genero, "familia": familia,
        "ocasiao": ocasiao, "preco_max": preco_max
    }
    salvar_usuarios(usuarios)
    
    info_html = f"""
    <div style="background:linear-gradient(145deg,#1c1829,#252137);border-radius:12px;padding:24px;border:1px solid rgba(201,169,110,0.3);margin:20px 0;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
        <h3 style="color:#c9a96e;margin-top:0;font-size:20px;">✅ Conta criada com sucesso, {nome}!</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px;">
            <div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <span style="color:#8a7f9e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">🧬 Gênero</span><br>
                <span style="color:#e8d5b7;font-weight:bold;">{genero}</span>
            </div>
            <div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <span style="color:#8a7f9e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">📍 Ocasião</span><br>
                <span style="color:#e8d5b7;font-weight:bold;">{ocasiao}</span>
            </div>
            <div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <span style="color:#8a7f9e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">💰 Preço Máx</span><br>
                <span style="color:#e8d5b7;font-weight:bold;">R$ {preco_max:.2f}</span>
            </div>
            <div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <span style="color:#8a7f9e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">🌸 Famílias Favoritas</span><br>
                <span style="color:#e8d5b7;font-weight:bold;">{', '.join(familia)}</span>
            </div>
        </div>
    </div>
    """
    return auth_output_payload("", info_html, False, True, email, nome, genero, familia, ocasiao, preco_max)


def login(email, senha):
    if not email or not senha:
        return auth_output_payload("⚠️ Digite e-mail e senha.", "", True, False, None, None, None, [], None, 200)
    usuarios = carregar_usuarios()
    if email not in usuarios:
        return auth_output_payload("❌ E-mail não encontrado. Cadastre-se primeiro.", "", True, False, None, None, None, [], None, 200)
    u = usuarios[email]
    if u.get("senha") != senha:
        return auth_output_payload("❌ Senha incorreta.", "", True, False, None, None, None, [], None, 200)

    familia = u["familia"] if isinstance(u["familia"], list) else [u["familia"]]

    info_html = f"""
    <div style="background:linear-gradient(145deg,#1c1829,#252137);border-radius:12px;padding:24px;border:1px solid rgba(201,169,110,0.3);margin:20px 0;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
        <h3 style="color:#c9a96e;margin-top:0;font-size:20px;">✅ Bem-vindo(a) de volta, {u['nome']}!</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px;">
            <div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <span style="color:#8a7f9e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">🧬 Gênero</span><br>
                <span style="color:#e8d5b7;font-weight:bold;">{u['genero']}</span>
            </div>
            <div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <span style="color:#8a7f9e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">📍 Ocasião</span><br>
                <span style="color:#e8d5b7;font-weight:bold;">{u['ocasiao']}</span>
            </div>
            <div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <span style="color:#8a7f9e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">💰 Preço Máx</span><br>
                <span style="color:#e8d5b7;font-weight:bold;">R$ {u['preco_max']:.2f}</span>
            </div>
            <div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <span style="color:#8a7f9e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">🌸 Famílias Favoritas</span><br>
                <span style="color:#e8d5b7;font-weight:bold;">{', '.join(familia)}</span>
            </div>
        </div>
    </div>
    """
    return auth_output_payload("", info_html, False, True, email, u["nome"], u["genero"], familia, u["ocasiao"], u["preco_max"])


def logout():
    return auth_output_payload("", "", True, False, None, None, None, [], None, 200)


def atualizar_perfil(email, genero, familia, ocasiao, preco_max):
    if not email:
        return "⚠️ Usuário não identificado.", gr.update(), genero, familia, ocasiao, preco_max
    
    if not familia or len(familia) == 0:
        return "⚠️ Selecione pelo menos uma família olfativa.", gr.update(), genero, familia, ocasiao, preco_max
    
    if not ocasiao:
        return "⚠️ Selecione uma ocasião de uso.", gr.update(), genero, familia, ocasiao, preco_max
    
    if not genero:
        return "⚠️ Selecione um gênero de perfume.", gr.update(), genero, familia, ocasiao, preco_max
    usuarios = carregar_usuarios()
    usuarios[email]["genero"]    = genero
    usuarios[email]["familia"]   = familia
    usuarios[email]["ocasiao"]   = ocasiao
    usuarios[email]["preco_max"] = preco_max
    salvar_usuarios(usuarios)

    nome = usuarios[email]["nome"]
    info_html = f"""
    <div style="background:linear-gradient(145deg,#1c1829,#252137);border-radius:12px;padding:24px;border:1px solid rgba(201,169,110,0.3);margin:20px 0;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
        <h3 style="color:#c9a96e;margin-top:0;font-size:20px;">✅ Bem-vindo(a), {nome}!</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px;">
            <div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <span style="color:#8a7f9e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">🧬 Gênero</span><br>
                <span style="color:#e8d5b7;font-weight:bold;">{genero}</span>
            </div>
            <div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <span style="color:#8a7f9e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">📍 Ocasião</span><br>
                <span style="color:#e8d5b7;font-weight:bold;">{ocasiao}</span>
            </div>
            <div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <span style="color:#8a7f9e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">💰 Preço Máx</span><br>
                <span style="color:#e8d5b7;font-weight:bold;">R$ {preco_max:.2f}</span>
            </div>
            <div style="background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <span style="color:#8a7f9e;font-size:11px;text-transform:uppercase;letter-spacing:1px;">🌸 Famílias Favoritas</span><br>
                <span style="color:#e8d5b7;font-weight:bold;">{', '.join(familia)}</span>
            </div>
        </div>
    </div>
    """
    return "✅ Perfil atualizado com sucesso!", info_html, genero, familia, ocasiao, preco_max


import base64

def gerar_vitrine(nome, genero, familias, ocasiao, preco_max):
    out_cols = [gr.update(visible=False)] * 5
    out_htmls = [gr.update(value="")] * 5
    out_checks = [gr.update(value=False)] * 5
    out_nomes = [""] * 5

    if not familias:
        return [gr.update(value="<p style='text-align:center;color:#888;padding:60px;'>Seu perfil não tem famílias olfativas cadastradas.</p>")] + out_cols + out_htmls + out_checks + out_nomes + [gr.update(visible=False)]

    tops, nivel_fallback, msg_fallback = recomendar_com_fallback(familias, ocasiao, genero, preco_max)

    for i in range(5):
        if i < len(tops):
            rank = i + 1
            idx, score, nota_svd = tops[i]
            r = df_produtos.iloc[idx]
            
            stars = "★" * int(round(nota_svd)) + "☆" * (5 - int(round(nota_svd)))
            notas = r["notas_olfativas"].split(", ")
            notas_html = "".join(f'<span style="background:rgba(201,169,110,0.15);color:#c9a96e;padding:2px 8px;border-radius:20px;font-size:11px;margin:2px;">{n}</span>' for n in notas[:4])

            img_prefix = r.get("imagem_path", "")
            img_html = '<div style="font-size:56px;opacity:0.9;">🧴</div>'
            lightbox_html = ""
            if pd.notna(img_prefix) and img_prefix:
                matches = list(BASE.glob(f"{img_prefix}.*"))
                if matches:
                    try:
                        with open(matches[0], "rb") as img_file:
                            encoded = base64.b64encode(img_file.read()).decode("utf-8")
                        ext = matches[0].suffix.lower().replace(".", "")
                        if ext == "jpg": ext = "jpeg"
                        data_src = f"data:image/{ext};base64,{encoded}"
                        img_html = f'''<img src="{data_src}" style="width:100%; height:100%; object-fit:cover; opacity:0.9; cursor:zoom-in;" onclick="document.getElementById('lbx_{i}').style.display='flex'" title="Clique para ampliar" />'''
                        lightbox_html = f'''<div id="lbx_{i}" onclick="this.style.display='none'" style="display:none;position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,0.92);align-items:center;justify-content:center;cursor:zoom-out;">
                            <img src="{data_src}" style="max-height:90vh;max-width:90vw;border-radius:12px;box-shadow:0 0 60px rgba(201,169,110,0.4);" />
                        </div>'''
                    except Exception:
                        pass

            card_html = f"""
            {lightbox_html}
            <div style="background:linear-gradient(145deg,#1c1829,#252137);border-radius:20px;overflow:hidden;
                        border:1px solid rgba(201,169,110,0.12);box-shadow:0 12px 40px rgba(0,0,0,0.4);
                        width:100%;flex-shrink:0;transition:transform .3s,box-shadow .3s;">
                <div style="height:200px;background:linear-gradient(135deg,#2a1f3d,#1a1225);display:flex;
                            align-items:center;justify-content:center;position:relative;">
                    {img_html}
                    <div style="position:absolute;top:12px;left:12px;background:linear-gradient(135deg,#c9a96e,#a8853a);
                                color:#fff;font-size:10px;font-weight:700;padding:4px 10px;border-radius:20px;
                                letter-spacing:1px;">TOP {rank}</div>
                    <div style="position:absolute;bottom:12px;right:12px;background:rgba(0,0,0,0.6);
                                color:#e8d5b7;font-size:12px;padding:4px 8px;border-radius:8px;">{stars}</div>
                </div>
                <div style="padding:16px;">
                    <div style="font-size:10px;color:#c9a96e;font-weight:700;letter-spacing:1.5px;
                                text-transform:uppercase;margin-bottom:4px;">{r['marca']}</div>
                    <div style="font-size:16px;font-weight:700;color:#e8d5b7;margin-bottom:6px;
                                line-height:1.3;">{r['nome']}</div>
                    <div style="font-size:11px;color:#8a7f9e;margin-bottom:8px;">{r['familia_olfativa']}</div>
                    <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:12px;">{notas_html}</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:20px;font-weight:800;color:#c9a96e;">R$ {r['preco']:.2f}</span>
                        <span style="font-size:10px;color:#6b6483;">Match {score*100:.0f}%</span>
                    </div>
                </div>
            </div>
            """
            out_cols[i] = gr.update(visible=True)
            out_htmls[i] = gr.update(value=card_html)
            out_checks[i] = gr.update(value=False)
            out_nomes[i] = str(r["nome"])

    # Mensagem informatica sobre qualidade das recomendações
    aviso_html = ""
    if nivel_fallback == 0:
        aviso_html = f"""
        <div style="background:linear-gradient(135deg,rgba(76,175,80,0.15),rgba(76,175,80,0.05));
                    border:2px solid #4caf50;border-radius:12px;padding:16px;margin-bottom:20px;">
            <div style="display:flex;gap:12px;align-items:flex-start;">
                <div style="font-size:24px;flex-shrink:0;">✨</div>
                <div>
                    <div style="color:#4caf50;font-size:14px;font-weight:700;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px;">
                        Recomendações Personalizadas
                    </div>
                    <p style="color:#8a7f9e;font-size:12px;line-height:1.5;margin:0;">
                        Estes perfumes foram selecionados especialmente para você com base em seu perfil e nas preferências de usuários similares. 💎
                    </p>
                </div>
            </div>
        </div>
        """
    elif nivel_fallback > 0:
        mensagens_por_nivel = {
            1: {
                "titulo": "Ocasião Restritiva",
                "icone": "⏰",
                "dica": "Tente escolher uma ocasião mais comum (ex: diurno casual) para mais opções.",
            },
            2: {
                "titulo": "Gênero Cruzado",
                "icone": "✨",
                "dica": "Seu perfil é raro. Mostrando perfumes similares de outros gêneros que combinam com sua preferência.",
            },
            3: {
                "titulo": "Sem Match com Critérios",
                "icone": "📌",
                "dica": "Nenhuma combinação ideal foi encontrada. Mostrando os perfumes mais populares da comunidade.",
            }
        }
        
        info = mensagens_por_nivel.get(nivel_fallback, {})
        icone = info.get("icone", "⚠️")
        titulo = info.get("titulo", "Aviso")
        dica = info.get("dica", msg_fallback)
        
        cor_mapa = {
            1: "#ffc107",  # amarelo
            2: "#ff6b6b",  # vermelho suave
            3: "#29b6f6"   # azul
        }
        cor = cor_mapa.get(nivel_fallback, "#c9a96e")
        
        aviso_html = f"""
        <div style="background:linear-gradient(135deg,rgba({int(cor[1:3], 16)},{int(cor[3:5], 16)},{int(cor[5:7], 16)},0.15),rgba({int(cor[1:3], 16)},{int(cor[3:5], 16)},{int(cor[5:7], 16)},0.05));
                    border:2px solid {cor};border-radius:12px;padding:16px;margin-bottom:20px;box-shadow:0 4px 12px rgba(0,0,0,0.2);">
            <div style="display:flex;gap:12px;align-items:flex-start;">
                <div style="font-size:24px;flex-shrink:0;line-height:1;">{icone}</div>
                <div style="flex:1;">
                    <div style="color:{cor};font-size:14px;font-weight:700;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px;">
                        ⚡ {titulo}
                    </div>
                    <p style="color:#8a7f9e;font-size:12px;line-height:1.5;margin:0 0 8px 0;">
                        {dica}
                    </p>
                    <div style="background:rgba({int(cor[1:3], 16)},{int(cor[3:5], 16)},{int(cor[5:7], 16)},0.15);padding:8px 12px;border-left:3px solid {cor};border-radius:4px;">
                        <span style="color:#8a7f9e;font-size:11px;line-height:1.4;">
                            <b>💡 Dica:</b> Recomendações abaixo do ideal detectadas. Considere ajustar suas preferências ou avaliar perfumes para melhorar futuras sugestões.
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """

    titulo = f"""
    <div style="text-align:center;margin-bottom:24px;">
        <h2 style="color:#e8d5b7;margin:0;font-size:24px;">✦ Recomendações para {nome}</h2>
        <p style="color:#8a7f9e;font-size:13px;margin-top:4px;">
            {', '.join(familias)} · {ocasiao} · até R${preco_max:.0f}</p>
    </div>
    {aviso_html}
    """
    visivel = len(tops) > 0
    btn_pedir_update = gr.update(visible=visivel)

    return [gr.update(value=titulo)] + out_cols + out_htmls + out_checks + out_nomes + [btn_pedir_update]


def confirmar_pedido(*args):
    checks = args[:5]
    nomes = args[5:]
    selecionados = [nomes[i] for i in range(5) if checks[i] and nomes[i]]
    if not selecionados:
        return "⚠️ Selecione ao menos um perfume marcando a caixa de seleção.", []
    return f"📦 **Pedido Confirmado!** Você pediu: {', '.join(selecionados)}.\n\nAcesse a aba **Feedback** para avaliar suas compras e deixar uma review textual.", selecionados


def preparar_feedback(pedidos):
    if not pedidos:
        return (
            "⚠️ Você ainda não fez nenhum pedido. Vá até **Minha Vitrine** e peça seus perfumes.",
            *[gr.update(visible=False)] * 5,
            *[gr.update()] * 5,
            *[gr.update(value="")] * 5,
            gr.update(visible=False)
        )
    
    row_updates = []
    slider_updates = []
    for i in range(5):
        if i < len(pedidos):
            row_updates.append(gr.update(visible=True))
            slider_updates.append(gr.update(label=f"Nota para: {pedidos[i]}", value=3))
        else:
            row_updates.append(gr.update(visible=False))
            slider_updates.append(gr.update())
            
    return (
        "📋 **Avalie os perfumes que você pediu:**", 
        *row_updates, 
        *slider_updates, 
        *[gr.update(value="")] * 5,
        gr.update(visible=True)
    )


def salvar_feedback(pedidos, s1, s2, s3, s4, s5, t1, t2, t3, t4, t5):
    if not pedidos:
        return "⚠️ Nenhum pedido para avaliar."
    notas  = [s1, s2, s3, s4, s5]
    textos = [t1, t2, t3, t4, t5]

    # persiste o feedback em feedback.csv para análise futura
    novo_arquivo = not FEEDBACK_CSV.exists()
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if novo_arquivo:
            writer.writerow(["perfume", "nota", "review"])
        for i in range(len(pedidos)):
            texto = textos[i].strip() if textos[i].strip() else ""
            writer.writerow([pedidos[i], int(notas[i]), texto])

    linhas = []
    for i in range(len(pedidos)):
        stars = '★' * int(notas[i]) + '☆' * (5 - int(notas[i]))
        texto = textos[i].strip() if textos[i].strip() else "Sem comentários."
        linhas.append(f"  • **{pedidos[i]}**: {stars}\n    *Review:* \"{texto}\"")
    return "✅ **Feedback registrado com sucesso!** Obrigado pela sua avaliação.\n\n" + "\n\n".join(linhas)


CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(160deg, #0f0c18 0%, #1a1528 40%, #12101e 100%) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #c9a96e, #8b6914) !important;
    border: none !important; color: #fff !important;
    font-weight: 600 !important; border-radius: 12px !important;
    padding: 12px 32px !important; font-size: 14px !important;
    transition: all 0.3s ease !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(201,169,110,0.3) !important;
}
footer { display: none !important; }
"""

HOME_MD = """
<div style="text-align:center; padding:40px 20px;">
    <div style="font-size:64px; margin-bottom:8px;">✦</div>
    <h1 style="font-size:42px; font-weight:900; margin:0;
               background:linear-gradient(135deg,#e8d5b7,#c9a96e,#e8d5b7);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               letter-spacing:2px;">O BOT-ICÁRIO</h1>
    <p style="color:#c9a96e; font-size:14px; letter-spacing:4px; margin-top:4px;
              text-transform:uppercase;">Haute Parfumerie Nationale</p>
    <div style="height:1px; background:linear-gradient(90deg,transparent,#c9a96e,transparent);
                margin:30px auto; max-width:400px;"></div>
</div>

<div style="max-width:800px; margin:0 auto; padding:0 20px;">
    <div style="text-align:center; margin-bottom:40px;">
        <p style="color:#a89880; font-size:18px; line-height:1.8; font-style:italic;">
            "Cada fragrância é uma cena — um filme que se desenrola na pele.<br>
            Notas de saída como um prólogo cinematográfico, o coração como o clímax,<br>
            e o fundo como os créditos que ninguém esquece."
        </p>
    </div>

    <div style="display:flex; gap:20px; flex-wrap:wrap; justify-content:center; margin-bottom:40px;">
        <div style="flex:1; min-width:220px; background:linear-gradient(145deg,#1e1a2e,#2a2340);
                    border-radius:20px; padding:28px; border:1px solid rgba(201,169,110,0.1);
                    text-align:center;">
            <div style="font-size:36px; margin-bottom:12px;">🌿</div>
            <h3 style="color:#e8d5b7; font-size:16px; margin:0 0 8px;">Curadoria Olfativa</h3>
            <p style="color:#8a7f9e; font-size:13px; line-height:1.6;">
                50 fragrâncias nacionais selecionadas. Cada produto possui 3 características chaves analisadas (Família Olfativa, Notas, Ocasião).
            </p>
        </div>
        <div style="flex:1; min-width:220px; background:linear-gradient(145deg,#1e1a2e,#2a2340);
                    border-radius:20px; padding:28px; border:1px solid rgba(201,169,110,0.1);
                    text-align:center;">
            <div style="font-size:36px; margin-bottom:12px;">🤖</div>
            <h3 style="color:#e8d5b7; font-size:16px; margin:0 0 8px;">IA Híbrida</h3>
            <p style="color:#8a7f9e; font-size:13px; line-height:1.6;">
                TF-IDF analisa o conteúdo textual. SVD aprende padrões complexos de 500 usuários.
            </p>
        </div>
        <div style="flex:1; min-width:220px; background:linear-gradient(145deg,#1e1a2e,#2a2340);
                    border-radius:20px; padding:28px; border:1px solid rgba(201,169,110,0.1);
                    text-align:center;">
            <div style="font-size:36px; margin-bottom:12px;">💎</div>
            <h3 style="color:#e8d5b7; font-size:16px; margin:0 0 8px;">Sistema de Avaliação</h3>
            <p style="color:#8a7f9e; font-size:13px; line-height:1.6;">
                Padrão de avaliação em 1 a 5 estrelas acompanhado de análise qualitativa (review em texto livre).
            </p>
        </div>
    </div>

    <div style="text-align:center; padding:30px; background:linear-gradient(135deg,#1e1a2e,#252137);
                border-radius:20px; border:1px solid rgba(201,169,110,0.08);">
        <p style="color:#c9a96e; font-size:13px; letter-spacing:2px; text-transform:uppercase; margin:0 0 8px;">
            Como funciona</p>
        <p style="color:#8a7f9e; font-size:14px; line-height:1.7; margin:0;">
            <b style="color:#e8d5b7;">1.</b> Acesse a aba <b style="color:#c9a96e;">Portal do Cliente</b> para fazer Login/Cadastro<br>
            <b style="color:#e8d5b7;">2.</b> Acesse <b style="color:#c9a96e;">Minha Vitrine</b> para pedir suas recomendações exclusivas<br>
            <b style="color:#e8d5b7;">3.</b> Avalie os perfumes na aba <b style="color:#c9a96e;">Feedback</b> atribuindo uma nota (1-5) e uma review textual.
        </p>
    </div>

    <div style="text-align:center; margin-top:30px;">
        <p style="color:#4a4560; font-size:11px;">
            Projeto 1 · Introdução a Inteligência Artificial · UnB 2026/1<br>
            TF-IDF + SVD · Gradio · Python
        </p>
    </div>
</div>
"""

with gr.Blocks(css=CUSTOM_CSS, title="O Bot-icário — Parfumerie IA") as demo:

    # Global states to pass around
    pedidos_state = gr.State([])
    email_st     = gr.State(None)
    nome_st      = gr.State(None)
    genero_st    = gr.State(None)
    familias_st  = gr.State([])
    ocasiao_st   = gr.State(None)
    preco_max_st = gr.State(200)

    with gr.Tabs():

        with gr.Tab("🏠 Home"):
            gr.HTML(HOME_MD)

        with gr.Tab("👤 Portal do Cliente"):
            gr.HTML("""
            <div style="text-align:center;padding:20px 0 10px;">
                <h2 style="color:#e8d5b7;margin:0;">👤 Portal do Cliente</h2>
                <div style="height:1px;background:linear-gradient(90deg,transparent,#c9a96e44,transparent);margin-top:16px;"></div>
            </div>
            """)
            
            with gr.Column(visible=True) as box_auth:
                with gr.Tabs():
                    with gr.Tab("🔑 Entrar"):
                        gr.Markdown("### Acesse sua conta")
                        login_email = gr.Textbox(label="📧 E-mail", placeholder="seu@email.com")
                        login_senha = gr.Textbox(label="🔒 Senha", type="password", placeholder="Sua senha")
                        btn_login = gr.Button("Entrar", variant="primary")
                        login_msg = gr.Markdown()
                        
                    with gr.Tab("📝 Criar Conta"):
                        gr.Markdown(r"### Cadastre seu perfil olfativo" + "\n" + r"*Campos marcados com * são obrigatórios")
                        cad_email = gr.Textbox(label="📧 E-mail *", placeholder="seu@email.com")
                        cad_senha = gr.Textbox(label="🔒 Senha *", type="password", placeholder="Sua senha")
                        cad_nome = gr.Textbox(label="👤 Nome *", placeholder="Seu nome")
                        cad_genero = gr.Radio(GENEROS, label="🧬 Gênero dos perfumes *", value="feminino")
                        cad_familias = gr.CheckboxGroup(FAMILIAS, label="🌸 Famílias olfativas preferidas *", value=["floral"])
                        cad_ocasiao = gr.Radio(OCASIOES, label="📍 Ocasião de uso *", value="diurno casual")
                        cad_preco_max = gr.Slider(50, 350, value=200, step=10, label="💰 Preço máximo (R$)")
                        btn_cadastrar = gr.Button("Cadastrar Perfil", variant="primary")
                        cad_msg = gr.Markdown()

            with gr.Column(visible=False) as box_perfil:
                msg_bem_vindo = gr.HTML()
                gr.HTML("<p style='text-align:center;color:#8a7f9e;font-size:14px;margin:20px 0;'>Navegue para a aba <b>🛍️ Minha Vitrine</b> para gerar suas recomendações!</p>")

                with gr.Accordion("✏️ Editar Preferências", open=False):
                    gr.Markdown("<p style='color:#c9a96e;font-size:12px;'><b>*</b> Campo obrigatório</p>")
                    edit_genero   = gr.Radio(GENEROS, label="🧬 Gênero dos perfumes *", value="feminino")
                    edit_familias = gr.CheckboxGroup(FAMILIAS, label="🌸 Famílias olfativas preferidas *", value=["floral"])
                    edit_ocasiao  = gr.Radio(OCASIOES, label="📍 Ocasião de uso *", value="diurno casual")
                    edit_preco    = gr.Slider(50, 350, value=200, step=10, label="💰 Preço máximo (R$)")
                    btn_salvar_perfil = gr.Button("💾 Salvar Preferências", variant="primary")
                    msg_editar_perfil = gr.Markdown("")

                btn_logout = gr.Button("🚪 Sair / Trocar de Perfil", variant="secondary")

        with gr.Tab("🛍️ Minha Vitrine"):
            gr.HTML("""
            <div style="text-align:center;padding:20px 0 10px;">
                <h2 style="color:#e8d5b7;margin:0;">🛍️ Minha Vitrine</h2>
                <div style="height:1px;background:linear-gradient(90deg,transparent,#c9a96e44,transparent);margin-top:16px;"></div>
            </div>
            """)
            
            with gr.Column(visible=True) as box_vitrine_anon:
                gr.Markdown("<p style='text-align:center;color:#888;padding:60px;'>Faça login no Portal do Cliente para acessar sua vitrine.</p>")
                
            with gr.Column(visible=False) as box_vitrine_auth:
                gr.Markdown("<p style='text-align:center;color:#8a7f9e;font-size:13px;'>Perfumes selecionados pela IA especialmente para o seu perfil</p>")
                btn_vitrine = gr.Button("✨ Gerar Recomendações", variant="primary", size="lg")
                vitrine_html = gr.HTML("")
                
                titulo_pedidos = gr.HTML("")
                
                cards_html = []
                cards_check = []
                cards_nome = []
                cards_col = []
                with gr.Row():
                    for i in range(5):
                        with gr.Column(visible=False, min_width=220) as col:
                            cards_col.append(col)
                            html_card = gr.HTML("")
                            check_card = gr.Checkbox(label="🛒 Adicionar ao pedido")
                            nome_st_card = gr.State("")
                            cards_html.append(html_card)
                            cards_check.append(check_card)
                            cards_nome.append(nome_st_card)

                btn_pedir = gr.Button("📦 Fazer Pedido", variant="primary", visible=False)
                msg_pedido = gr.Markdown("")

        with gr.Tab("⭐ Feedback"):
            gr.HTML("""
            <div style="text-align:center;padding:20px 0 10px;">
                <h2 style="color:#e8d5b7;margin:0;">⭐ Feedback</h2>
                <div style="height:1px;background:linear-gradient(90deg,transparent,#c9a96e44,transparent);margin-top:16px;"></div>
            </div>
            """)
            
            with gr.Column(visible=True) as box_feedback_anon:
                gr.Markdown("<p style='text-align:center;color:#888;padding:60px;'>Faça login no Portal do Cliente para avaliar seus perfumes.</p>")
                
            with gr.Column(visible=False) as box_feedback_auth:
                gr.Markdown("<p style='text-align:center;color:#8a7f9e;font-size:13px;'>Avalie os perfumes que você pediu atribuindo nota (1-5) e review textual.</p>")
                btn_carregar_fb = gr.Button("📋 Carregar meus pedidos", variant="primary")
                msg_fb = gr.Markdown("")

                rows = []
                sliders = []
                textboxes = []
                for i in range(5):
                    with gr.Row(visible=False) as r:
                        with gr.Column(scale=1):
                            s = gr.Slider(1, 5, value=3, step=1, label=f"Nota do Perfume {i+1}")
                        with gr.Column(scale=3):
                            t = gr.Textbox(label="Avaliação / Review", placeholder="Deixe seu comentário textual sobre este perfume...", lines=2)
                    rows.append(r)
                    sliders.append(s)
                    textboxes.append(t)

                btn_enviar_fb = gr.Button("📤 Enviar Avaliações e Reviews", variant="primary", visible=False)
                fb_resultado = gr.Markdown("")

    # Wiring logic
    auth_outputs = [
        login_msg, msg_bem_vindo,
        box_auth, box_perfil,
        box_vitrine_auth, box_vitrine_anon,
        box_feedback_auth, box_feedback_anon,
        email_st, nome_st, genero_st, familias_st, ocasiao_st, preco_max_st
    ]
    cad_outputs = [
        cad_msg, msg_bem_vindo,
        box_auth, box_perfil,
        box_vitrine_auth, box_vitrine_anon,
        box_feedback_auth, box_feedback_anon,
        email_st, nome_st, genero_st, familias_st, ocasiao_st, preco_max_st
    ]

    btn_login.click(login, [login_email, login_senha], auth_outputs)
    btn_cadastrar.click(cadastrar, [cad_email, cad_senha, cad_nome, cad_genero, cad_familias, cad_ocasiao, cad_preco_max], cad_outputs)
    btn_logout.click(logout, [], auth_outputs)

    btn_salvar_perfil.click(
        atualizar_perfil,
        [email_st, edit_genero, edit_familias, edit_ocasiao, edit_preco],
        [msg_editar_perfil, msg_bem_vindo, genero_st, familias_st, ocasiao_st, preco_max_st]
    )

    btn_vitrine.click(
        gerar_vitrine, 
        [nome_st, genero_st, familias_st, ocasiao_st, preco_max_st], 
        [titulo_pedidos] + cards_col + cards_html + cards_check + cards_nome + [btn_pedir]
    )

    btn_pedir.click(confirmar_pedido, cards_check + cards_nome, [msg_pedido, pedidos_state])

    btn_carregar_fb.click(
        preparar_feedback, 
        [pedidos_state], 
        [msg_fb] + rows + sliders + textboxes + [btn_enviar_fb]
    )
    
    btn_enviar_fb.click(
        salvar_feedback, 
        [pedidos_state] + sliders + textboxes, 
        fb_resultado
    )

if __name__ == "__main__":
    demo.launch(
        share=False, inbrowser=True,
        allowed_paths=[str(BASE)], ssr_mode=False,
    )
