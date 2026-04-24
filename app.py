import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

BASE = Path(__file__).parent
DADOS = BASE / "dados"
USUARIOS_JSON = BASE / "usuarios.json"

df_produtos = pd.read_csv(DADOS / "produtos.csv")
df_matriz = pd.read_csv(DADOS / "matriz_utilidade.csv", index_col="usuario_id")

df_produtos["corpus"] = (
    df_produtos["familia_olfativa"] + " " +
    df_produtos["notas_olfativas"] + " " +
    df_produtos["ocasiao"]
)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_produtos["corpus"])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

mat = df_matriz.values.astype(float)
mat[mat == 0] = np.nan
user_means = np.nanmean(mat, axis=1)
user_means = np.where(np.isnan(user_means), 3.0, user_means)
mat_filled = np.where(np.isnan(mat), user_means[:, np.newaxis], mat)
mat_centered = mat_filled - user_means[:, np.newaxis]
U, sigma, Vt = np.linalg.svd(mat_centered, full_matrices=False)
k = 15
pred_matrix = (U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]) + user_means[:, np.newaxis]
pred_matrix = np.clip(pred_matrix, 1.0, 5.0)

FAMILIAS = sorted(df_produtos["familia_olfativa"].unique().tolist())
OCASIOES = sorted(df_produtos["ocasiao"].unique().tolist())
GENEROS = ["feminino", "masculino", "unissex"]


def recomendar_hibrido(familias_pref, ocasiao_pref, genero_pref, faixa_preco, top_n=5):
    perfil = " ".join(familias_pref) + " " + ocasiao_pref
    perfil_vec = tfidf.transform([perfil])
    scores_tfidf = cosine_similarity(perfil_vec, tfidf_matrix).flatten()

    mask = df_produtos["genero"].isin([genero_pref, "unissex"])
    if faixa_preco == "ate_100":
        mask &= df_produtos["preco"] <= 100
    elif faixa_preco == "100_200":
        mask &= (df_produtos["preco"] > 100) & (df_produtos["preco"] <= 200)
    elif faixa_preco == "200_300":
        mask &= (df_produtos["preco"] > 200) & (df_produtos["preco"] <= 300)
    else:
        mask &= df_produtos["preco"] > 300

    if mask.sum() == 0:
        mask = df_produtos["genero"].isin([genero_pref, "unissex"])

    indices = df_produtos[mask].index.tolist()
    candidatos = sorted([(i, scores_tfidf[i]) for i in indices], key=lambda x: x[1], reverse=True)[:20]
    top20 = [i for i, _ in candidatos]
    svd_scores = pred_matrix[:, top20].mean(axis=0)

    resultado = []
    for rank, idx in enumerate(top20):
        tfidf_s = scores_tfidf[idx]
        svd_s = (svd_scores[rank] - 1) / 4
        final = 0.7 * tfidf_s + 0.3 * svd_s
        resultado.append((idx, final, svd_scores[rank]))

    resultado.sort(key=lambda x: x[1], reverse=True)
    return resultado[:top_n]


def carregar_usuarios():
    if USUARIOS_JSON.exists():
        with open(USUARIOS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
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
                        nome, genero, familia, ocasiao, preco_max):
    return (
        status_msg, 
        msg_bem_vindo, 
        gr.update(visible=auth_visible), 
        gr.update(visible=perfil_visible), 
        gr.update(visible=perfil_visible), # vitrine_auth
        gr.update(visible=auth_visible),   # vitrine_anon
        gr.update(visible=perfil_visible), # feedback_auth
        gr.update(visible=auth_visible),   # feedback_anon
        nome, genero, familia, ocasiao, preco_max
    )


def cadastrar(email, senha, nome, genero, familia, ocasiao, preco_max):
    if not email or not senha or not nome:
        return auth_output_payload("⚠️ Preencha e-mail, senha e nome.", "", True, False, None, None, [], None, 200)
    usuarios = carregar_usuarios()
    if email in usuarios:
        return auth_output_payload("⚠️ E-mail já cadastrado. Faça login.", "", True, False, None, None, [], None, 200)
        
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
    return auth_output_payload("", info_html, False, True, nome, genero, familia, ocasiao, preco_max)


def login(email, senha):
    if not email or not senha:
        return auth_output_payload("⚠️ Digite e-mail e senha.", "", True, False, None, None, [], None, 200)
    usuarios = carregar_usuarios()
    if email not in usuarios:
        return auth_output_payload("❌ E-mail não encontrado. Cadastre-se primeiro.", "", True, False, None, None, [], None, 200)
    u = usuarios[email]
    if u.get("senha") != senha:
        return auth_output_payload("❌ Senha incorreta.", "", True, False, None, None, [], None, 200)
    
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
    return auth_output_payload("", info_html, False, True, u["nome"], u["genero"], familia, u["ocasiao"], u["preco_max"])


def logout():
    return auth_output_payload("", "", True, False, None, None, [], None, 200)


import base64

def gerar_vitrine(nome, genero, familias, ocasiao, preco_max):
    out_cols = [gr.update(visible=False)] * 5
    out_htmls = [gr.update(value="")] * 5
    out_checks = [gr.update(value=False)] * 5
    out_nomes = [""] * 5

    if not familias:
        return [gr.update(value="<p style='text-align:center;color:#888;padding:60px;'>Seu perfil não tem famílias olfativas cadastradas.</p>")] + out_cols + out_htmls + out_checks + out_nomes + [gr.update(visible=False)]

    faixa = faixa_from_slider(preco_max)
    tops = recomendar_hibrido(familias, ocasiao, genero, faixa)

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

    titulo = f"""
    <div style="text-align:center;margin-bottom:24px;">
        <h2 style="color:#e8d5b7;margin:0;font-size:24px;">✦ Recomendações para {nome}</h2>
        <p style="color:#8a7f9e;font-size:13px;margin-top:4px;">
            {', '.join(familias)} · {ocasiao} · até R${preco_max:.0f}</p>
    </div>
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
    notas = [s1, s2, s3, s4, s5]
    textos = [t1, t2, t3, t4, t5]
    linhas = []
    for i in range(len(pedidos)):
        stars = '★' * int(notas[i]) + '☆' * (5 - int(notas[i]))
        texto = textos[i].strip() if textos[i].strip() else "Sem comentários."
        linhas.append(f"  • **{pedidos[i]}**: {stars}\n    *Review:* \"{texto}\"")
    return "✅ **Feedback e Review registrados com sucesso!** Obrigado pela sua avaliação.\n\n" + "\n\n".join(linhas)


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
    nome_st = gr.State(None)
    genero_st = gr.State(None)
    familias_st = gr.State([])
    ocasiao_st = gr.State(None)
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
                        gr.Markdown("### Cadastre seu perfil olfativo")
                        cad_email = gr.Textbox(label="📧 E-mail", placeholder="seu@email.com")
                        cad_senha = gr.Textbox(label="🔒 Senha", type="password", placeholder="Sua senha")
                        cad_nome = gr.Textbox(label="👤 Nome", placeholder="Seu nome")
                        cad_genero = gr.Radio(GENEROS, label="🧬 Gênero dos perfumes", value="feminino")
                        cad_familias = gr.CheckboxGroup(FAMILIAS, label="🌸 Famílias olfativas preferidas", value=["floral"])
                        cad_ocasiao = gr.Radio(OCASIOES, label="📍 Ocasião de uso", value="diurno-casual")
                        cad_preco_max = gr.Slider(50, 350, value=200, step=10, label="💰 Preço máximo (R$)")
                        btn_cadastrar = gr.Button("Cadastrar Perfil", variant="primary")
                        cad_msg = gr.Markdown()

            with gr.Column(visible=False) as box_perfil:
                msg_bem_vindo = gr.HTML()
                gr.HTML("<p style='text-align:center;color:#8a7f9e;font-size:14px;margin:20px 0;'>Navegue para a aba <b>🛍️ Minha Vitrine</b> para gerar suas recomendações!</p>")
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
        nome_st, genero_st, familias_st, ocasiao_st, preco_max_st
    ]
    cad_outputs = [
        cad_msg, msg_bem_vindo, 
        box_auth, box_perfil, 
        box_vitrine_auth, box_vitrine_anon,
        box_feedback_auth, box_feedback_anon,
        nome_st, genero_st, familias_st, ocasiao_st, preco_max_st
    ]

    btn_login.click(login, [login_email, login_senha], auth_outputs)
    btn_cadastrar.click(cadastrar, [cad_email, cad_senha, cad_nome, cad_genero, cad_familias, cad_ocasiao, cad_preco_max], cad_outputs)
    btn_logout.click(logout, [], auth_outputs)

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
