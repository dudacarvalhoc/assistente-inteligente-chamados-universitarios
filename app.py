
import streamlit as st

from classificador import (
    treinar_modelos,
    classificar_chamado
)

from rag import (
    preparar_base,
    carregar_llm,
    processar_pergunta
)


# ==========================================
# CONFIGURACAO DA PAGINA
# ==========================================

st.set_page_config(
    page_title="Assistente de Chamados Universitarios",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# DESIGN DA INTERFACE
# ==========================================

st.markdown("""
<style>
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #F8FAFC !important;
    color: #1E293B !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #EAF2FF !important;
    color: #1E293B !important;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp p,
.stApp label,
.stApp li {
    color: #1E293B;
}

.stApp h1 {
    color: #2563EB !important;
    font-weight: 750;
}

.stApp h2,
.stApp h3 {
    color: #1D4ED8 !important;
}

.stButton button[kind="primary"] {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border-radius: 10px;
    border: none;
}

.stButton button[kind="primary"] p {
    color: #FFFFFF !important;
}

.stButton button:hover {
    border-color: #2563EB;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DBEAFE;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #DBEAFE;
    border-radius: 10px;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

textarea,
input {
    color: #1E293B !important;
    background-color: #FFFFFF !important;
}

hr {
    border-color: #DBEAFE !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# CARREGAMENTO DOS MODELOS
# ==========================================

@st.cache_resource(
    show_spinner="Treinando o Random Forest..."
)
def carregar_classificador():
    return treinar_modelos()


@st.cache_resource(
    show_spinner="Preparando embeddings e documentos..."
)
def carregar_sistema_rag():
    base = preparar_base()
    llm = carregar_llm()
    return base, llm


try:
    (
        vetorizador,
        modelo_categoria,
        modelo_prioridade,
        metricas
    ) = carregar_classificador()

    base, llm = carregar_sistema_rag()

except Exception as erro:
    st.error(
        "Não foi possível inicializar os modelos."
    )
    st.exception(erro)
    st.stop()


# ==========================================
# ESTADO DA APLICACAO
# ==========================================

if "historico" not in st.session_state:
    st.session_state.historico = []

if "ultimo_resultado" not in st.session_state:
    st.session_state.ultimo_resultado = None


# ==========================================
# BARRA LATERAL
# ==========================================

with st.sidebar:

    st.title("🎓 Central Inteligente")

    st.write(
        "Sistema acadêmico de gerenciamento "
        "de chamados universitários."
    )

    st.divider()

    st.subheader("Tecnologias utilizadas")

    st.write("🌳 Random Forest")
    st.write("🧠 Embeddings semânticos")
    st.write("🔎 Similaridade do cosseno")
    st.write("📚 RAG")
    st.write("🤖 Llama 3.2")
    st.write("🐍 Python + Streamlit")

    st.divider()

    st.subheader("Base de conhecimento")

    st.metric(
        "Documentos disponíveis",
        len(base["documentos"])
    )

    st.divider()

    st.subheader("Avaliação do modelo")

    st.metric(
        "Acurácia — Categoria",
        f"{metricas['acuracia_categoria']:.1%}"
    )

    st.metric(
        "Acurácia — Prioridade",
        f"{metricas['acuracia_prioridade']:.1%}"
    )

    st.caption(
        "Métricas obtidas com dados simulados. "
        "Não representam desempenho garantido "
        "em chamados reais."
    )

    st.divider()

    if st.button(
        "🗑️ Limpar histórico",
        use_container_width=True
    ):
        st.session_state.historico = []
        st.session_state.ultimo_resultado = None
        st.rerun()

    st.caption(
        "Projeto acadêmico — Inteligência Artificial "
        "e Aprendizagem de Máquina."
    )


# ==========================================
# CABECALHO
# ==========================================

st.title("🎓 Assistente Inteligente de Chamados")

st.markdown(
    "### Atendimento Universitário com Inteligência Artificial"
)

st.write(
    "Descreva um problema para que o sistema "
    "identifique a categoria, estime a prioridade "
    "e encontre orientações na base documental."
)

st.info(
    "Protótipo acadêmico que integra "
    "Random Forest, Embeddings e RAG."
)

st.divider()


# ==========================================
# TIPO DE ENTRADA
# ==========================================

st.subheader("1. Como deseja registrar o chamado?")

tipo_entrada = st.radio(
    "Selecione o tipo de entrada:",
    [
        "📝 Texto",
        "🎙️ Voz simulada",
        "📷 Imagem simulada"
    ],
    horizontal=True
)


# ==========================================
# RECEBER O CHAMADO
# ==========================================

if tipo_entrada == "📝 Texto":

    st.write(
        "Digite o problema relatado pelo aluno "
        "ou funcionário."
    )

    relato = st.text_area(
        "Descrição do chamado",
        placeholder=(
            "Ex.: O projetor da sala 204 não liga "
            "e minha aula começa em 10 minutos."
        ),
        height=130
    )

elif tipo_entrada == "🎙️ Voz simulada":

    st.write(
        "Nesta versão, simulamos a transcrição "
        "de um relato de voz."
    )

    st.caption(
        "Não ocorre gravação nem reconhecimento "
        "automático de áudio."
    )

    relato = st.text_area(
        "Transcrição do áudio",
        placeholder=(
            "Ex.: Não consigo acessar "
            "o laboratório virtual."
        ),
        height=130
    )

else:

    st.write(
        "Nesta versão, simulamos a informação "
        "extraída de uma imagem."
    )

    st.caption(
        "Descreva o que aparece na foto. "
        "Não ocorre análise automática de imagem."
    )

    relato = st.text_area(
        "Descrição da imagem",
        placeholder=(
            "Ex.: A foto mostra um cabo solto "
            "atrás do computador da sala 204."
        ),
        height=130
    )


# ==========================================
# INFORMACOES ADICIONAIS
# ==========================================

with st.expander(
    "Adicionar informações sobre o chamado"
):

    local = st.text_input(
        "Local",
        placeholder="Ex.: Laboratório 204"
    )

    solicitante = st.selectbox(
        "Tipo de solicitante",
        [
            "Aluno",
            "Professor",
            "Funcionário"
        ]
    )

    reincidencia = st.checkbox(
        "Esse problema já aconteceu anteriormente?"
    )


# ==========================================
# JUSTIFICATIVA DA CLASSIFICACAO
# ==========================================

def justificar_classificacao(
    categoria,
    prioridade,
    confianca_categoria,
    confianca_prioridade
):
    justificativa = (
        f"O modelo Random Forest identificou "
        f"a categoria '{categoria}' "
        f"e a prioridade '{prioridade}', "
        f"com base nos padrões aprendidos "
        f"durante o treinamento com chamados "
        f"históricos simulados."
    )

    if confianca_categoria < 60:
        justificativa += (
            " A classificação da categoria "
            "apresentou baixa confiança estimada "
            "e recomenda revisão humana."
        )

    if confianca_prioridade < 60:
        justificativa += (
            " A prioridade também apresentou "
            "baixa confiança estimada."
        )

    return justificativa


# ==========================================
# PROCESSAR CHAMADO
# ==========================================

st.divider()

processar = st.button(
    "🚀 Analisar chamado",
    type="primary",
    use_container_width=True
)

if processar:

    if not relato.strip():

        st.warning(
            "Descreva o problema antes de continuar."
        )

    else:

        with st.spinner(
            "Classificando o chamado "
            "e consultando documentos..."
        ):

            try:

                classificacao = classificar_chamado(
                    relato,
                    vetorizador,
                    modelo_categoria,
                    modelo_prioridade
                )

                resultado_rag = processar_pergunta(
                    relato,
                    base,
                    llm
                )

                justificativa = justificar_classificacao(
                    classificacao["categoria"],
                    classificacao["prioridade"],
                    classificacao["confianca_categoria"],
                    classificacao["confianca_prioridade"]
                )

                resultado = {
                    "relato": relato,
                    "entrada": tipo_entrada,
                    "local": local,
                    "solicitante": solicitante,
                    "reincidencia": reincidencia,
                    "classificacao": classificacao,
                    "justificativa": justificativa,
                    "rag": resultado_rag
                }

                st.session_state.ultimo_resultado = resultado

                st.session_state.historico.append(
                    resultado
                )

            except Exception as erro:

                st.error(
                    "Ocorreu um erro ao analisar o chamado."
                )

                st.exception(erro)


# ==========================================
# EXIBIR RESULTADO
# ==========================================

resultado = st.session_state.ultimo_resultado

if resultado:

    st.divider()

    st.subheader("2. Resultado da análise")

    classificacao = resultado["classificacao"]
    resposta_rag = resultado["rag"]

    coluna1, coluna2 = st.columns(2)

    with coluna1:
        st.metric(
            "Categoria do chamado",
            classificacao["categoria"]
        )

    with coluna2:
        st.metric(
            "Prioridade estimada",
            classificacao["prioridade"]
        )

    st.caption(
        f"Confiança estimada da categoria: "
        f"{classificacao['confianca_categoria']}% | "
        f"Confiança estimada da prioridade: "
        f"{classificacao['confianca_prioridade']}%"
    )

    st.subheader("3. Justificativa da classificação")

    st.write(
        resultado["justificativa"]
    )

    if (
        classificacao["confianca_categoria"] < 60
        or classificacao["confianca_prioridade"] < 60
    ):
        st.warning(
            "Classificação com baixa confiança "
            "estimada. Recomenda-se revisão humana."
        )

    st.divider()

    st.subheader("4. Orientação da IA")

    st.write(
        resposta_rag["resposta"]
    )

    if resposta_rag["fontes"]:

        with st.expander(
            "📖 Fontes que sustentam a orientação"
        ):

            for fonte in resposta_rag["fontes"]:

                st.write(
                    f"**{fonte['titulo']}** "
                    f"(ID: {fonte['id']})"
                )

                st.caption(
                    f"Similaridade: "
                    f"{fonte['similaridade']:.4f}"
                )

    st.divider()

    st.subheader("5. Top 3 documentos relacionados")

    for posicao, documento in enumerate(
        resposta_rag["top3"],
        start=1
    ):

        with st.expander(
            f"{posicao}. {documento['titulo']} "
            f"— Similaridade "
            f"{documento['similaridade']:.4f}"
        ):

            st.write(
                f"**Categoria:** "
                f"{documento['categoria']}"
            )

            st.write(
                f"**ID:** {documento['id']}"
            )

            st.write(
                documento["conteudo"]
            )

    st.caption(
        "Similaridade indica proximidade semântica, "
        "não garantia de que o documento responde "
        "ao chamado."
    )


# ==========================================
# HISTORICO DOS CHAMADOS
# ==========================================

st.divider()

st.subheader("6. Histórico de chamados analisados")

if not st.session_state.historico:

    st.write(
        "Nenhum chamado analisado nesta sessão."
    )

else:

    st.write(
        f"Total de chamados analisados: "
        f"{len(st.session_state.historico)}"
    )

    for indice, chamado in enumerate(
        reversed(st.session_state.historico),
        start=1
    ):

        with st.expander(
            f"Chamado {indice} — "
            f"{chamado['classificacao']['categoria']}"
        ):

            st.write(
                f"**Relato:** {chamado['relato']}"
            )

            st.write(
                f"**Entrada:** {chamado['entrada']}"
            )

            st.write(
                f"**Categoria:** "
                f"{chamado['classificacao']['categoria']}"
            )

            st.write(
                f"**Prioridade:** "
                f"{chamado['classificacao']['prioridade']}"
            )

            st.write(
                f"**Orientação:** "
                f"{chamado['rag']['resposta']}"
            )


# ==========================================
# RODAPE
# ==========================================

st.divider()

st.caption(
    "Assistente Inteligente de Chamados Universitários "
    "| Protótipo acadêmico com dados simulados. "
    "As classificações e orientações devem "
    "ser validadas antes de qualquer uso real."
)
