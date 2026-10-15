
from pathlib import Path
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# CONFIGURACOES DO PROJETO
# ==========================================

PASTA_BASE = Path(__file__).resolve().parent

CAMINHO_DOCUMENTOS = (
    PASTA_BASE / "documentos" / "base_conhecimento.csv"
)

MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"


# ==========================================
# 1. CARREGAR DOCUMENTOS
# ==========================================

def carregar_documentos():
    dados = pd.read_csv(
        CAMINHO_DOCUMENTOS,
        encoding="utf-8-sig"
    )

    colunas = [
        "id",
        "titulo",
        "categoria",
        "conteudo"
    ]

    for coluna in colunas:
        if coluna not in dados.columns:
            raise ValueError(
                f"Coluna obrigatoria ausente: {coluna}"
            )

    dados = dados.dropna(
        subset=colunas
    ).reset_index(drop=True)

    return dados


# ==========================================
# 2. CARREGAR MODELO DE EMBEDDINGS
# ==========================================

def carregar_modelo():
    print("Carregando modelo de embeddings...")

    modelo = SentenceTransformer(
        MODELO_EMBEDDINGS
    )

    return modelo


# ==========================================
# 3. GERAR EMBEDDINGS DOS DOCUMENTOS
# ==========================================

def gerar_embeddings(documentos, modelo):

    textos = (
        documentos["titulo"].astype(str)
        + ". "
        + documentos["conteudo"].astype(str)
    ).tolist()

    vetores = modelo.encode(
        textos,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return vetores


# ==========================================
# 4. BUSCAR DOCUMENTOS SEMANTICAMENTE
# ==========================================

def buscar_top_k(
    pergunta,
    documentos,
    vetores,
    modelo,
    k=3
):
    if not pergunta.strip():
        return []

    if len(documentos) == 0:
        return []

    # Transformar pergunta em embedding
    vetor_pergunta = modelo.encode(
        [pergunta],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Comparar pergunta e documentos
    similaridades = cosine_similarity(
        vetor_pergunta,
        vetores
    )[0]

    # Selecionar os documentos mais proximos
    indices = np.argsort(
        similaridades
    )[::-1][:k]

    resultados = []

    for indice in indices:

        documento = documentos.iloc[indice]

        resultados.append({
            "id": str(documento["id"]),
            "titulo": str(documento["titulo"]),
            "categoria": str(documento["categoria"]),
            "conteudo": str(documento["conteudo"]),
            "similaridade": round(
                float(similaridades[indice]),
                4
            )
        })

    return resultados


# ==========================================
# 5. TESTE PELO TERMINAL
# ==========================================

if __name__ == "__main__":

    print("\nBUSCADOR SEMANTICO UNIVERSITARIO")
    print("--------------------------------")

    documentos = carregar_documentos()

    print(
        f"Documentos carregados: {len(documentos)}"
    )

    modelo = carregar_modelo()

    print("Gerando embeddings...")

    vetores = gerar_embeddings(
        documentos,
        modelo
    )

    print("Embeddings gerados com sucesso!")

    print(
        f"Dimensoes dos vetores: {vetores.shape[1]}"
    )

    while True:

        pergunta = input(
            "\nDigite sua pesquisa (ou 'sair'): "
        ).strip()

        if pergunta.lower() == "sair":
            print("Encerrando buscador...")
            break

        if not pergunta:
            print("Digite uma pergunta valida.")
            continue

        resultados = buscar_top_k(
            pergunta,
            documentos,
            vetores,
            modelo,
            k=3
        )

        print("\nTOP 3 DOCUMENTOS RELACIONADOS")
        print("--------------------------------")

        for posicao, resultado in enumerate(
            resultados,
            start=1
        ):

            print(
                f"\n{posicao}. {resultado['titulo']}"
            )

            print(
                f"ID: {resultado['id']}"
            )

            print(
                f"Categoria: {resultado['categoria']}"
            )

            print(
                f"Similaridade: "
                f"{resultado['similaridade']:.4f}"
            )

            print(
                f"Conteudo: {resultado['conteudo']}"
            )