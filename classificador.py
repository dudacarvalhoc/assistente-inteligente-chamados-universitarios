
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# CONFIGURACOES
# ==========================================

PASTA_BASE = Path(__file__).resolve().parent

CAMINHO_DADOS = (
    PASTA_BASE / "dados" / "chamados_historicos.csv"
)


# ==========================================
# 1. CARREGAR OS DADOS
# ==========================================

def carregar_dados():
    dados = pd.read_csv(CAMINHO_DADOS)

    colunas = [
        "texto",
        "categoria",
        "prioridade"
    ]

    dados = dados.dropna(subset=colunas)

    return dados


# ==========================================
# 2. TREINAR RANDOM FOREST
# ==========================================

def treinar_modelos():
    dados = carregar_dados()

    X = dados["texto"].astype(str)

    y_categoria = dados["categoria"]
    y_prioridade = dados["prioridade"]

    indices_treino, indices_teste = train_test_split(
        range(len(dados)),
        test_size=0.25,
        random_state=42
    )

    textos_treino = X.iloc[indices_treino]
    textos_teste = X.iloc[indices_teste]

    # Transformar os textos em numeros
    vetorizador = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    X_treino = vetorizador.fit_transform(
        textos_treino
    )

    X_teste = vetorizador.transform(
        textos_teste
    )

    # Modelo para categoria
    modelo_categoria = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    modelo_categoria.fit(
        X_treino,
        y_categoria.iloc[indices_treino]
    )

    # Modelo para prioridade
    modelo_prioridade = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    modelo_prioridade.fit(
        X_treino,
        y_prioridade.iloc[indices_treino]
    )

    # Avaliacao
    previsoes_categoria = modelo_categoria.predict(
        X_teste
    )

    previsoes_prioridade = modelo_prioridade.predict(
        X_teste
    )

    acuracia_categoria = accuracy_score(
        y_categoria.iloc[indices_teste],
        previsoes_categoria
    )

    acuracia_prioridade = accuracy_score(
        y_prioridade.iloc[indices_teste],
        previsoes_prioridade
    )

    resultados = {
        "acuracia_categoria": acuracia_categoria,
        "acuracia_prioridade": acuracia_prioridade
    }

    return (
        vetorizador,
        modelo_categoria,
        modelo_prioridade,
        resultados
    )


# ==========================================
# 3. CLASSIFICAR NOVOS CHAMADOS
# ==========================================

def classificar_chamado(
    texto,
    vetorizador,
    modelo_categoria,
    modelo_prioridade
):
    vetor = vetorizador.transform([texto])

    categoria = modelo_categoria.predict(vetor)[0]

    prioridade = modelo_prioridade.predict(vetor)[0]

    confianca_categoria = max(
        modelo_categoria.predict_proba(vetor)[0]
    )

    confianca_prioridade = max(
        modelo_prioridade.predict_proba(vetor)[0]
    )

    return {
        "categoria": categoria,
        "prioridade": prioridade,
        "confianca_categoria": round(
            float(confianca_categoria) * 100, 1
        ),
        "confianca_prioridade": round(
            float(confianca_prioridade) * 100, 1
        )
    }


# ==========================================
# 4. TESTE PELO TERMINAL
# ==========================================

if __name__ == "__main__":

    print("\nASSISTENTE DE CHAMADOS UNIVERSITARIOS")
    print("Treinando Random Forest...")

    (
        vetorizador,
        modelo_categoria,
        modelo_prioridade,
        resultados
    ) = treinar_modelos()

    print("\nTreinamento concluido!")

    print(
        "Acuracia categoria:",
        f"{resultados['acuracia_categoria']:.2%}"
    )

    print(
        "Acuracia prioridade:",
        f"{resultados['acuracia_prioridade']:.2%}"
    )

    while True:

        texto = input(
            "\nDescreva o chamado (ou 'sair'): "
        ).strip()

        if texto.lower() == "sair":
            break

        if not texto:
            continue

        resultado = classificar_chamado(
            texto,
            vetorizador,
            modelo_categoria,
            modelo_prioridade
        )

        print("\nRESULTADO DA CLASSIFICACAO")

        print(
            "Categoria:",
            resultado["categoria"]
        )

        print(
            "Prioridade:",
            resultado["prioridade"]
        )

        print(
            "Confianca categoria:",
            f"{resultado['confianca_categoria']}%"
        )

        print(
            "Confianca prioridade:",
            f"{resultado['confianca_prioridade']}%"
        )
