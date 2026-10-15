# Assistente Inteligente de Chamados Universitários

Protótipo acadêmico de triagem de chamados universitários que combina **Random Forest**, **embeddings semânticos** e **RAG (Retrieval-Augmented Generation)**. Os dados utilizados são simulados.

## Objetivo

Receber a descrição de um chamado, estimar sua **categoria** e **prioridade**, recuperar os **três documentos mais relacionados** da base institucional e apresentar uma orientação sustentada pela documentação, quando houver evidência suficiente.

O protótipo oferece três modos de entrada:

- **Texto:** relato digitado.
- **Voz simulada:** texto de uma suposta transcrição; **não** grava ou transcreve áudio.
- **Imagem simulada:** descrição textual de uma imagem; **não** analisa imagens reais.

## Arquitetura

1. **Entrada** — relato em texto ou simulações textuais de voz e imagem.
2. **Classificação** — `classificador.py` usa TF-IDF + dois classificadores Random Forest para estimar categoria e prioridade.
3. **Busca semântica** — `busca_semantica.py` transforma os documentos e a consulta em embeddings com `paraphrase-multilingual-MiniLM-L12-v2`; utiliza similaridade do cosseno para selecionar o Top 3.
4. **Resposta documental** — `rag.py` filtra os documentos recuperados e apresenta orientações apoiadas em evidências. Emprega Ollama/Llama 3.2 em verificações de trechos; também contém uma regra específica para recuperação de senha do portal do aluno.
5. **Interface** — `app.py` exibe classificação, estimativas de confiança, documentos, orientação e histórico **durante a sessão**.

## Estrutura do projeto

```text
Assistente Inteligente de Chamados Universitarios/
├── app.py
├── classificador.py
├── busca_semantica.py
├── rag.py
├── dados/
│   └── chamados_historicos.csv
├── documentos/
│   └── base_conhecimento.csv
├── Relatorio_Tecnico_Assistente_Chamados_Universitarios.pdf
├── requirements.txt
└── README.md
```

As pastas `.venv/` e `__pycache__/` podem existir no computador de desenvolvimento, mas não devem ser incluídas na entrega.

## Tecnologias

- Python 3.11
- Streamlit — interface
- pandas e NumPy — dados e vetores
- scikit-learn — TF-IDF, Random Forest, divisão treino/teste e similaridade do cosseno
- sentence-transformers — modelo de embeddings multilíngue
- langchain-ollama + Ollama — acesso ao modelo local Llama 3.2 (`llama3.2:3b`)

## Pré-requisitos

- Python 3.11 instalado.
- Ollama instalado e em execução: https://ollama.com/download
- Internet na primeira execução para obter dependências e o modelo de embeddings.

## Instalação e execução

Abra um terminal **na pasta principal** do projeto.

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
python -m streamlit run app.py
```

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
python -m streamlit run app.py
```

> Instale e inicie o Ollama antes de rodar o aplicativo. Na primeira execução, o `sentence-transformers` poderá baixar o modelo `paraphrase-multilingual-MiniLM-L12-v2`.

Abra o endereço informado pelo Streamlit, normalmente **http://localhost:8501**.

Para testar cada módulo isoladamente:

```bash
python classificador.py
python busca_semantica.py
python rag.py
```

## Testes realizados durante o desenvolvimento

| Entrada / relato | Comportamento observado |
|---|---|
| Texto: recuperação de senha no portal do aluno | Classificação TI, prioridade Média e orientação do documento `DOC001 — Portal do aluno` |
| Voz simulada: falha no laboratório virtual | Classificação TI, prioridade Média e orientação do documento `Laboratório virtual` |
| Imagem simulada: cabo desconectado de computador | Classificação TI, prioridade Baixa e resposta de documentação insuficiente; revisão humana recomendada |
| Pergunta sobre senha particular do Wi-Fi do diretor | Resposta de documentação insuficiente, sem apresentar credencial |

Na divisão de teste dos **dados simulados**, foram observadas acurácias de aproximadamente **88,89%** para categoria e **86,11%** para prioridade. Esses números **não** validam o uso em situações reais. As probabilidades exibidas pelo Random Forest são estimativas do modelo, não garantias de acerto.

## Limitações e cuidados

- O conjunto de treinamento e a base de conhecimento são **fictícios**.
- A triagem de prioridades pode falhar: por exemplo, o problema do cabo desconectado foi classificado como prioridade Baixa, com baixa confiança.
- Uma similaridade semântica alta não prova que o documento contém a resposta. O sistema pode deixar de responder perguntas legítimas ou, em outros casos, selecionar evidência inadequada.
- O mecanismo de controle de respostas reduz alguns erros observados, mas **não elimina alucinações** nem substitui avaliação humana.
- O histórico é mantido apenas durante a sessão do Streamlit, não em banco de dados persistente.
- O protótipo não abre chamados em um sistema institucional, não autentica usuários e não realiza análise real de áudio/imagem.
- Não utilize dados pessoais, credenciais reais ou informações sigilosas nos arquivos de demonstração.

## Documentação

Consulte `Relatorio_Tecnico_Assistente_Chamados_Universitarios.pdf` para descrição mais ampla do pipeline, justificativas das tecnologias, riscos, alternativas de arquitetura e discussão dos testes.

**Projeto acadêmico para demonstração. Não destinado a operação institucional real sem revisão técnica, segurança e validação.**
