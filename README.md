# 🏥 [Agente SDR com IA para Venda de Planos de Saúde](https://sdr-agent-demo.streamlit.app/)

Este projeto implementa um Agente de IA para Vendas (SDR - *Sales Development Representative*), construído com uma arquitetura RAG (*Retrieval-Augmented Generation*), para qualificar e guiar potenciais clientes na escolha de planos de saúde empresariais através de um chatbot interativo.

O agente atua como um assistente virtual 24/7, automatizando a fase inicial do funil de vendas, fazendo perguntas-chave para entender o perfil do cliente (número de empregados, faixa etária, orçamento, dependentes) e, ao final, direcionando os leads qualificados para um vendedor humano finalizar a negociação.

## ✨ Principais Funcionalidades

  * **Chatbot Conversacional Inteligente:** Interface construída com Streamlit que permite uma interação para cotações e tira-dúvidas.
  * **Pipeline de Dados Automatizado:** Scripts para processar, limpar e extrair informações estruturadas (preços, planos, faixas etárias) de múltiplos catálogos de planos de saúde em formato PDF.
  * **Base de Conhecimento Vetorizada:** Utiliza o Supabase como banco de dados vetorial para armazenar e consultar as informações extraídas dos documentos.
  * **Qualificação de Leads:** O agente é instruído a fazer perguntas estratégicas para qualificar o interesse e a necessidade do cliente, simulando o trabalho de um SDR.
  * **Visualização de Dados Dinâmica:** Geração de tabelas comparativas e gráficos (com Plotly) em tempo real para ajudar o cliente a visualizar e comparar as opções de planos.
  * **Dashboard Interativo:** Painel com filtros, ações rápidas e estatísticas da sessão de atendimento para uma experiência de usuário rica e eficiente.

## 🏛️ Arquitetura e Fluxo de Funcionamento

O projeto é dividido em duas fases principais: **1. Processamento e Ingestão de Dados** e **2. Aplicação de Chat com RAG**.

### Fase 1: Pipeline de Ingestão de Dados (`main_v3.py`)

1.  **Coleta de Dados:** Reúne PDFs de catálogos e tabelas de preços de operadoras de saúde (ex: SulAmérica, Qualicorp).
2.  **Extração de Conteúdo:** Utiliza a biblioteca `PyMuPDF` para extrair texto bruto e, crucialmente, identificar e extrair tabelas de preços de dentro dos PDFs.
3.  **Limpeza e Estruturação:** Funções customizadas com RegEx são aplicadas para limpar, validar e normalizar os dados extraídos, como faixas etárias, nomes de planos e valores.
4.  **Chunking:** O conteúdo textual e os registros das tabelas são divididos em "chunks" (fragmentos) de informação coesos e otimizados para a busca semântica.
5.  **Geração de Embeddings:** Usando o modelo `models/text-embedding-004` da Google, cada chunk é convertido em um vetor numérico (embedding).
6.  **Armazenamento:** Os vetores e seus metadados associados (fonte do documento, número da página, nome do plano, etc.) são enviados e armazenados no Supabase.

### Fase 2: Agente Conversacional RAG (`app_v3.py`)

1.  **Interface do Usuário (UI):** Um chatbot completo desenvolvido com Streamlit serve como front-end para a interação com o usuário.
2.  **Orquestração com LangChain:** O framework LangChain é utilizado para gerenciar todo o fluxo da conversa e a lógica do RAG.
3.  **Retriever Consciente de Histórico:** A pergunta do usuário é primeiro reformulada com base no histórico da conversa para criar uma consulta mais precisa e contextual.
4.  **Busca Vetorial (Retrieval):** O sistema busca no Supabase os chunks de informação mais relevantes semanticamente para a pergunta do usuário.
5.  **Aumento de Contexto (Augmentation):** Os documentos recuperados são injetados em um prompt robusto, que instrui o LLM a agir como um SDR especialista.
6.  **Geração da Resposta (Generation):** O modelo **Gemini 1.5 Flash** recebe o prompt aumentado e gera uma resposta consultiva, utilizando apenas as informações fornecidas para garantir a precisão e evitar alucinações.
7.  **Hand-off:** Ao final de uma qualificação bem-sucedida, o agente direciona o usuário a um vendedor humano.

## 🚀 Stack Tecnológica

  * **Linguagem:** Python 3.10+
  * **Frameworks de IA e Orquestração:**
      * LangChain: Para construir e gerenciar a cadeia RAG.
      * Google Generative AI: Para acesso aos modelos Gemini e de embedding.
  * **Modelos de IA:**
      * **LLM:** `gemini-1.5-flash`
      * **Embedding:** `models/text-embedding-004`
  * **Interface Web e Visualização:**
      * Streamlit: Para a criação da interface do chatbot e dashboard.
      * Plotly: Para a geração de gráficos interativos.
  * **Processamento de Dados:**
      * Pandas: Para manipulação de dados tabulares.
      * PyMuPDF: Para extração de texto e tabelas de PDFs.
  * **Banco de Dados Vetorial:**
      * Supabase: Para armazenamento e busca de vetores.

## ⚙️ Instalação e Configuração

**Pré-requisitos:**

  * Python 3.10 ou superior
  * Conta no Supabase (com URL e Chave de API)
  * Chave de API da Google AI

**Passos:**

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2.  **Crie e ative um ambiente virtual:**

    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure as variáveis de ambiente:**
    Crie um arquivo chamado `.env` na raiz do projeto e adicione suas chaves:

    ```env
    SUPABASE_URL="https://SEU_PROJETO.supabase.co"
    SUPABASE_KEY="SUA_CHAVE_API_SUPABASE"
    GOOGLE_API_KEY="SUA_CHAVE_API_GOOGLE"
    ```

## ▶️ Como Executar

### 1\. Processar os PDFs e Popular o Banco de Dados

Primeiro, você precisa executar o pipeline de ingestão para processar seus documentos e enviá-los ao Supabase.

1.  Coloque todos os arquivos PDF que deseja processar em uma pasta (ex: `documentos/`).
2.  Ajuste o caminho da pasta no arquivo `main_v3.py` (na função `main`).
3.  Execute o script:
    ```bash
    python main_v3.py
    ```
    Aguarde o término do processo. O terminal exibirá um relatório detalhado da extração.

### 2\. Iniciar a Aplicação do Chatbot

Com o banco de dados populado, inicie a interface do Streamlit.

1.  Execute o seguinte comando no terminal:

    ```bash
    streamlit run t.py
    ```

2.  Abra seu navegador e acesse o endereço fornecido (geralmente `http://localhost:8501`).

## 🔮 Próximos Passos e Melhorias

  * **Integração com CRM:** Conectar o agente a um CRM (como Salesforce ou HubSpot) para registrar automaticamente os leads qualificados.
  * **Análise de Sentimentos:** Implementar análise de sentimentos para medir a satisfação do cliente durante a conversa.
  * **Expandir Base de Conhecimento:** Adicionar mais documentos, como informações sobre rede credenciada, carências e políticas de reembolso.
  * **Deploy:** Publicar a aplicação em serviços como Streamlit Community Cloud, Google Cloud Run ou AWS.
