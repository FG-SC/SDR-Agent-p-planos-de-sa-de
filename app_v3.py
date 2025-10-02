import streamlit as st
import os
from dotenv import load_dotenv
import asyncio
import grpc

# --- Monkey Patch para gRPC e asyncio (necessário em alguns ambientes) ---
_original_secure_channel = grpc.aio.secure_channel
_original_insecure_channel = grpc.aio.insecure_channel

def setup_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

def patched_secure_channel(*args, **kwargs):
    setup_event_loop()
    return _original_secure_channel(*args, **kwargs)

def patched_insecure_channel(*args, **kwargs):
    setup_event_loop()
    return _original_insecure_channel(*args, **kwargs)

grpc.aio.secure_channel = patched_secure_channel
grpc.aio.insecure_channel = patched_insecure_channel

# --- Importações do LangChain e outras bibliotecas ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
# NOVAS IMPORTAÇÕES PARA O RETRIEVER INTELIGENTE
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# Carrega variáveis de ambiente
load_dotenv()

# --- Configurações Principais ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY]):
    st.error("Erro Crítico: Configure as variáveis de ambiente SUPABASE_URL, SUPABASE_KEY, e GOOGLE_API_KEY.")
    st.stop()

# --- Configuração da Página Streamlit ---
st.set_page_config(page_title="Agente SDR de Planos de Saúde v3", page_icon="🚀", layout="wide")
st.title("Agente SDR de Planos de Saúde")
st.markdown("Olá! Sou seu agente especializado em planos de saúde. Me diga o que você precisa e consultarei os documentos para encontrar a melhor opção para você.")

# --- Gerenciamento do Histórico de Conversa ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state.chat_history

@st.cache_resource
def initialize_rag_model():
    """
    Inicializa e cacheia todos os componentes do modelo RAG, agora com o SelfQueryRetriever.
    """
    st.info("Inicializando o agente inteligente... Por favor, aguarde.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )

    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

    # **MUDANÇA 1: Conectar à nova tabela e função de busca**
    vectorstore = SupabaseVectorStore(
        embedding=embeddings,
        client=supabase_client,
        table_name="planos_saude_docs_v2",  # <-- APONTANDO PARA A NOVA TABELA
        query_name="match_documents_v2"   # <-- USANDO A NOVA FUNÇÃO
    )

    # **MUDANÇA 2: Definir os metadados para a busca inteligente**
    # Descrevemos cada campo de metadados para que o LLM saiba como usá-los para filtrar.
    metadata_field_info = [
        AttributeInfo(
            name="plan_name",
            description="O nome específico de um plano de saúde, como 'Especial 100 Adesão', 'Clássico Adesão', etc.",
            type="string",
        ),
        AttributeInfo(
            name="age_range",
            description="A faixa etária para a qual um preço se aplica, como '24 a 28 anos' ou 'Até 18 anos'.",
            type="string",
        ),
        AttributeInfo(
            name="price",
            description="O valor mensal em Reais (R$) de um plano de saúde para uma determinada faixa etária.",
            type="string", # Usamos string pois o valor já vem formatado
        ),
        AttributeInfo(
            name="document_type",
            description="O tipo de informação contida no documento, sendo 'Tabela de Preços' ou 'Texto'.",
            type="string",
        ),
        AttributeInfo(
            name="source",
            description="O nome do arquivo PDF de onde a informação foi extraída.",
            type="string",
        ),
    ]
    document_content_description = "Informações sobre planos de saúde, incluindo preços, coberturas e rede credenciada."

    # **MUDANÇA 3: Criar o SelfQueryRetriever**
    # Este retriever usará o LLM para converter a pergunta do usuário em uma query estruturada com filtros.
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        document_content_description,
        metadata_field_info,
        verbose=True # Ajuda a debugar, mostrando os filtros que o LLM cria
    )

    # --- O restante da cadeia continua similar, mas agora usando o retriever inteligente ---

    contextualize_q_system_prompt = (
        "Dada uma conversa de chat e a última pergunta do usuário, "
        "formule uma pergunta independente que possa ser entendida sem o histórico do chat. "
        "NÃO responda à pergunta, apenas reformule-a se necessário, caso contrário, retorne-a como está."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """
    Você é um AI Agent especialista em planos de saúde. Sua principal função é analisar os documentos fornecidos e responder às perguntas do usuário com base **EXCLUSIVAMENTE** neles.

    **REGRAS E DIRETRIZES ESTRITAS:**
    1.  **FONTE ÚNICA DE VERDADE:** O `Contexto` abaixo, que contém trechos de documentos de planos de saúde, é sua **única** fonte de informação. Não utilize nenhum conhecimento prévio ou externo.
    2.  **OBRIGATORIEDADE DE USO DO CONTEXTO:** Você **DEVE** basear 100% da sua resposta nas informações encontradas no `Contexto`.
    3.  **INFORMAÇÃO NÃO ENCONTRADA:** Se a resposta para a pergunta do usuário não puder ser encontrada nos documentos do `Contexto`, você **DEVE** responder de forma clara e direta: "Com base nos documentos que consultei, não encontrei a informação sobre [tópico da pergunta]." **NÃO TENTE INVENTAR UMA RESPOSTA.**
    4.  **SEJA UM SDR:** Aja como um SDR (Sales Development Representative): seja prestativo, claro e guie o usuário. Se faltarem informações para uma recomendação (como cidade, idade, profissão), faça perguntas para coletar esses dados.
    5.  **APRESENTAÇÃO:** Organize as informações de forma clara, usando tabelas ou listas se os dados permitirem. Ao apresentar preços, sempre mencione o nome completo do plano e a faixa etária correspondente.

    **Contexto (Documentos Recuperados):**
    {context}

    **Pergunta do Usuário:**
    {input}

    **Sua Resposta (gerada exclusivamente a partir do Contexto acima):**
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    st.success("Agente inteligente inicializado com sucesso!")
    return rag_chain

# --- O restante do código de UI do Streamlit permanece o mesmo ---
rag_chain = initialize_rag_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("Fontes Consultadas"):
                for doc in message["context"]:
                    source_name = doc.metadata.get('source', 'Fonte Desconhecida').split('/')[-1].split('\\')[-1]
                    st.markdown(f"**Fonte:** `{source_name}`")
                    st.markdown(f"> _{doc.page_content}_")

if prompt := st.chat_input("Ex: Qual o preço do plano Executivo para 28 anos?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analisando sua pergunta e consultando os documentos..."):
            try:
                chat_history = get_session_history("main_chat")
                response_dict = rag_chain.invoke({"input": prompt, "chat_history": chat_history.messages})
                assistant_response = response_dict.get("answer", "Desculpe, não consegui processar sua pergunta.")
                
                st.markdown(assistant_response)

                retrieved_context = response_dict.get("context")
                if retrieved_context:
                    with st.expander("Fontes Consultadas para esta resposta"):
                        for doc in retrieved_context:
                            source_name = doc.metadata.get('source', 'Fonte Desconhecida').split('/')[-1].split('\\')[-1]
                            st.markdown(f"**Fonte:** `{source_name}`")
                            st.markdown(f"> _{doc.page_content}_")

                chat_history.add_user_message(prompt)
                chat_history.add_ai_message(assistant_response)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "context": retrieved_context
                })

            except Exception as e:
                error_message = f"Ocorreu um erro: {str(e)}"
                st.error(error_message)
                import traceback
                st.code(traceback.format_exc())
