import streamlit as st
import os
from dotenv import load_dotenv
import asyncio
import grpc
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import re # Adicionado para a nova lógica de filtro

# --- Monkey Patch para gRPC e asyncio (sem alterações) ---
_original_secure_channel = grpc.aio.secure_channel
_original_insecure_channel = grpc.aio.insecure_channel
def setup_event_loop():
    try: return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
def patched_secure_channel(*args, **kwargs):
    setup_event_loop(); return _original_secure_channel(*args, **kwargs)
def patched_insecure_channel(*args, **kwargs):
    setup_event_loop(); return _original_insecure_channel(*args, **kwargs)
grpc.aio.secure_channel = patched_secure_channel
grpc.aio.insecure_channel = patched_insecure_channel

# --- Importações do LangChain (sem alterações) ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client, Client
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Carrega variáveis de ambiente
load_dotenv()

# --- Configurações (sem alterações) ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not all([SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY, COHERE_API_KEY]):
    st.error("❌ Erro: Configure TODAS as variáveis de ambiente no arquivo .env (incluindo COHERE_API_KEY)"); st.stop()

# --- Configuração da Página e CSS (sem alterações) ---
st.set_page_config(page_title="Agente SDR Qualicorp/SulAmérica", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style> .main-header { padding: 1rem 0; background: linear-gradient(90deg, #1f4e79 0%, #2e5c87 100%); color: white; border-radius: 10px; text-align: center; margin-bottom: 2rem; } .metric-card { background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #1f4e79; box-shadow: 0 2px 4px rgba(0,0,0,0.1); } .chat-container { border: 1px solid #e0e0e0; border-radius: 10px; padding: 1rem; background: #fafafa; } .quick-action-btn { background: #1f4e79; color: white; border: none; padding: 0.5rem 1rem; border-radius: 5px; margin: 0.2rem; } </style>""", unsafe_allow_html=True)

# --- Header Principal e Gerenciamento do Histórico (sem alterações) ---
st.markdown("""<div class="main-header"><h1>🏥 Agente SDR Qualicorp/SulAmérica</h1><p>Seu assistente inteligente especializado em planos de saúde empresariais</p></div>""", unsafe_allow_html=True)
if "chat_history" not in st.session_state: st.session_state.chat_history = ChatMessageHistory()
if "messages" not in st.session_state: st.session_state.messages = []
if "session_stats" not in st.session_state: st.session_state.session_stats = {"queries_count": 0, "price_queries": 0, "plans_consulted": set(), "start_time": datetime.now()}
def get_session_history(session_id: str) -> BaseChatMessageHistory: return st.session_state.chat_history

@st.cache_resource
def initialize_rag_model():
    """Inicializa o modelo RAG com configurações otimizadas e RERANKING."""
    with st.spinner("🔄 Inicializando sistema inteligente com reranking..."):
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1, convert_system_message_to_human=True)
            supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
            vectorstore = SupabaseVectorStore(embedding=embeddings, client=supabase_client, table_name="planos_saude_docs_v2", query_name="match_documents_v2")

            base_retriever = vectorstore.as_retriever(search_kwargs={"k": 400})
            compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-multilingual-v3.0", top_n=8)
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

            contextualize_q_system_prompt = ("Dada uma conversa de chat e a última pergunta do usuário, formule uma pergunta independente que possa ser entendida sem o histórico do chat. NÃO responda à pergunta, apenas reformule-a se necessário.")
            contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
            history_aware_retriever = create_history_aware_retriever(llm, compression_retriever, contextualize_q_prompt)
            
            # ================================================================
            # ============= PROMPT HYPER-ESPECÍFICO APLICADO =================
            # ================================================================
            qa_system_prompt = """
            Você é um AI Agent SDR especialista da Qualicorp/SulAmérica, focado em vendas consultivas de planos de saúde empresariais.
            **LÓGICA DE SELEÇÃO DE DADOS (SUA TAREFA MAIS IMPORTANTE):**
            1.  **ACHAR E DESTACAR O MELHOR RESULTADO:** Sua primeira tarefa é analisar o `{context}` e encontrar a faixa de idade que contém a idade pedida pelo usuário. Esta é a **recomendação principal**.
            2.  **INCLUIR COMPARAÇÕES ÚTEIS:** Depois de apresentar a recomendação principal, verifique se no `{context}` existem outras faixas de idade **próximas e relevantes** (ex: a faixa imediatamente seguinte). Você pode apresentá-las de forma secundária, para que o cliente possa comparar os preços.
            3.  **EXCLUIR O LIXO IRRELEVANTE:** Ignore e não mencione faixas de idade muito distantes ou que claramente não fazem sentido para a consulta. Se o usuário pediu 30 anos, não mostre um plano de "59 anos".
            4.  **PLANO B (SE NÃO ACHAR NADA):** Apenas se nenhuma faixa de idade no contexto contiver a idade solicitada, informe ao usuário e mostre as opções mais próximas que encontrar.
            **SEUS OBJETIVOS COMO SDR:**
            1. **QUALIFICAR LEADS**: Identificar necessidades específicas do cliente
            2. **APRESENTAR SOLUÇÕES**: Recomendar planos adequados ao perfil
            3. **GERAR INTERESSE**: Destacar benefícios e vantagens competitivas
            4. **CONDUZIR À VENDA**: Orientar próximos passos no processo

            **ESTRUTURA DOS DADOS:**
            Os documentos contêm informações no formato:
            "Plano: [nome] | Faixa etária: [idade] | Valor mensal: R$ [preço] | Região: [local] | Categoria: [dependentes]"

            **REGRAS DE COMUNICAÇÃO:**
            ✅ Use APENAS informações do contexto fornecido
            ✅ Seja consultivo, não apenas informativo  
            ✅ Faça perguntas qualificadoras quando necessário
            ✅ Organize respostas em tabelas quando apropriado
            ✅ Destaque benefícios e diferenciais
            ✅ Sugira próximos passos quando relevante

            **PERGUNTAS QUALIFICADORAS TÍPICAS:**
            - Quantos funcionários tem a empresa?
            - Qual faixa etária predominante?
            - Já possuem plano de saúde? Qual operadora?
            - Orçamento médio por funcionário?
            - Região de atuação da empresa?

            **FORMATO DE APRESENTAÇÃO:**
            - 📊 Tabelas para comparação de preços
            - 💡 Destaque de benefícios únicos
            - 🎯 Recomendações personalizadas
            - 📞 Call-to-action quando apropriado

            **PLANOS DISPONÍVEIS:**
            - **Clássico**: Entrada, ideal para PMEs
            - **Especial**: Intermediário, bom custo-benefício  
            - **Executivo**: Premium, para executivos
            - **Direto**: Acesso direto, sem referenciamento

            **REGIÕES DE COBERTURA:**
            - Capital SP e Região Metropolitana
            - Interior (1, 2, 3) - diferentes sub-regiões

            **CONTEXTO DA CONSULTA:**
            {context}

            **CONSULTA DO CLIENTE:**
            {input}

            **SUA RESPOSTA CONSULTIVA:**
            """
            
            qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            return rag_chain

        except Exception as e:
            st.error(f"❌ Erro ao inicializar o sistema: {str(e)}")
            st.stop()

# --- Funções de Análise e Outras (sem alterações) ---
def create_price_comparison_chart(price_data):
    if not price_data: return None
    df = pd.DataFrame(price_data)
    df['Preço_Num'] = df['Preço'].str.replace('R$ ', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    fig = px.bar(df, x='Faixa Etária', y='Preço_Num', color='Plano', title='Comparação de Preços por Faixa Etária', labels={'Preço_Num': 'Valor (R$)'}, text='Preço')
    fig.update_layout(xaxis_title="Faixa Etária", yaxis_title="Valor Mensal (R$)", showlegend=True, height=400)
    return fig

def extract_price_data(documents):
    price_data = []
    for doc in documents:
        if doc.metadata.get('document_type') == 'Tabela de Preços':
            price_str = doc.metadata.get('price', '0')
            try: price_float = float(price_str.replace('.', '', 10).replace(',', '.'))
            except (ValueError, AttributeError): price_float = 0
            price_data.append({'Plano': doc.metadata.get('plan_name', 'N/A'), 'Faixa Etária': doc.metadata.get('age_range', 'N/A'), 'Preço': f"R$ {doc.metadata.get('price', 'N/A')}", 'Região': doc.metadata.get('region', 'N/A'), 'Categoria': doc.metadata.get('dependents_type', 'N/A'), 'Preço_Num': price_float})
    return price_data

def update_session_stats(query, context):
    st.session_state.session_stats["queries_count"] += 1
    if "R$" in query or "preço" in query.lower() or "valor" in query.lower(): st.session_state.session_stats["price_queries"] += 1
    for doc in context:
        if doc.metadata.get('plan_name'): st.session_state.session_stats["plans_consulted"].add(doc.metadata['plan_name'])

# --- Layout Principal ---
col_main, col_sidebar = st.columns([3, 1])
with col_main:
    rag_chain = initialize_rag_model()
    st.success("✅ Sistema SDR com Reranking inicializado e pronto para atendimento!")
    st.markdown("### 💬 Consultor Virtual Qualicorp/SulAmérica")
    chat_container = st.container()
    # Exibe histórico (sem alterações na lógica)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "context" in message and message["context"]:
                    # RE-RENDERIZA A TABELA FILTRADA PARA CADA MENSAGEM NO HISTÓRICO
                    price_data = extract_price_data(message["context"])
                    if price_data:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            df = pd.DataFrame(price_data); st.markdown("#### 📊 Opções Encontradas"); st.dataframe(df[['Plano', 'Faixa Etária', 'Preço', 'Região', 'Categoria']], use_container_width=True, hide_index=True)
                        with col2:
                            prices = [p['Preço_Num'] for p in price_data if p['Preço_Num'] > 0]
                            if prices: st.metric("Menor Preço", f"R$ {min(prices):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')); st.metric("Maior Preço", f"R$ {max(prices):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')); st.metric("Preço Médio", f"R$ {sum(prices)/len(prices):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
                    with st.expander("📚 Fontes Consultadas", expanded=False):
                        for doc in message["context"]: source = doc.metadata.get('source', 'Fonte Desconhecida').replace('.pdf', ''); st.markdown(f"**📄 {source}** | Página {doc.metadata.get('page', 'N/A')}"); st.markdown(f"> *{doc.page_content[:200]}...*"); st.markdown("---")

    # --- Lógica de Chat ---
    if prompt := st.chat_input("💭 Ex: Preciso de um plano para empresa com 50 funcionários..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analisando, recuperando e ranqueando os melhores documentos..."):
                try:
                    chat_history = get_session_history("main_chat")
                    response_dict = rag_chain.invoke({"input": prompt, "chat_history": chat_history.messages})
                    assistant_response = response_dict.get("answer", "❌ Não consegui processar sua pergunta.")
                    retrieved_context = response_dict.get("context", [])
                    update_session_stats(prompt, retrieved_context)
                    
                    st.markdown(assistant_response)

                    # ================================================================
                    # ============== LÓGICA DE FILTRO DA INTERFACE ===================
                    # ================================================================
                    
                    # 1. Extrai os nomes dos planos da tabela Markdown na resposta da IA
                    # A regex busca por linhas que começam com '|' e captura o primeiro grupo de texto.
                    mentioned_plans = [plan.strip() for plan in re.findall(r"\|\s*(.*?)\s*\|", assistant_response)]

                    # 2. Filtra o contexto original, mantendo apenas os documentos cujos planos foram mencionados
                    if mentioned_plans:
                        filtered_context = [doc for doc in retrieved_context if doc.metadata.get('plan_name') in mentioned_plans]
                    else:
                        # Se não achar tabela, usa o contexto original como fallback
                        filtered_context = retrieved_context

                    # 3. Salva e exibe a tabela da interface APENAS com o contexto filtrado
                    if filtered_context:
                        price_data = extract_price_data(filtered_context)
                        if price_data:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                df = pd.DataFrame(price_data)
                                st.markdown("#### 📊 Opções Encontradas")
                                st.dataframe(df[['Plano', 'Faixa Etária', 'Preço', 'Região', 'Categoria']], use_container_width=True, hide_index=True)
                            with col2:
                                prices = [p['Preço_Num'] for p in price_data if p['Preço_Num'] > 0]
                                if prices:
                                    st.metric("💰 Menor Valor", f"R$ {min(prices):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
                                    st.metric("💎 Maior Valor", f"R$ {max(prices):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
                                    st.metric("📊 Valor Médio", f"R$ {sum(prices)/len(prices):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
                                    st.markdown("---")
                                    if st.button("📞 Solicitar Proposta", type="primary", key=f"proposal_{datetime.now().timestamp()}"):
                                        st.success("✅ Em breve nosso consultor entrará em contato!")
                            if len(price_data) > 3:
                                chart = create_price_comparison_chart(price_data)
                                if chart: st.plotly_chart(chart, use_container_width=True)

                    # ================================================================
                    
                    chat_history.add_user_message(prompt)
                    chat_history.add_ai_message(assistant_response)
                    
                    # Salva o CONTEXTO FILTRADO no histórico para re-renderizar corretamente
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response,
                        "context": filtered_context # <-- MUITO IMPORTANTE: salvamos o contexto já filtrado
                    })

                except Exception as e:
                    st.error(f"❌ **Erro ao processar solicitação:** {str(e)}")
                    # Adiciona um traceback para debug, se necessário
                    import traceback
                    st.error(traceback.format_exc())


# --- Sidebar e Footer (sem alterações) ---
with col_sidebar:
    st.markdown("### 🎯 Ações Rápidas")
    quick_queries = ["💼 Planos para PME (10-50 funcionários)", "🏢 Planos corporativos (50+ funcionários)", "👥 Comparar Clássico vs Especial", "📍 Cobertura Interior SP", "💰 Faixas de preço por idade"]
    for query in quick_queries:
        if st.button(query, key=f"quick_{hash(query)}", use_container_width=True): st.session_state.pending_query = query.split(" ", 1)[1]; st.rerun()
    st.markdown("---")
    st.markdown("### 📊 Dashboard da Sessão")
    stats = st.session_state.session_stats; col1, col2 = st.columns(2)
    with col1: st.metric("Consultas", stats["queries_count"])
    with col2: st.metric("Preços", stats["price_queries"])
    if stats["plans_consulted"]:
        st.markdown("**Planos Consultados:**")
        for plan in list(stats["plans_consulted"])[:5]:
            st.markdown(f"• {plan}")
    session_duration = datetime.now() - stats["start_time"]; st.metric("Tempo de Sessão", f"{session_duration.seconds // 60} min")
    st.markdown("---")
    with st.expander("🔍 Filtros Avançados"):
        selected_plan = st.selectbox("Tipo de Plano:", ["Todos", "Clássico", "Especial", "Executivo", "Direto"])
        selected_region = st.selectbox("Região:", ["Todas", "Capital", "Interior 1", "Interior 2", "Interior 3"])
        age_range = st.slider("Faixa Etária:", 18, 65, (25, 45), 5)
        if st.button("🔍 Buscar com Filtros", type="primary"): filter_query = f"Planos {selected_plan} na região {selected_region} para idades entre {age_range[0]} e {age_range[1]} anos"; st.session_state.pending_query = filter_query; st.rerun()
    st.markdown("---")
    st.markdown("### ⚙️ Controles")
    if st.button("🗑️ Limpar Conversa", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = ChatMessageHistory()
        st.session_state.session_stats = {"queries_count": 0, "price_queries": 0, "plans_consulted": set(), "start_time": datetime.now()}
        st.success("✅ Conversa limpa!"); st.rerun()
    if st.button("📋 Exportar Histórico", use_container_width=True):
        export_data = {"sessao": {"inicio": stats["start_time"].isoformat(), "consultas": stats["queries_count"], "consultas_precos": stats["price_queries"], "planos_consultados": list(stats["plans_consulted"])}, "historico": [{"role": msg["role"], "content": msg["content"], "timestamp": datetime.now().isoformat()} for msg in st.session_state.messages]}
        st.download_button(label="⬇️ Baixar JSON", data=json.dumps(export_data, indent=2, ensure_ascii=False), file_name=f"sessao_sdr_{datetime.now().strftime('%Y%m%d_%H%M')}.json", mime="application/json")
    st.markdown("---")
    st.markdown("### ℹ️ Sistema"); st.markdown("""**Versão:** 4.5 Final Sync\n**Modelo:** Gemini-2.0-Flash\n**Base:** Qualicorp/SulAmérica\n**Atualização:** Jul/2025""")

if hasattr(st.session_state, 'pending_query'):
    pending_query = st.session_state.pending_query; delattr(st.session_state, 'pending_query'); st.session_state.messages.append({"role": "user", "content": pending_query}); st.rerun()

st.markdown("---"); st.markdown("<div style='text-align: center; color: #666;'>🏥 Agente SDR Qualicorp/SulAmérica | Desenvolvido com ❤️ para resultados excepcionais</div>", unsafe_allow_html=True)