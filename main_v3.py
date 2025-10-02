import os
import glob
import logging
import re
import pymupdf
import pandas as pd
from dotenv import load_dotenv
from supabase.client import create_client, Client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Carregamento de Variáveis de Ambiente ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Validação das Variáveis ---
if not all([SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY]):
    logging.error("Variáveis de ambiente (SUPABASE_URL, SUPABASE_KEY, GOOGLE_API_KEY) não definidas.")
    exit()

# --- Constantes e Configurações ---
NEW_TABLE_NAME = "planos_saude_docs_v2"
NEW_QUERY_NAME = "match_documents_v2"
EMBEDDING_MODEL = "models/text-embedding-004"

# Padrões específicos observados nos PDFs
AGE_PATTERNS = [
    r'até\s+\d+\s+anos',
    r'de\s+\d+\s+a\s+\d+\s+anos',
    r'a\s+partir\s+de\s+\d+\s+anos',
    r'\d+\s+anos',
    r'acima\s+de\s+\d+'
]

PLAN_PATTERNS = [
    r'clássico',
    r'especial',
    r'executivo',
    r'direto',
    r'vital',
    r'mais'
]

REGION_PATTERNS = [
    r'capital',
    r'interior\s*\d*',
    r'nacional'
]

INVALID_PRICE_PATTERNS = [
    r'^(lab|ps|int|mat|ps/int|ps/mat|int/mat|ps/int/mat)$',
    r'^[a-z]+$',  # Apenas letras
    r'^-+$',      # Apenas traços
    r'nan',
    r'none',
    r'null'
]


def clean_text(text):
    """Função auxiliar para limpar o texto extraído."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('\n', ' ').replace('\r', '')
    return text


def is_valid_age_range(text):
    """Verifica se o texto representa uma faixa etária válida."""
    if not isinstance(text, str):
        return False
    
    text_clean = text.lower().strip()
    
    # Verifica se corresponde aos padrões de idade observados
    for pattern in AGE_PATTERNS:
        if re.search(pattern, text_clean):
            return True
    
    return False


def is_valid_price(text):
    """Verifica se o texto representa um preço válido baseado nos padrões observados."""
    if not isinstance(text, str):
        return False
    
    text_clean = text.lower().strip()
    
    # Rejeita padrões inválidos
    for pattern in INVALID_PRICE_PATTERNS:
        if re.search(pattern, text_clean):
            return False
    
    # Deve conter números
    if not re.search(r'\d', text_clean):
        return False
    
    # Verifica se parece com um valor monetário brasileiro
    # Remove caracteres não numéricos exceto vírgula e ponto
    numbers_only = re.sub(r'[^\d,.]', '', text_clean)
    
    if not numbers_only:
        return False
    
    # Padrões válidos: 123,45 ou 1.234,56 ou 123
    price_patterns = [
        r'^\d{1,3}(?:\.\d{3})*(?:,\d{2})?$',  # 1.234,56
        r'^\d+,\d{2}$',                        # 123,45
        r'^\d+$'                               # 123
    ]
    
    for pattern in price_patterns:
        if re.match(pattern, numbers_only):
            return True
    
    return False


def extract_plan_name(text):
    """Extrai e normaliza o nome do plano."""
    if not isinstance(text, str):
        return ""
    
    text_clean = clean_text(text)
    
    # Remove códigos ANS e outros padrões não relevantes
    text_clean = re.sub(r'\d{3}\.\d{3}/\d{2}-\d', '', text_clean)
    text_clean = re.sub(r'QC|QP|COP|RM|AHO|R1|RC', '', text_clean)
    text_clean = re.sub(r'Trad\.\s*\d+\s*F', '', text_clean)
    
    return clean_text(text_clean)


def normalize_price(price_text):
    """Normaliza o formato do preço."""
    if not isinstance(price_text, str):
        return None
    
    # Remove caracteres não numéricos exceto vírgula e ponto
    numbers_only = re.sub(r'[^\d,.]', '', price_text.strip())
    
    if not numbers_only:
        return None
    
    try:
        # Converte para formato float e depois para string brasileira
        if ',' in numbers_only and '.' in numbers_only:
            # Formato: 1.234,56
            numbers_only = numbers_only.replace('.', '').replace(',', '.')
        elif ',' in numbers_only:
            # Verifica se é decimal (123,45) ou milhares (1,234)
            parts = numbers_only.split(',')
            if len(parts) == 2 and len(parts[1]) == 2:
                # É decimal: 123,45
                numbers_only = numbers_only.replace(',', '.')
        
        price_float = float(numbers_only)
        
        # Verifica se está na faixa razoável para planos de saúde
        if 10 <= price_float <= 15000:
            return f"{price_float:.2f}".replace('.', ',')
        else:
            return None
            
    except ValueError:
        return None


def detect_table_type_and_region(page_text, table_position):
    """Detecta o tipo de tabela e região baseado no texto da página."""
    page_text_lower = page_text.lower()
    
    # Detecta região
    region = "Não especificado"
    for pattern in REGION_PATTERNS:
        matches = re.findall(pattern, page_text_lower)
        if matches:
            region = matches[0].title()
            break
    
    # Detecta tipo de dependentes
    dependents_type = "Não especificado"
    if 'titular + 2 ou mais dependentes' in page_text_lower:
        dependents_type = "Titular + 2 ou Mais Dependentes"
    elif 'titular + 1 dependente' in page_text_lower or 'titular ou titular + 1' in page_text_lower:
        dependents_type = "Titular ou Titular + 1 Dependente"
    
    return region, dependents_type


def parse_specific_price_table(df, filename, page_num, page_text):
    """Processa tabela de preços com base na estrutura específica observada nos PDFs."""
    docs = []
    
    if df.empty or df.shape[1] < 2:
        return docs
    
    logging.info(f"Processando tabela {df.shape} em {filename}, página {page_num}")
    
    # Detecta região e tipo de dependentes
    region, dependents_type = detect_table_type_and_region(page_text, None)
    
    # Limpa nomes das colunas
    df.columns = [clean_text(str(col)) for col in df.columns]
    
    # Identifica coluna de faixas etárias (primeira coluna)
    age_column = df.columns[0]
    
    # Identifica colunas de planos (excluindo a primeira)
    plan_columns = []
    for col in df.columns[1:]:
        col_clean = extract_plan_name(col)
        if col_clean and len(col_clean) > 2:
            plan_columns.append((col, col_clean))
    
    if not plan_columns:
        logging.warning("Nenhuma coluna de plano identificada")
        return docs
    
    # Processa cada linha da tabela
    valid_rows = 0
    for idx, row in df.iterrows():
        age_text = str(row[age_column]).strip()
        
        if not is_valid_age_range(age_text):
            continue
        
        age_range_clean = clean_text(age_text)
        
        # Processa cada plano
        for original_col, plan_name in plan_columns:
            price_raw = str(row[original_col]).strip()
            
            if not is_valid_price(price_raw):
                continue
            
            price_normalized = normalize_price(price_raw)
            if not price_normalized:
                continue
            
            # Cria conteúdo estruturado
            content = f"Plano: {plan_name} | Faixa etária: {age_range_clean} | Valor mensal: R$ {price_normalized}"
            
            if region != "Não especificado":
                content += f" | Região: {region}"
            
            if dependents_type != "Não especificado":
                content += f" | Categoria: {dependents_type}"
            
            # Metadados enriquecidos
            metadata = {
                "source": filename,
                "page": page_num,
                "document_type": "Tabela de Preços",
                "plan_name": plan_name,
                "age_range": age_range_clean,
                "price": price_normalized,
                "region": region,
                "dependents_type": dependents_type,
                "raw_price": price_raw
            }
            
            docs.append(Document(page_content=content, metadata=metadata))
            valid_rows += 1
    
    logging.info(f"Extraídos {valid_rows} registros válidos da tabela")
    return docs


def parse_price_tables(pdf_path):
    """Extrai tabelas de preços usando a estrutura específica observada."""
    docs = []
    try:
        pdf_document = pymupdf.open(pdf_path)
        filename = os.path.basename(pdf_path)

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            
            table_finder = page.find_tables()
            
            try:
                tables = list(table_finder)
                if not tables:
                    continue
                    
                logging.info(f"Encontradas {len(tables)} tabelas na página {page_num + 1} de '{filename}'")

                for i, table in enumerate(tables):
                    try:
                        df = table.to_pandas()
                        
                        table_docs = parse_specific_price_table(df, filename, page_num + 1, page_text)
                        if table_docs:
                            docs.extend(table_docs)
                            logging.info(f"✅ Tabela {i+1}: {len(table_docs)} registros extraídos")
                        else:
                            logging.info(f"⚠️ Tabela {i+1}: Nenhum registro válido")
                            
                    except Exception as e:
                        logging.error(f"❌ Erro ao processar tabela {i+1} na página {page_num + 1}: {e}")
                        continue
                        
            except Exception as e:
                logging.error(f"❌ Erro ao processar tabelas na página {page_num + 1}: {e}")
                continue

    except Exception as e:
        logging.error(f"❌ Erro ao abrir PDF {pdf_path}: {e}")
    
    return docs


def extract_text_content(pdf_path):
    """Extrai conteúdo textual relevante complementar."""
    docs = []
    try:
        pdf_document = pymupdf.open(pdf_path)
        filename = os.path.basename(pdf_path)

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            
            if not text.strip() or len(text.strip()) < 100:
                continue
            
            # Filtra apenas texto informativo (não tabelas)
            lines = text.split('\n')
            filtered_lines = []
            
            for line in lines:
                line_clean = line.strip()
                
                # Pula linhas que são claramente de tabelas
                if (not line_clean or 
                    is_valid_age_range(line_clean) or
                    re.search(r'^R\$\s*[\d.,]+', line_clean) or
                    len(line_clean.split()) < 3 or
                    re.search(r'^\d+[,.]?\d*$', line_clean)):
                    continue
                    
                filtered_lines.append(line_clean)
            
            if len(filtered_lines) < 5:
                continue
            
            text_content = ' '.join(filtered_lines)
            text_content = clean_text(text_content)
            
            if len(text_content) < 150:
                continue
            
            # Chunking inteligente
            chunk_size = 800
            overlap = 100
            
            if len(text_content) <= chunk_size:
                metadata = {
                    "source": filename,
                    "page": page_num + 1,
                    "document_type": "Informações Gerais"
                }
                docs.append(Document(page_content=text_content, metadata=metadata))
            else:
                # Divide em chunks com overlap
                for i in range(0, len(text_content), chunk_size - overlap):
                    chunk = text_content[i:i + chunk_size]
                    
                    # Tenta terminar em uma frase completa
                    if i + chunk_size < len(text_content):
                        last_period = chunk.rfind('.')
                        if last_period > chunk_size * 0.7:
                            chunk = chunk[:last_period + 1]
                    
                    if len(chunk.strip()) < 100:
                        continue
                    
                    metadata = {
                        "source": filename,
                        "page": page_num + 1,
                        "document_type": "Informações Gerais",
                        "chunk": i // (chunk_size - overlap) + 1
                    }
                    
                    docs.append(Document(page_content=chunk, metadata=metadata))

    except Exception as e:
        logging.error(f"❌ Erro ao extrair texto do PDF {pdf_path}: {e}")
    
    return docs


def process_all_pdfs(pdf_paths):
    """Processa todos os PDFs com relatório detalhado."""
    all_documents = []
    
    for pdf_path in pdf_paths:
        logging.info(f"\n{'='*80}")
        logging.info(f"📄 PROCESSANDO: {os.path.basename(pdf_path)}")
        logging.info(f"{'='*80}")
        
        # Processa tabelas de preços
        table_docs = parse_price_tables(pdf_path)
        if table_docs:
            all_documents.extend(table_docs)
            logging.info(f"✅ TABELAS: {len(table_docs)} registros de preços extraídos")
        else:
            logging.warning(f"⚠️ TABELAS: Nenhum registro de preço encontrado")
        
        # Processa texto informativo
        text_docs = extract_text_content(pdf_path)
        if text_docs:
            all_documents.extend(text_docs)
            logging.info(f"✅ TEXTO: {len(text_docs)} chunks informativos extraídos")
        else:
            logging.warning(f"⚠️ TEXTO: Nenhum conteúdo textual útil encontrado")

    return all_documents


def store_embeddings(documents, supabase_client: Client):
    """Armazena embeddings com relatório de progresso."""
    if not documents:
        logging.error("❌ Nenhum documento para processar.")
        return None

    try:
        logging.info(f"🔄 Inicializando embedding model: {EMBEDDING_MODEL}")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )

        logging.info(f"🗑️ Limpando tabela '{NEW_TABLE_NAME}'...")
        try:
            result = supabase_client.table(NEW_TABLE_NAME).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            logging.info(f"✅ Tabela limpa: {len(result.data)} registros removidos")
        except Exception as e:
            logging.warning(f"⚠️ Aviso ao limpar tabela: {e}")

        logging.info(f"📤 Enviando {len(documents)} documentos para Supabase...")
        
        vectorstore = SupabaseVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            client=supabase_client,
            table_name=NEW_TABLE_NAME,
            query_name=NEW_QUERY_NAME,
            chunk_size=50  # Lotes menores para evitar timeout
        )
        
        logging.info("✅ Documentos vetorizados e armazenados com sucesso!")
        return vectorstore

    except Exception as e:
        logging.error(f"❌ Falha ao armazenar embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_processing_report(all_docs):
    """Gera relatório detalhado do processamento."""
    logging.info(f"\n{'='*80}")
    logging.info(f"📊 RELATÓRIO FINAL DE PROCESSAMENTO")
    logging.info(f"{'='*80}")
    
    # Estatísticas gerais
    total_docs = len(all_docs)
    logging.info(f"📄 Total de documentos processados: {total_docs}")
    
    if total_docs == 0:
        logging.error("❌ Nenhum documento foi processado!")
        return
    
    # Estatísticas por tipo
    by_type = {}
    by_source = {}
    
    for doc in all_docs:
        doc_type = doc.metadata.get('document_type', 'Desconhecido')
        source = doc.metadata.get('source', 'Desconhecido')
        
        by_type[doc_type] = by_type.get(doc_type, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1
    
    logging.info(f"\n📋 Documentos por tipo:")
    for doc_type, count in sorted(by_type.items()):
        logging.info(f"  - {doc_type}: {count}")
    
    logging.info(f"\n📁 Documentos por arquivo:")
    for source, count in sorted(by_source.items()):
        logging.info(f"  - {source}: {count}")
    
    # Estatísticas específicas de preços
    price_docs = [doc for doc in all_docs if doc.metadata.get('document_type') == 'Tabela de Preços']
    if price_docs:
        logging.info(f"\n💰 Análise de preços ({len(price_docs)} registros):")
        
        plans = set()
        regions = set()
        for doc in price_docs:
            if doc.metadata.get('plan_name'):
                plans.add(doc.metadata['plan_name'])
            if doc.metadata.get('region'):
                regions.add(doc.metadata['region'])
        
        logging.info(f"  - Planos únicos encontrados: {len(plans)}")
        for plan in sorted(plans):
            logging.info(f"    • {plan}")
        
        logging.info(f"  - Regiões únicas encontradas: {len(regions)}")
        for region in sorted(regions):
            logging.info(f"    • {region}")


def main():
    """Função principal otimizada."""
    base_path = r"Bot_PH3A/Tabelas_de_vendas_SP_06_25"
    pdf_pattern = os.path.join(base_path, "*.pdf" #"QUALIPRO_SAS*.pdf"
                               )
    pdf_files = glob.glob(pdf_pattern)

    if not pdf_files:
        logging.error(f"❌ Nenhum PDF encontrado em: {pdf_pattern}")
        return

    logging.info(f"🚀 INICIANDO PROCESSAMENTO DE {len(pdf_files)} PDFs")
    
    all_docs = process_all_pdfs(pdf_files)
    
    # Gera relatório detalhado
    generate_processing_report(all_docs)

    if all_docs:
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        store_embeddings(all_docs, supabase_client)
        logging.info(f"\n🎉 PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    else:
        logging.error(f"\n❌ FALHA NO PROCESSAMENTO: Nenhum documento foi gerado!")


if __name__ == "__main__":
    main()