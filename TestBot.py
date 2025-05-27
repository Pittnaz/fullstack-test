import os
import logging
import google.generativeai as genai
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction
from dotenv import load_dotenv
import psycopg
import uuid
import datetime
import asyncio
import pypdf
import docx
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import numpy as np
import pdfplumber
from collections import defaultdict
from typing import List, Dict, Any, Optional

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --- Load Environment Variables ---
load_dotenv()


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.critical("TELEGRAM_TOKEN (Telegram Bot Token) not set. Bot cannot start.")
    exit(1)
else:
    logger.info("Telegram Bot Token loaded.")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set. Gemini functionality might be limited.")

# --- Database Connection Parameters ---
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_port_str = os.getenv("DB_PORT")

db_port = 5432
if db_port_str:
    try:
        db_port = int(db_port_str)
    except ValueError:
        logger.critical(f"Invalid value for DB_PORT: '{db_port_str}'. Using default 5432.")
        db_port = 5432

if not all([db_host, db_name, db_user, db_password]):
    logger.warning("Database connection parameters are not fully set. DB functionality might be limited.")

# --- AI Model Configuration ---
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        available_models = [m.name for m in genai.list_models()]

        found_model_name = None
        preferred_models = [
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro-latest",
            "models/gemini-1.5-pro",
            "models/gemini-pro",
            "models/chat-bison-001"
        ]

        for preferred in preferred_models:
            if preferred in available_models:
                try:
                    if "generateContent" in genai.get_model(preferred).supported_generation_methods:
                        found_model_name = preferred
                        logger.info(f"Found and using preferred Gemini model for RAG response: {found_model_name}")
                        break
                    else:
                        logger.debug(f"Preferred model {preferred} does not support generateContent.")
                except Exception as model_e:
                    logger.warning(f"Could not check methods for model {preferred}: {model_e}")

        if not found_model_name:
            logger.warning(
                "No suitable preferred generative Gemini model found. Searching for any available generative model.")
            for model_name in available_models:
                try:
                    if "generateContent" in genai.get_model(model_name).supported_generation_methods:
                        found_model_name = model_name
                        logger.info(f"Falling back to using available generative model: {found_model_name}")
                        break
                except Exception as model_e:
                    logger.warning(f"Could not check methods for model {model_name}: {model_e}")

        if found_model_name:
            gemini_model = genai.GenerativeModel(found_model_name)
            logger.info(f"Initialized generative Gemini model: {found_model_name}")
        else:
            logger.critical(
                "No suitable generative Gemini model found that supports generateContent. Chat and RAG response generation will not work.")
            gemini_model = None

    except Exception as e:
        logger.error(f"Failed to configure Gemini API or generative model: {e}", exc_info=True)
        gemini_model = None
        logger.critical("Gemini generative model not available. Chat and RAG response generation will not work.")

else:
    logger.warning("GEMINI_API_KEY not set. Gemini generative model will not be initialized.")

# Embedding model for getting vector embeddings
embedding_model = None
embedding_dimension = 384
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info(f"SentenceTransformer embedding model loaded: all-MiniLM-L6-v2 ({embedding_dimension} dimensions)")
except Exception as e:
    logger.critical(f"Failed to load embedding model: {e}", exc_info=True)
    embedding_model = None
    logger.critical("Embedding model not available. Document processing/RAG functionality will not work.")



def is_follow_up_question(query: str) -> bool:
    """
    Determines if the query is a follow-up question.
    """
    follow_up_indicators = [
        "and", "also", "more", "when", "where", "how", "why",
        "what", "which", "is there", "was there", "can you",
        "tell me more", "what else", "anything else"
    ]

    query_lower = query.lower()
    return any(indicator in query_lower for indicator in follow_up_indicators)

# --- Database Connection Function ---
def get_db_connection():
    """Establishes and returns a new database connection using psycopg (v3)."""
    conn = None
    if not all([db_host, db_name, db_user, db_password]):
        logger.error("Essential DB connection parameters are missing. Cannot establish connection.")
        return None

    try:
        conn = psycopg.connect(
            host=db_host,
            dbname=db_name,
            user=db_user,
            password=db_password,
            port=db_port,
            options="-c search_path=public"
        )
        logger.debug("Database connection successful (using psycopg v3).")
        return conn
    except psycopg.OperationalError as e:
        logger.critical(f"Database connection failed (psycopg v3): {e}", exc_info=True)
        return None
    except Exception as e:
        logger.critical(f"An unexpected error occurred during database connection (psycopg v3): {e}", exc_info=True)
        return None

# --- Database Initialization ---
def initialize_db():
    """Initializes the database table and index for document chunks."""
    logger.info("Initializing database (creating table and index)...")
    conn = get_db_connection()
    if conn is None:
        logger.critical("Database connection failed during initialization. DB setup aborted.")
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    document_id UUID NOT NULL,
                    s3_key VARCHAR(255),
                    chunk_index INTEGER NOT NULL,
                    content TEXT,
                    vector_embedding VECTOR(384),
                    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("Ensured document_chunks table exists.")

            index_name = "document_chunks_vector_embedding_idx"
            try:
                cur.execute(f"""
                    SELECT EXISTS (
                        SELECT 1
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE c.relname = %s
                        AND n.nspname = current_schema()
                    );
                """, (index_name,))
                index_exists = cur.fetchone()[0]
            except Exception as check_e:
                logger.warning(
                    f"Failed to check if index '{index_name}' exists: {check_e}. Assuming it doesn't exist for safety.",
                    exc_info=True)
                index_exists = False

            if not index_exists:
                logger.info(f"Index '{index_name}' does not exist. Attempting to create index...")
                index_created_successfully = False

                try:
                    cur.execute(
                        f"CREATE INDEX {index_name} ON document_chunks USING hnsw (vector_embedding vector_cosine_ops);")
                    logger.info(f"Index '{index_name}' created using vector_cosine_ops.")
                    index_created_successfully = True
                except Exception as e_cosine:
                    logger.warning(f"Failed to create index using vector_cosine_ops: {e_cosine}")
                    logger.info(f"Attempting to create index using vector_l2_ops instead...")

                    try:
                        cur.execute(
                            f"CREATE INDEX {index_name} ON document_chunks USING hnsw (vector_embedding vector_l2_ops);")
                        logger.info(f"Index '{index_name}' created using vector_l2_ops.")
                        index_created_successfully = True
                    except Exception as e_l2:
                        logger.error(f"Failed to create index using vector_l2_ops: {e_l2}", exc_info=True)
                        logger.error(f"Vector index creation failed completely.")
                        pass

            else:
                logger.info(f"Index '{index_name}' already exists. Skipping index creation.")

        conn.commit()
        logger.info("Database initialization (table and index creation attempt) committed.")
        logger.info("Database initialization successful.")
    except Exception as e:
        logger.error(f"Failed during database initialization: {e}", exc_info=True)
        if conn:
            conn.rollback()
            logger.warning(f"Database initialization transaction rolled back due to error.")
    finally:
        if conn:
            conn.close()
            logger.debug("DB connection closed after initialization.")

# --- Database Interaction Functions ---
def add_vectors_to_db(user_id, document_id, s3_key, chunks_with_vectors):
    """Adds document chunks and their vector embeddings to the database."""
    if not chunks_with_vectors:
        logger.warning("No chunks provided to add to the database.")
        return

    logger.info(
        f"Attempting to add {len(chunks_with_vectors)} chunks for document {document_id} (user {user_id}) to DB.")
    conn = get_db_connection()
    if conn is None:
        logger.error("Failed to add vectors: Could not connect to database.")
        return

    try:
        with conn.cursor() as cur:
            data_to_insert = []
            for chunk_info in chunks_with_vectors:
                chunk_index = chunk_info.get('index')
                content = chunk_info.get('content')
                vector_embedding = chunk_info.get('vector')

                if chunk_index is None or content is None or vector_embedding is None:
                    logger.warning(f"Missing data for a chunk (doc {document_id}, user {user_id}). Skipping.")
                    continue
                if not isinstance(vector_embedding, (list, tuple)) or len(vector_embedding) != embedding_dimension:
                    logger.error(
                        f"Vector embedding for chunk {chunk_index} (doc {document_id}) is not a list/tuple or wrong dim ({len(vector_embedding) if hasattr(vector_embedding, '__len__') else 'N/A'} vs {embedding_dimension}). Skipping.")
                    continue

                data_to_insert.append((user_id, document_id, s3_key, chunk_index, content, vector_embedding))

            try:
                insert_sql = """
                    INSERT INTO document_chunks (user_id, document_id, s3_key, chunk_index, content, vector_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """
                cur.executemany(insert_sql, data_to_insert)
                logger.debug(f"Executed executemany for {len(data_to_insert)} chunks.")

            except Exception as em_e:
                logger.warning(f"executemany failed for vectors: {em_e}. Falling back to execute loop.", exc_info=True)
                conn.rollback()
                logger.warning("Rolling back failed executemany and trying execute loop.")
                with conn.cursor() as cur_fallback:
                    for row_data in data_to_insert:
                        cur_fallback.execute(insert_sql, row_data)
                    logger.debug(f"Executed execute loop for {len(data_to_insert)} chunks.")

            conn.commit()
            logger.info(f"Successfully added {len(data_to_insert)} chunks for document {document_id} to DB.")

    except Exception as e:
        logger.error(f"Failed to add vectors for document {document_id}: {e}", exc_info=True)
        if conn:
            conn.rollback()
            logger.warning(f"Vector insertion transaction for document {document_id} rolled back.")
    finally:
        if conn:
            conn.close()
            logger.debug("DB connection closed after adding vectors.")

def search_vectors_in_db(user_id, query_vector_list, k=5):
    """Searches the database for the most similar vector embeddings."""
    logger.info(f"Attempting vector search for user {user_id} (k={k}).")

    if not isinstance(query_vector_list, (list, tuple, np.ndarray)) or len(query_vector_list) != embedding_dimension:
        logger.error(
            f"Query vector is not a list/tuple/ndarray or wrong dim ({len(query_vector_list) if hasattr(query_vector_list, '__len__') else 'N/A'} vs {embedding_dimension}). Cannot perform search.")
        return []

    conn = get_db_connection()
    if conn is None:
        logger.error("Failed to perform vector search: Could not connect to database.")
        return []

    results = []
    try:
        with conn.cursor() as cur:
            search_sql = """
                SELECT id, user_id, document_id, chunk_index, content, 
                       vector_embedding <=> %s::vector AS distance,
                       vector_embedding
                FROM document_chunks
                ORDER BY distance ASC
                LIMIT %s;
            """

            cur.execute(search_sql, (query_vector_list, k))

            raw_results = cur.fetchall()
            logger.debug(f"Vector search query executed. Found {len(raw_results)} raw results.")

            processed_results = []
            for row in raw_results:
                if len(row) != 7:
                    logger.error(
                        f"Unexpected number of columns in search result row: {len(row)}. Expected 7. Skipping row.")
                    logger.debug(f"Faulty row: {row}")
                    continue

                (chunk_id, chunk_user_id, doc_id, chunk_idx, content, distance, vector_raw_value) = row

                vector_parsed = None
                vector_parse_success = False

                if isinstance(vector_raw_value, (list, tuple, np.ndarray)) and len(
                        vector_raw_value) == embedding_dimension:
                    vector_parsed = vector_raw_value
                    vector_parse_success = True
                elif isinstance(vector_raw_value, str):
                    try:
                        vector_str_list = vector_raw_value.strip("[]").split(",")
                        vector_parsed = [float(x) for x in vector_str_list if x.strip()]
                        vector_parse_success = True
                    except Exception as parse_e:
                        logger.error(
                            f"Failed to parse vector string for chunk {chunk_id}: {vector_raw_value[:100]}... Error: {parse_e}",
                            exc_info=True)
                        vector_parsed = None
                else:
                    logger.warning(
                        f"Vector embedding for chunk {chunk_id} has unexpected type: {type(vector_raw_value)}. Skipping.")
                    vector_parsed = None

                if vector_parse_success and vector_parsed is not None and len(vector_parsed) == embedding_dimension:
                    processed_results.append({
                        'chunk_id': chunk_id,
                        'user_id': chunk_user_id,
                        'document_id': doc_id,
                        'chunk_index': chunk_idx,
                        'content': content,
                        'distance': distance,
                        'vector': vector_parsed
                    })
                else:
                    if vector_parse_success and vector_parsed is not None:
                        logger.warning(
                            f"Skipping chunk {chunk_id} from search results due to incorrect parsed vector dimension ({len(vector_parsed) if hasattr(vector_parsed, '__len__') else 'N/A'} vs {embedding_dimension}).")
                    else:
                        logger.warning(
                            f"Skipping chunk {chunk_id} from search results due to vector parsing failure or incorrect type.")

            logger.info(f"Vector search completed. Processed {len(processed_results)} valid results.")
            results = processed_results

    except Exception as e:
        logger.error(f"Failed to perform vector search for user {user_id}. SQL Query: {search_sql}", exc_info=True)
        return []

    finally:
        if conn:
            conn.close()
            logger.debug("DB connection closed after vector search.")

    return results

# --- Document Processing Pipeline ---
async def save_file_locally(file, file_name):
    """Saves the uploaded file temporarily and returns the path."""
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file_name)
    logger.debug(f"Saving file temporarily to {file_path}")
    try:
        await file.download_to_drive(file_path)
        logger.info(f"File saved successfully to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save file {file_name} locally: {e}", exc_info=True)
        return None

def extract_text_from_document(file_path):
    """Extracts text content from document files (PDF, DOCX, TXT) using alternative libraries."""
    logger.info(f"Attempting to extract text from {file_path}")
    text = ""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            # Try standard text extraction first
            try:
                reader = pypdf.PdfReader(file_path)
                if reader.pages:
                    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    logger.info("Text extracted from PDF using pypdf.")
                else:
                    logger.warning(f"pypdf: PDF file {file_path} contains no readable pages or extraction failed.")
                    text = ""

                # If pypdf failed, try pdfplumber
                if not text or not text.strip():
                    logger.info(f"pypdf extraction failed or returned empty text for {file_path}. Trying pdfplumber...")
                    try:
                        with pdfplumber.open(file_path) as pdf:
                            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                        logger.info("Text extracted from PDF using pdfplumber.")
                    except Exception as pdfplumber_e:
                        logger.error(f"PDF extraction with pdfplumber failed for {file_path}: {pdfplumber_e}",
                                     exc_info=True)
                        text = ""

                # If both standard methods failed, try OCR
                if not text or not text.strip():
                    logger.info(f"Standard PDF extraction methods failed. Attempting OCR extraction...")
                    try:
                        # Convert PDF pages to images and perform OCR
                        from pdf2image import convert_from_path
                        import pytesseract
                        from PIL import Image

                        # Convert PDF to images
                        images = convert_from_path(file_path)
                        logger.info(f"Converted PDF to {len(images)} images for OCR processing.")

                        # Perform OCR on each page
                        ocr_texts = []
                        for i, image in enumerate(images):
                            try:
                                # Convert PIL image to grayscale for better OCR
                                image = image.convert('L')
                                # Perform OCR
                                page_text = pytesseract.image_to_string(image, lang='eng+ukr')
                                if page_text.strip():
                                    ocr_texts.append(page_text)
                                    logger.info(f"OCR successful for page {i + 1}")
                            except Exception as page_e:
                                logger.error(f"OCR failed for page {i + 1}: {page_e}", exc_info=True)

                        if ocr_texts:
                            text = "\n---\n".join(ocr_texts)
                            logger.info("Text extracted from PDF using OCR.")
                        else:
                            logger.warning("OCR extraction yielded no text.")
                            text = ""

                    except Exception as ocr_e:
                        logger.error(f"OCR extraction failed for {file_path}: {ocr_e}", exc_info=True)
                        text = ""

            except Exception as pdf_e:
                logger.error(f"An unexpected error occurred during PDF processing: {pdf_e}", exc_info=True)
                text = ""

        elif file_extension == '.docx':
            try:
                doc = docx.Document(file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                logger.info("Text extracted from DOCX using python-docx.")
            except Exception as docx_e:
                logger.error(f"Failed to extract text from DOCX {file_path}: {docx_e}", exc_info=True)
                text = ""

        elif file_extension == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                logger.info("Text extracted from TXT.")
            except Exception as txt_e:
                logger.error(f"Failed to extract text from TXT {file_path}: {txt_e}", exc_info=True)
                text = ""
        else:
            logger.warning(
                f"Unsupported document extension for standard extraction: {file_extension} for file {file_path}. Text extraction skipped.")
            text = ""

    except Exception as e:
        logger.error(f"An unexpected error occurred during text extraction from {file_path}: {e}", exc_info=True)
        text = ""

    cleaned_text = ' '.join(text.split()).strip()
    logger.debug(f"Extracted text (first 200 chars): {cleaned_text[:200]}...")
    if not cleaned_text:
        logger.warning(f"Extracted text is empty or only whitespace for file {file_path}.")
        return ""

    return cleaned_text

def extract_text_from_photo(file_path):
    """Extracts text from image files using OCR."""
    logger.info(f"Attempting to extract text from photo {file_path} using OCR.")
    text = ""
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        logger.info("Text extracted from photo using OCR.")
    except Exception as e:
        logger.error(f"Failed to extract text from photo {file_path} using OCR: {e}", exc_info=True)
        if "pytesseract.TesseractNotFoundError" in str(e):
            logger.error(
                "Tesseract executable not found. Please install it on your system: sudo apt install tesseract-ocr tesseract-ocr-eng")
        text = ""

    cleaned_text = ' '.join(text.split()).strip()
    logger.debug(f"Extracted OCR text (first 200 chars): {cleaned_text[:200]}...")
    if not cleaned_text:
        logger.warning(f"Extracted OCR text is empty or only whitespace for file {file_path}.")
        return ""

    return cleaned_text

def chunk_text(text, max_chunk_size=1000):
    """Splits text into smaller chunks."""
    if not text:
        logger.warning("No text provided for chunking.")
        return []

    logger.debug(f"Chunking text (length {len(text)}) into size {max_chunk_size}...")
    chunks = []
    words = text.split()
    current_chunk = []
    current_size = 0

    for word in words:
        if current_size + len(word) + (1 if current_chunk else 0) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + (1 if current_chunk else 0)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info(f"Text chunked into {len(chunks)} chunks.")
    return chunks

def get_embedding(text):
    """Gets the vector embedding for a given text string using the embedding model."""
    if embedding_model is None:
        logger.error("Embedding model is not loaded. Cannot get embedding.")
        return None

    if not text or not text.strip():
        logger.warning("Empty or whitespace-only text provided for embedding.")
        return None

    try:
        embeddings = embedding_model.encode([text])
        embedding_vector = embeddings[0]
        logger.debug(f"Got embedding for text (first 5 elements): {embedding_vector[:5]}...")
        return embedding_vector.tolist()
    except Exception as e:
        logger.error(f"Failed to get embedding for text: {e}", exc_info=True)
        logger.debug(f"Failed text (first 100 chars): {text[:100]}...")
        return None

# --- RAG Pipeline Steps Orchestration ---
async def process_document_pipeline(update: Update, file_path: str, s3_key: str = None) -> str:
    """Orchestrates the document processing pipeline."""
    user_id = update.effective_user.id
    document_id = uuid.uuid4()
    file_name = os.path.basename(file_path)
    logger.info(f"Starting document processing pipeline for file: {file_name} (User: {user_id}, Doc ID: {document_id})")

    try:
        await update.effective_chat.send_message(f"Starting document processing: {file_name}...")
        await update.effective_chat.send_chat_action(ChatAction.TYPING)
    except Exception as msg_e:
        logger.warning(f"Failed to send initial status or typing message: {msg_e}")

    extracted_text = ""
    file_extension = os.path.splitext(file_path)[1].lower()

    is_document = file_extension in ['.pdf', '.docx', '.txt']
    is_image = file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    if is_document:
        extracted_text = extract_text_from_document(file_path)
        if extracted_text:
            logger.info(f"Standard text extraction successful for {file_name}. Length: {len(extracted_text)}")
            try:
                await update.effective_chat.send_message(
                    f"Text extracted ({len(extracted_text)} characters). Chunking...")
            except Exception as msg_e:
                logger.warning(f"Failed to send text extraction status: {msg_e}")

    if (not extracted_text or not is_document) and is_image:
        logger.info(
            f"Attempting OCR extraction for file {file_name} (is_image: {is_image}, extracted_text empty: {not extracted_text}).")
        if not extracted_text and is_image:
            try:
                await update.effective_chat.send_message(
                    "Could not extract text directly. Attempting to recognize text in image...")
            except Exception as msg_e:
                logger.warning(f"Failed to send OCR attempt status: {msg_e}")

        extracted_text = extract_text_from_photo(file_path)
        if extracted_text:
            logger.info(f"OCR extraction successful for {file_name}. Length: {len(extracted_text)}")
            try:
                await update.effective_chat.send_message(
                    f"Text recognized ({len(extracted_text)} characters). Chunking...")
            except Exception as msg_e:
                logger.warning(f"Failed to send OCR success status: {msg_e}")

        else:
            logger.error(f"OCR extraction yielded empty text or failed for {file_name}.")

    if not extracted_text:
        logger.error(f"Failed to extract text from document {file_name} using all available methods.")
        status_message = f"Failed to extract text from file {file_name}. Processing stopped.\n"
        try:
            await update.effective_chat.send_message(status_message)
        except Exception as msg_e:
            logger.warning(f"Failed to send final extraction failed status (no text): {msg_e}")

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as rm_e:
                logger.error(f"Failed to remove temp file {file_path}: {rm_e}")
        return status_message

    chunks = chunk_text(extracted_text)
    if not chunks:
        status_message = "Failed to create meaningful chunks from text (text empty after processing?).\n"
        logger.error(f"Failed to create chunks from text for document {file_name} (empty text?).")
        try:
            await update.effective_chat.send_message(status_message)
        except Exception as msg_e:
            logger.warning(f"Failed to send chunking failed status: {msg_e}")

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as rm_e:
                logger.error(f"Failed to remove temp file {file_path}: {rm_e}")
        return status_message

    status_message = f"Text split into {len(chunks)} parts. Generating vector embeddings....\n"
    try:
        await update.effective_chat.send_message(status_message)
    except Exception as msg_e:
        logger.warning(f"Failed to send chunking success status: {msg_e}")

    if embedding_model is None:
        status_message = "Embedding model is not available. Cannot create vector representations.\n"
        logger.critical("Embedding model is None during document processing.")
        try:
            await update.effective_chat.send_message(status_message)
        except Exception as msg_e:
            logger.warning(f"Failed to send embedding model missing status: {msg_e}")

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as rm_e:
                logger.error(f"Failed to remove temp file {file_path}: {rm_e}")
        return status_message

    chunks_with_vectors = []
    batch_size = 32
    logger.debug(f"Starting embedding generation for {len(chunks)} chunks in batches of {batch_size}.")

    progress_interval = max(1, len(chunks) // 10)
    chunks_embedded_count = 0

    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i: i + batch_size]
        try:
            batch_embeddings = embedding_model.encode(chunk_batch, show_progress_bar=False)
            logger.debug(f"Generated embeddings for batch starting with chunk index {i}.")

            for j, embedding_np in enumerate(batch_embeddings):
                chunk_index = i + j
                if isinstance(embedding_np, np.ndarray) and len(embedding_np) == embedding_dimension:
                    chunks_with_vectors.append({
                        'index': chunk_index,
                        'content': chunk_batch[j],
                        'vector': embedding_np.tolist()
                    })
                    chunks_embedded_count += 1
                else:
                    logger.warning(
                        f"Embedding for chunk index {chunk_index} is incorrect format ({type(embedding_np)}) or wrong dimension ({len(embedding_np) if hasattr(embedding_np, '__len__') else 'N/A'} vs {embedding_dimension}). Skipping this chunk.")

            if (i // batch_size + 1) % progress_interval == 0 or (i + batch_size) >= len(chunks):
                progress = min(100, int(((i + batch_size) / len(chunks)) * 100))
                progress_text = f"Embedding generation: {chunks_embedded_count}/{len(chunks)} ({progress}%) complete..."
                try:
                    await update.effective_chat.send_message(progress_text)
                    await update.effective_chat.send_chat_action(ChatAction.TYPING)
                except Exception as msg_e:
                    logger.warning(f"Failed to send embedding progress or typing message: {msg_e}")

        except Exception as embed_e:
            logger.error(f"Failed to get embeddings for a batch of chunks (batch starting index {i}): {embed_e}",
                         exc_info=True)

    if not chunks_with_vectors:
        status_message = "Failed to create vector representations (embeddings) for text parts.\n"
        logger.error(f"No chunks with embeddings generated for document {file_name}.")
        try:
            await update.effective_chat.send_message(status_message)
        except Exception as msg_e:
            logger.warning(f"Failed to send embedding failed status: {msg_e}")

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as rm_e:
                logger.error(f"Failed to remove temp file {file_path}: {rm_e}")
        return status_message

    status_message = f"Embeddings generated for {len(chunks_with_vectors)} parts. Saving to database...\n"
    try:
        await update.effective_chat.send_message(status_message)
    except Exception as msg_e:
        logger.warning(f"Failed to send embedding success status: {msg_e}")

    add_vectors_to_db(user_id, document_id, s3_key, chunks_with_vectors)

    status_message = f"Attempted to save {len(chunks_with_vectors)} parts to database for Document ID: {document_id}.\n"
    status_message += "Document processing complete.\n"
    logger.info(
        f"Document processing pipeline finished for file: {file_name} (User: {user_id}, Doc ID: {document_id}).")
    try:
        await update.effective_chat.send_message(status_message)
    except Exception as msg_e:
        logger.warning(f"Failed to send final pipeline finished status: {msg_e}")

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as rm_e:
            logger.error(f"Failed to remove temp file {file_path}: {rm_e}")

    return status_message

# --- Function to handle RAG query ---
async def handle_rag_query(update: Update, context: ContextTypes.DEFAULT_TYPE, query_text: str) -> str:
    """Handles a user query using RAG: search DB, get context, ask LLM."""
    user_id = update.effective_user.id
    logger.info(f"Initiating RAG query for user {user_id}: '{query_text[:100]}...'")

    thinking_message = None
    try:
        thinking_message = await update.effective_chat.send_message("Searching documents...")
        await update.effective_chat.send_chat_action(ChatAction.TYPING)
        logger.debug("Sent 'searching' message for RAG search.")
    except Exception as tm_e:
        logger.warning(f"Failed to send 'searching' message: {tm_e}")

    missing_components = []
    if embedding_model is None: missing_components.append("Embedding model")
    if gemini_model is None: missing_components.append("Generative model")
    if not all([db_host, db_name, db_user, db_password]): missing_components.append("DB configuration")

    if missing_components:
        logger.error(f"RAG search failed due to missing components: {', '.join(missing_components)}")
        if thinking_message:
            try:
                await context.bot.delete_message(chat_id=update.effective_chat.id,
                                                 message_id=thinking_message.message_id)
            except Exception as del_e:
                logger.warning(f"Failed to delete thinking message: {del_e}")
        return ""

    query_vector = get_embedding(query_text)
    if query_vector is None:
        logger.error("Failed to get embedding for query text.")
        if thinking_message:
            try:
                await context.bot.delete_message(chat_id=update.effective_chat.id,
                                                 message_id=thinking_message.message_id)
            except Exception as del_e:
                logger.warning(f"Failed to delete thinking message: {del_e}")
        return ""

    if not isinstance(query_vector, (list, tuple)) or len(query_vector) != embedding_dimension:
        logger.error(
            f"Query vector obtained has incorrect format or dimension after get_embedding ({len(query_vector) if hasattr(query_vector, '__len__') else 'N/A'} vs {embedding_dimension}).")
        if thinking_message:
            try:
                await context.bot.delete_message(chat_id=update.effective_chat.id,
                                                 message_id=thinking_message.message_id)
            except Exception as del_e:
                logger.warning(f"Failed to delete thinking message: {del_e}")
        return ""

    logger.debug(f"Query vector obtained (first 5 elements): {query_vector[:5]}...")

    k_search_results = 5
    logger.info(f"Calling search_vectors_in_db with k={k_search_results}")
    search_results = search_vectors_in_db(user_id, query_vector, k=k_search_results)

    if not search_results:
        logger.info("Vector search returned no relevant chunks.")
        if thinking_message:
            try:
                await context.bot.delete_message(chat_id=update.effective_chat.id,
                                                 message_id=thinking_message.message_id)
            except Exception as del_e:
                logger.warning(f"Failed to delete thinking message: {del_e}")
        return ""

    logger.info(f"Vector search completed successfully. Found {len(search_results)} relevant chunks.")

    context_chunks_content = [result['content'] for result in search_results if result.get('content')]
    if not context_chunks_content:
        logger.warning("All retrieved chunks have empty content. Cannot generate RAG response.")
        if thinking_message:
            try:
                await context.bot.delete_message(chat_id=update.effective_chat.id,
                                                 message_id=thinking_message.message_id)
            except Exception as del_e:
                logger.warning(f"Failed to delete thinking message: {del_e}")
        return ""

    context_text = "\n---\n".join(context_chunks_content)

    logger.debug(f"Prepared context text (first 500 chars):\n{context_text[:500]}...")

    rag_prompt = f"""
     Answer the following question based only on the provided context. If the answer cannot be found in the context, state that you cannot answer based on the provided information.

     Context:
     {context_text}

     Question:
     {query_text}

     Answer:
     """
    logger.debug(f"Prepared RAG prompt (first 800 chars):\n{rag_prompt[:800]}...")

    logger.info("Calling generative model with RAG context.")

    try:
        # Call the generative model without await
        response = gemini_model.generate_content(rag_prompt)

        if response and hasattr(response, 'text'):
            rag_answer = response.text
        else:
            rag_answer = "Could not get a response from the AI model based on the provided context."
            logger.warning(f"Generative model response had no text attribute for RAG query. Response: {response}")
            if response and hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                logger.warning(f"RAG response blocked for reason: {response.prompt_feedback.block_reason}")

        logger.info(f"Received RAG answer from Gemini (first 200 chars): {rag_answer[:200]}...")
        return rag_answer  # Просто возвращаем ответ, не отправляем его

    except Exception as e:
        logger.error(f"Error calling generative model with RAG context for user {user_id}: {e}", exc_info=True)
        return ""

    finally:
        if thinking_message:
            try:
                await context.bot.delete_message(chat_id=update.effective_chat.id,
                                                 message_id=thinking_message.message_id)
                logger.debug("Deleted 'searching' message.")
            except Exception as del_e:
                logger.warning(f"Failed to delete 'searching' message: {del_e}")

# --- Error Handler ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error by Telegram apps."""
    logger.error("Exception while handling an update:", exc_info=context.error)

# --- Telegram Bot Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command. Greets the user."""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    logger.info(f"User {user_id} ({user_name}) started the bot.")
    await update.message.reply_text(
        f"Hello, {user_name}! I am an AI assistant. Send me a document (TXT, PDF, DOCX) or an image (PNG, JPG) to process, or ask a question in text.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    document = update.message.document
    file_name = document.file_name
    logger.info(f"User {user_id} sent document: {file_name}")

    if embedding_model is None:
        await update.message.reply_text("Sorry, I cannot process documents right now. Embedding model is not loaded.")
        return
    if not all([db_host, db_name, db_user, db_password]):
        await update.message.reply_text("Sorry, I cannot process documents right now. Database not configured.")
        return

    file = await context.bot.get_file(document.file_id)
    temp_file_path = await save_file_locally(file, file_name)
    extracted_text = extract_text_from_document(temp_file_path) if temp_file_path else ""

    # Сохраняем в базу всегда
    if temp_file_path:
        await process_document_pipeline(update, temp_file_path, s3_key=file_name)

    # Если есть подпись — GPT-режим по подписи и тексту
    if update.message.caption and extracted_text:
        user_prompt = update.message.caption
        prompt = f"{user_prompt}\n\n---\nТекст документа:\n{extracted_text}"
        response = gemini_model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            await update.message.reply_text(response.text)
        else:
            await update.message.reply_text("Could not get a response from the AI model.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    photo_sizes = update.message.photo
    if not photo_sizes:
        await update.message.reply_text("Received photo, but could not get photo data.")
        return

    photo = photo_sizes[-1]
    file_id = photo.file_id
    file_name = f"photo_{file_id}.jpg"

    file = await context.bot.get_file(photo.file_id)
    temp_file_path = await save_file_locally(file, file_name)
    extracted_text = extract_text_from_photo(temp_file_path) if temp_file_path else ""

    # Сохраняем в базу всегда
    if temp_file_path:
        await process_document_pipeline(update, temp_file_path, s3_key=file_name)

    # Если есть подпись — GPT-режим по подписи и тексту
    if update.message.caption and extracted_text:
        user_prompt = update.message.caption
        prompt = f"{user_prompt}\n\n---\nТекст документа:\n{extracted_text}"
        response = gemini_model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            await update.message.reply_text(response.text)
        else:
            await update.message.reply_text("Could not get a response from the AI model.")
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user_text = update.message.text

    # --- Групповой чат: реагируем только на / и @ ---
    if update.effective_chat.type != "private":
        bot_username = context.bot.username
        if not (user_text.startswith(f"@{bot_username}") or user_text.startswith("/")):
            return
        if user_text.startswith(f"@{bot_username}"):
            user_text = user_text[len(f"@{bot_username}"):].strip()
        if not user_text:
            return

    logger.info(f"--- handle_text function called ---")
    logger.info(f"User {user_id} in chat {chat_id} sent text: '{user_text[:100]}...'")

    if all([embedding_model, gemini_model, db_host, db_name, db_user, db_password]):
        response = await handle_rag_query(update, context, user_text)

        no_info_found = (
            not response or
            "cannot answer" in response.lower() or
            "cannot find" in response.lower() or
            "does not contain" in response.lower() or
            "provided text" in response.lower()
        )

        if no_info_found:
            logger.info("No relevant information found in knowledge base. Switching to GPT mode.")
            response = await handle_gpt_chat(update, context, user_text)
        await update.message.reply_text(response)
    else:
        logger.info("Knowledge base not available. Using GPT mode.")
        response = await handle_gpt_chat(update, context, user_text)
        await update.message.reply_text(response)

async def handle_gpt_chat(update: Update, context: ContextTypes.DEFAULT_TYPE, query_text: str) -> str:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if gemini_model is None:
        logger.error("Generative model is not initialized for GPT chat.")
        return "AI model is not available for chat. Please check bot configuration."

    thinking_message = None
    try:
        thinking_message = await update.effective_chat.send_message("Thinking...")
        await update.effective_chat.send_chat_action(ChatAction.TYPING)
    except Exception as tm_e:
        logger.warning(f"Failed to send 'thinking' or typing message: {tm_e}")

    try:
        prompt = query_text
        response = gemini_model.generate_content(prompt)

        if response and hasattr(response, 'text'):
            answer = response.text
        else:
            answer = "Could not get a response from the AI model."
            logger.warning(f"Generative model response had no text attribute for GPT chat.")

        return answer

    except Exception as e:
        logger.error(f"Error in GPT chat for user {user_id}: {e}", exc_info=True)
        return "Sorry, an error occurred while generating the response."
    finally:
        if thinking_message:
            try:
                await context.bot.delete_message(
                    chat_id=update.effective_chat.id,
                    message_id=thinking_message.message_id
                )
            except Exception as del_e:
                logger.warning(f"Failed to delete 'thinking' message: {del_e}")

# --- Main function to run the bot ---
def main() -> None:
    """Starts the bot application and sets up handlers."""
    logger.info("Starting bot application...")

    # Check if Telegram token is set
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("Telegram Bot Token is missing. Bot cannot start.")
        return

    # --- DATABASE INITIALIZATION ---
    if all([db_host, db_name, db_user, db_password]):
        initialize_db()
    else:
        logger.warning("DB connection parameters are incomplete. Skipping database initialization.")

    # --- CHECK ESSENTIAL RAG COMPONENTS ---
    if embedding_model is None:
        logger.warning("Embedding model not loaded. Document processing and RAG search will NOT work.")
    if not all([db_host, db_name, db_user, db_password]):
        logger.warning("DB connection parameters incomplete. Database functionality will NOT work.")
    if gemini_model is None:
        logger.warning("Generative AI model not loaded. Chat and RAG response generation will NOT work.")

    # Build the Telegram Application
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # --- Add Handlers ---
    app.add_handler(CommandHandler("start", start_command))  # Handles /start command
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))  # Handles all document messages
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))  # Handles all photo messages

    # ВАЖНО: сначала команды, потом обычный текст!
    app.add_handler(MessageHandler(filters.COMMAND, handle_text))  # Handles /commands (кроме /start)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))  # Handles обычный текст

    app.add_error_handler(error_handler)  # Handles exceptions during update processing

    logger.info("Bot is configured and ready to poll.")
    app.run_polling(poll_interval=5)

# --- Entry point of the script ---
if __name__ == "__main__":
    # Check for Telegram token again before starting
    if not TELEGRAM_BOT_TOKEN:
         logger.critical("Telegram Bot Token is missing. Script will exit.")
         exit(1)

    # Call the main function to start the bot
    main()

