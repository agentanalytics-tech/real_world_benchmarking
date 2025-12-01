import os
import time
import pandas as pd
from tqdm import tqdm
import PyPDF2
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
import os
import time
import pandas as pd
from tqdm import tqdm
import PyPDF2
import docx  # for DOCX files
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime
from dotenv import load_dotenv
load_dotenv() 
# Initialize the SentenceTransformer model  
model = SentenceTransformer(os.getenv("MODEL_NAME"))
# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Check if index exists, if not create it
PINECONE_INDEX_NAME=os.getenv("PINECONE_INDEX_NAME")
DATA_DIR_FORMATTED = os.getenv("DATA_DIR_FORMATTED")
BATCH_SIZE=int(os.getenv("BATCH_SIZE"))

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=model.get_sentence_embedding_dimension(),
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(PINECONE_INDEX_NAME)

# -------------------------------
# Text extraction functions
# -------------------------------
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = " ".join([para.text for para in doc.paragraphs])
    return text.strip()

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_text(file_path):
    """Detect file type and extract text accordingly"""
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.lower().endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

# -------------------------------
# Metrics dataframe
# -------------------------------
metrics_df = pd.DataFrame(columns=[
    'Batch', 'Files Processed', 'Embedding Time (s)', 
    'Index Preparation Time (s)', 'Upload Time (s)', 'Total Time (s)',
    'Timestamp'
])

# Get all PDF/DOCX files
all_files = [f for f in os.listdir(DATA_DIR_FORMATTED) if f.lower().endswith(('.pdf', '.docx','.txt'))]
total_files = len(all_files)
print(f"Found {total_files} files to process")

# -------------------------------
# Process files in batches
# -------------------------------
batch_num = 1
for i in range(0, total_files, BATCH_SIZE):
    batch_files = all_files[i:i + BATCH_SIZE]
    batch_size = len(batch_files)
    
    print(f"\nProcessing Batch {batch_num} ({batch_size} files)")
    
    # Track metrics
    start_time = time.time()
    embedding_start_time = time.time()
    
    # Process files and generate embeddings
    embeddings = []
    texts = []
    ids = []
    
    for file_name in tqdm(batch_files, desc="Generating embeddings"):
        file_path = os.path.join(DATA_DIR_FORMATTED, file_name)
        file_id = os.path.splitext(file_name)[0]
        
        try:
            # Extract text based on file type
            text = extract_text(file_path)
            
            if text.strip():
                texts.append(text)
                ids.append(file_id)
            else:
                print(f"Warning: Empty text from {file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    
    # Generate embeddings
    if texts:
        batch_embeddings = model.encode(texts)
        
        for i, embedding in enumerate(batch_embeddings):
            embeddings.append((ids[i], embedding.tolist(), {"text": texts[i][:1000]}))  # metadata limited
    
    embedding_time = time.time() - embedding_start_time
    print(f"Embedding generation completed in {embedding_time:.2f} seconds")
    
    # Prepare vectors for Pinecone
    index_prep_start_time = time.time()
    vectors_to_upsert = [
        {'id': id, 'values': embedding, 'metadata': metadata}
        for id, embedding, metadata in embeddings
    ]
    index_prep_time = time.time() - index_prep_start_time
    print(f"Index preparation completed in {index_prep_time:.2f} seconds")
    
    # Upload to Pinecone
    upload_start_time = time.time()
    upsert_batch_size = 100
    for j in range(0, len(vectors_to_upsert), upsert_batch_size):
        upsert_batch = vectors_to_upsert[j:j + upsert_batch_size]
        index.upsert(vectors=upsert_batch, namespace="")
    
    upload_time = time.time() - upload_start_time
    total_time = time.time() - start_time
    
    print(f"Upload to Pinecone completed in {upload_time:.2f} seconds")
    print(f"Total processing time for batch {batch_num}: {total_time:.2f} seconds")
    
    # Record metrics
    metrics_df = pd.concat([metrics_df, pd.DataFrame([{
        'Batch': batch_num,
        'Files Processed': batch_size,
        'Embedding Time (s)': round(embedding_time, 2),
        'Index Preparation Time (s)': round(index_prep_time, 2),
        'Upload Time (s)': round(upload_time, 2),
        'Total Time (s)': round(total_time, 2),
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])], ignore_index=True)
    
    # Save metrics after each batch
    excel_path = r"results\\pinecone_upload_logs.xlsx"
    metrics_df.to_excel(excel_path, index=False)
    print(f"Metrics saved to {excel_path}")
    
    batch_num += 1

print("\nAll batches processed successfully!")
print(f"Total files processed: {total_files}")
print(f"Performance metrics saved to {excel_path}")
