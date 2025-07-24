# """
# extract text from pdfs
# use gemini text embeddings to create vectors
# prepare metadata for those vectors
# pinecone client
# upsert/upload the vectors into pinecone

# """

from pinecone import Pinecone
from dotenv import load_dotenv
import os
from google import genai
import fitz #PyMuPDF

load_dotenv()
pinecone_api = os.getenv("PINECONE_API")
gemini_api = os.getenv("GEMINI_API")
pinecone_client = Pinecone(api_key=pinecone_api)

vector_index = pinecone_client.Index("student-kb")

google_client = genai.Client(api_key=gemini_api)

def extract_text_from_pdf(pdf_path):
    # poppler)poor implementation --> ghost script --> pdf2text --> fitz
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
        return text

def embed_text(text):
    response = google_client.models.embed_content(
        model= "gemini-embedding-001",
        contents=text,
        config = {
            'output_dimensionality':768
        }
    )
    vector = response.embeddings[0].values

    return vector

def upsert_vectors_to_pinecone(document_texts):
    upsert_data = []
    for idx,text in enumerate(document_texts):
        embedding = embed_text(text)
        vector_id = f"doc-{idx}"
        meta_data = {"text": text}
        upsert_data.append((vector_id,embedding,meta_data))
    vector_index.upsert(upsert_data)
    print(upsert_data)
    print("Vectors upserted succesfully")

if __name__ == "__main__":
    document_texts = []
    document_dirs = os.listdir("documents")
    for file_path in document_dirs:
        text = extract_text_from_pdf(os.path.join("documents", file_path))
        document_texts.append(text)
    upsert_vectors_to_pinecone(document_texts)
    print("all documents process and vectors upserted.")