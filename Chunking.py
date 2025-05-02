import os 
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Iniciando um client persistente no Chroma
client = chromadb.PersistentClient("./")

# Iniciando uma nova collection que é um container que serve para armazenar as embeddings e seus dados associados (metadatas)
collection = client.get_or_create_collection(
    name="RAG_Transcription_Assistent",
    metadata={"hnsw:space": "cosine"}
)

# Criando um divisor de textos
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", "? ", "! "],  # Lista de caracteres para priorizar na divisão
    chunk_size=1000,  # Tamanho máximo das chunks 
    chunk_overlap=50,  # Overleap máximo das chunks 
)

docs_dir = "./docs"

# Arquiva os dados em files e separa metadata
files = [
    {
        "title": os.path.splitext(filename)[0].replace("_", " ").title(),
        "source_url": "None",
        "filename": filename
    }
    for filename in os.listdir(docs_dir)
    if filename.endswith(".txt")
]

# Inicializa listas para guardar os documentos, a metadata e o id
metadatas = []
ids = []
documents = []

# Leitura dos arquivos e criação dos chunks
for file_meta in files:
    file_path = os.path.join(docs_dir, file_meta["filename"])
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        chunks = text_splitter.create_documents([content])

        for index, chunk in enumerate(chunks):
            metadatas.append({
                "title": file_meta["title"],
                "source_url": file_meta["source_url"],
                "chunk_idx": index
            })
            ids.append(f"{file_meta['filename']}_{index}")
            documents.append(chunk.page_content)

print("Add Iniciado")
collection.add(documents=documents, metadatas=metadatas, ids=ids)
print("Add Finalizado")

print("Número de documentos armazenados:", collection.count())
