import chromadb
import streamlit as st
import os 
from google import genai
from dotenv import load_dotenv

# Configuração da chave da API Gemini
api_key = os.getenv("GOOGLE_API_KEY")
google_client = genai.Client(api_key=api_key)


# Conecta ao Chroma persistente
client = chromadb.PersistentClient("./")
collection = client.get_collection("RAG_Transcription_Assistent")

# Interface Streamlit
st.title("Similarity Search App (Gemini)")
st.markdown("This app uses Chroma to perform similarity searches on a collection of documents and Gemini (Google) to answer questions based on the search results.")
st.sidebar.title("Configuration")
st.sidebar.markdown("Adjust the settings for your query.")
n_results = st.sidebar.number_input("Number of results", min_value=1, max_value=10, value=3)
user_question = st.text_area("Ask a question", key="user_question")


# Função separada para gerar resposta da OpenAI
def get_completion(prompt):
    response = google_client.models.generate_content(
        model="gemini-2.0-flash", contents=["You're a helpful assistant who looks answers up for a user in a textbook and returns the answer to the user's question. If the answer is not in the textbook, you say 'I'm sorry, I don't have access to that information.", prompt]
    )
    return response.text

# Quando o botão é clicado
if st.button("Get Answers"):
    st.write(f"Question: {user_question}")
    st.write(f"Number of Results: {n_results}")
    
    results = collection.query(
        query_texts=[user_question],
        n_results=n_results,
        include=["documents", "metadatas"]
    )

    search_results = []
    for res in results["documents"]:
        for doc, meta in zip(res, results["metadatas"][0]):
            metadata_str = ", ".join(f"{key}: {value}" for key, value in meta.items())
            search_results.append(f"{doc}\nMetadata: {metadata_str}")
    
    search_text = "\n\n".join(search_results)

    prompt = f"""Your task is to answer the following user question using the supplied search results.\n
            User Question: {user_question}\n
            Search Results: {search_text}"""
    response = get_completion(prompt)
    st.markdown("### 💬 Gemini Answer")
    st.write(response)