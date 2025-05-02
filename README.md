# RAG_app_with_Gemini and ChromaDB
Uses ChromaDB and LangChain to split .txt files into chunks, generate embeddings, and store them for vector search. On user query, it retrieves the most relevant content and sends it to Gemini to provide context-aware responses.

This project is a RAG (Retrieval-Augmented Generation) application that uses **ChromaDB** for vector-based document storage and **Google Gemini (via API)** to generate answers based on user queries. The interface is built with **Streamlit**, enabling semantic search and real-time interaction.

---

### Functionalities
	•	Text chunking using RecursiveCharacterTextSplitter
	•	Similarity search with ChromaDB
	•	Answer generation using the gemini-2.0-flash model

---

### Dependencies 
    •	ChromaDb -- https://docs.trychroma.com/docs/overview/introduction
    •	Streamlit --  https://docs.streamlit.io/develop/api-reference
    •	Google GenAI -- https://ai.google.dev/gemini-api/docs?hl=pt-br
    •	Langchain RecursiveCharacterTextSplitter -- https://python.langchain.com/docs/how_to/recursive_text_splitter/

---

### Installation 
1. Clone the repository: 
    ```
    git clone https://github.com/ViniciusMarcon1/RAG_app_with_Gemini.git
    cd RAG_app_with_Gemini
    ```
2. Install the required packages:
    ```
    pip install chromadb streamlit google-genai langchain python-dotenv
    ```
3. Create a .env file to securely store your Gemini API key:
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    ```
4. (Optional) Run the document chunking script with your own dataset on docs:
    ```
    python Chunking.py
    ```
5. Launch the Streamlit app:
    ```
    streamlit run Search_Generate.py
    ```
Open http://localhost:8501 in your browser to use the app.