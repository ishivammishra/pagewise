# 📄 Pagewise

> Turn any PDF into a conversation.

Pagewise is a RAG-powered document chat app. Upload any PDF and ask
questions about it in plain English, no reading required.

## 🌐 Live Demo

https://ishivammishra-pagewise.streamlit.app

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq — LLaMA 3.1 8B (free) |
| RAG Pipeline | LangChain |
| Embeddings | HuggingFace paraphrase-MiniLM-L3-v2 |
| Vector Store | ChromaDB (in-memory) |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

## 🧠 How RAG Works

1. PDF uploaded and split into 1000-char chunks
2. Chunks converted to vectors using HuggingFace embeddings
3. Vectors stored in ChromaDB in-memory
4. User question vectorized and matched via MMR search
5. Top 5 relevant chunks sent to LLaMA 3.1 with question
6. Answer generated based only on document content

## ⚙️ Run Locally

Clone the repo and install dependencies:

    git clone https://github.com/ishivammishra/pagewise
    cd pagewise
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

Create a .env file:

    GROQ_API_KEY=your_key_here

Run:

    streamlit run app.py

## 🐳 Run with Docker

    docker build -t pagewise .
    docker run -p 8501:8501 -e GROQ_API_KEY=your_key pagewise

## 👤 Author

Shivam Mishra
- LinkedIn: https://www.linkedin.com/in/shivam-mishra-679a811b5/
- GitHub: https://github.com/ishivammishra