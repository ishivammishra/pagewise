import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate


load_dotenv()


# Safety check
if not os.getenv("GROQ_API_KEY"):

    raise ValueError("Missing GROQ_API_KEY in .env")


# Cache embeddings
@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="paraphrase-MiniLM-L3-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_and_split_pdf(path):

    loader = PyPDFLoader(path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    return splitter.split_documents(docs)


def create_vector_store(chunks):

    embeddings = load_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


def build_qa_chain(vector_store):

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
    )

    prompt = PromptTemplate(
        template="""

You are a helpful assistant.

Use ONLY provided context.

If answer not found, say:

"I couldn't find this information in the uploaded document."


Context:

{context}


Question:

{question}


Answer:

""",
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain
