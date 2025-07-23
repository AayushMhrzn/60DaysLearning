import os
import json
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from tempfile import NamedTemporaryFile


st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 ChatGuru - Chat with Memory + Document QA")

# Load API key securely
groq_api_key = os.getenv("GROQ_API_KEY")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar controls
st.sidebar.title("🛠️ Settings")
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()

mode = st.sidebar.selectbox(
    "Select Chat Mode",
    ["General Assistant", "Python Helper", "Nepali Translator"]
)

# File upload for RAG
uploaded_file = st.sidebar.file_uploader("Upload PDF for QA", type="pdf")
retriever = None
qa_chain = None

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        filepath = tmp.name

    loader = PyMuPDFLoader(filepath)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = splitter.split_documents(pages)

    # Using HuggingFaceInstructEmbeddings instead of sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    st.sidebar.success("Document processed. Ask questions below ⬇️")

# Prompt templates for chatbot modes
TEMPLATES = {
    "General Assistant": """
You are a helpful, friendly assistant.

Conversation so far:
{chat_history}

User: {input}
Assistant:""",
    "Python Helper": """
You are a Python expert. Answer with code when appropriate and explain briefly.

Conversation so far:
{chat_history}

User: {input}
PythonBot:""",
    "Nepali Translator": """
You are a Nepali language translator. Translate clearly between English and Nepali as needed.

Conversation so far:
{chat_history}

User: {input}
Translator:"""
}

# Define prompt for RAG
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the following context to answer the question:

Context:
{context}

Question:
{question}

Helpful Answer:
"""
)

# Initialize Groq LLM
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key)

# Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup RAG chain if retriever available
if retriever:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

# Setup default chain for chatbot modes
prompt = PromptTemplate(input_variables=["chat_history", "input"], template=TEMPLATES[mode])
default_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Chat UI input
user_input = st.chat_input("Type your message...")
if user_input:
    if qa_chain:
        response = qa_chain.run({"question": user_input})
    else:
        history_str = "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history if msg])
        response = default_chain.run({"input": user_input, "chat_history": history_str})

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", response))

# Display chat messages
for role, msg in st.session_state.chat_history:
    if msg:
        with st.chat_message(role):
            st.markdown(msg)
