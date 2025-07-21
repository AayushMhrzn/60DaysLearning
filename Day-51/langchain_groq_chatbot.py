import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM (Mixtral, Gemma, LLaMA3)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"  # You can also try "llama3-70b-8192" or "gemma-7b-it"
)

# Streamlit UI
st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")
st.title("CHATGURU 💬")

# Chat mode selection
mode = st.sidebar.selectbox(
    "Select Chat Mode",
    ["General Assistant", "Python Helper", "Nepali Translator"]
)

# Prompt templates for different modes
TEMPLATES = {
    "General Assistant": """
You are a helpful, friendly assistant.

Conversation so far:
{history}

User: {input}
Assistant:""",
    "Python Helper": """
You are a Python expert. Answer with code when appropriate and explain briefly.

Conversation so far:
{history}

User: {input}
PythonBot:""",
    "Nepali Translator": """
You are a Nepali language translator. Translate clearly between English and Nepali as needed.

Conversation so far:
{history}

User: {input}
Translator:"""
}

# Initialize memory
memory = ConversationBufferMemory(memory_key="history")

# Build prompt and chain
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=TEMPLATES[mode]
)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input from user
if prompt_input := st.chat_input("Type your message..."):
    # Run LLMChain
    response = chain.run(prompt_input)

    # Store messages
    st.session_state.chat_history.append(("user", prompt_input))
    st.session_state.chat_history.append(("assistant", response))

# Display messages
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)