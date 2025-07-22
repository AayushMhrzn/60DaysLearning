import streamlit as st
import json
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")
history_file = "chat_history.json"

# Load chat history from JSON
def load_history():
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            return json.load(f)
    return []

# Save chat history to JSON
def save_history(history):
    with open(history_file, "w") as f:
        json.dump(history, f)

# Clear chat history
def clear_history():
    if os.path.exists(history_file):
        os.remove(history_file)

# Load and display sidebar
st.set_page_config(page_title="Persistent Groq Chatbot", page_icon="🧠")
st.title("CHATGURU 💬")
st.sidebar.title("🛠️ Settings")
if st.sidebar.button("Clear Chat History"):
    clear_history()
    st.session_state.chat_history = []
    st.rerun()

# Select chatbot mode
mode = st.sidebar.selectbox(
    "Select Chat Mode",
    ["General Assistant", "Python Helper", "Nepali Translator"]
)

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

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"
)

# Restore history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history()

# Use LangChain memory with restored messages
memory = ConversationBufferMemory()
for role, msg in st.session_state.chat_history:
    if msg and role == "user":
        memory.chat_memory.add_user_message(msg)
    elif msg and role == "assistant":
        memory.chat_memory.add_ai_message(msg)
# Prompt and chain
prompt = PromptTemplate(input_variables=["history", "input"], template=TEMPLATES[mode])
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Input
if user_input := st.chat_input("Type your message..."):
    response = chain.run(user_input)

    # Save to session state
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", response))
    save_history(st.session_state.chat_history)

# Display messages
for role, msg in st.session_state.chat_history:
    if msg is not None:
        with st.chat_message(role):
            st.markdown(msg)