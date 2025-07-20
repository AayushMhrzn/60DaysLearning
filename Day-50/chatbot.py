import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

st.set_page_config(page_title=" BlenderBot Chatbot", page_icon="🤖")
st.title("💬 Chatbot - The Yapper ")

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
    return tokenizer, model

tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask something...")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    inputs = tokenizer(prompt, return_tensors="pt")
    reply_ids = model.generate(**inputs, max_length=150)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
