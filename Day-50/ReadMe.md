# Day50 - Build a Basic LLM Chatbot App using Streamlit

To understand how to build a basic chatbot app using a pre-trained Large Language Model (LLM) and Streamlit. The goal is to explore how inputs are handled, responses are generated, and the conversation is maintained within a Streamlit interface.

---

##  Theoretical Concepts

###  What is a Local LLM Chatbot?

A chatbot powered by a local or lightweight language model that can generate human-like responses based on user prompts. we use Hugging Face’s transformer models like `facebook/blenderbot-400M-distill`.

### Streamlit Overview

Streamlit is a Python-based open-source framework for rapidly building and deploying interactive web apps, especially for machine learning and data science projects.

Key features used:

* `st.set_page_config()`: Sets title and favicon.
* `st.title()`: Adds app title.
* `st.chat_input()`: Adds a chat-style input box.
* `st.chat_message()`: Displays chat bubbles.
* `st.session_state`: Persists data across reruns (used for storing messages).

---

##  How the Chatbot Works

###  Step-by-Step Workflow:

1. **Model Loading**:

```python
@st.cache_resource
```

The model and tokenizer are loaded only once and cached for performance.

2. **Session State for Messages**:

```python
if "messages" not in st.session_state:
    st.session_state["messages"] = []
```

This maintains conversation context by storing all messages exchanged in a list.

3. **Chat Interface Display**:

```python
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
```

Each message (user or assistant) is displayed using `st.chat_message()`.

4. **Taking User Input**:

```python
prompt = st.chat_input("Ask something...")
```

User types their message here.

5. **Generating Response**:

```python
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = model.generate(...)
reply = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
```

* The user’s input is tokenized.
* The model generates a response (text prediction).
* The response is decoded and displayed.

6. **Updating Session State**:

```python
st.session_state.messages.append(...)
```

Both user’s input and assistant’s response are saved to maintain the conversation thread.

---


##  Output Example

**User**: Hello, who are you?

**Assistant**: I'm a helpful assistant trained to chat with you! How can I help you today?

---

##  Key Takeaways

* Streamlit can be used to quickly prototype LLM-based chat apps.
* The `facebook/blenderbot-400M-distill` model provides a free experience.
* Session state maintains persistent conversation.
* Chat UIs can be easily built with Streamlit’s `chat_input()` and `chat_message()` components.

---
