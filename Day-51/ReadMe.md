# Day 51: Advancing Our Chatbot with LangChain and Groq 

upgraded our basic Streamlit chatbot by integrating **LangChain** for structured chaining and memory, and **Groq** as the fast LLM backend. We aimed to:

* Build a cleaner Chat UI
* Add prompt templates for controlled behavior
* Enable dynamic mode switching (e.g., translator, coder, general assistant)

---

### Why Use LangChain?

LangChain acts as a framework that helps you build **real-world, production-ready AI applications**, not just basic LLM prompts. Here's what it adds to your chatbot:

#### Without LangChain:

* Your chatbot is just sending prompts and getting responses.
* There's no memory or history of the conversation.
* You can't manage complex workflows (e.g. using tools or chaining logic).
* Prompt engineering must be manually managed.

#### With LangChain:

*  **Structured Conversation Chains**: LangChain helps you define a sequence of steps (chains), combining inputs, memory, tools, and responses in a logical flow.
*  **Built-in Memory**: You can store chat history across multiple turns using memory classes like `ConversationBufferMemory`.
*  **Prompt Templates**: You can modularly create custom prompts and easily swap between them.
*  **Tool Integration**: Easily plug in external data (like calculators, search, or PDF retrieval).
*  **Retrieval-Augmented Generation (RAG)**: Seamlessly add document search into your chatbot with vector databases.
*  **Agent Framework**: Build chatbots that can think, plan, and decide what tool or action to take.

In short, **LangChain transforms a simple chatbot into a modular, contextual, and intelligent conversational agent**.

---

###  Today's Features Added

#### 1.  Clean Chat UI with `st.chat_message()`

We used Streamlit's `st.chat_input()` and `st.chat_message()` for a ChatGPT-like experience.

#### 2.  Mode Selection via Sidebar

The user can now select between multiple chatbot personalities:

* General Assistant
* Python Helper
* Nepali Translator

#### 3.  Prompt Templates per Mode

Each mode feeds a different `PromptTemplate` into LangChain, controlling how the bot responds.

---

###  Final Code Highlights

```python
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=TEMPLATES[mode]
)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
response = chain.run(prompt_input)
```

```python
# Chat UI rendering
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
```

---

###  Why Groq?

* Blazing-fast inference speed
* Supports high-performing models like Mixtral, LLaMA3, Gemma
* Works well with LangChain via `langchain-groq`

---
