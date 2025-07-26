# Day 56 — Interview Bot with Webcam & Chatbot UI (Streamlit)

On Day 56, started building a practical Interview Bot application by integrating several tools and concepts we learned over previous days, including:

- Real-time webcam video feed using streamlit-webrtc

- Interactive chatbot-style UI for recruiter questions and candidate answers using Streamlit’s native chat components

- State management using st.session_state to handle multi-turn conversation flow

## Features Implemented

**Webcam Integration**
Display real-time webcam feed on the app page to simulate a live interview setup.

**Chatbot-style Question & Answer Flow**
The Interview Bot asks a series of predefined questions one-by-one.
Candidates respond by typing their answers into a chat input box.
Conversation history is displayed in a chat format with alternating user (candidate) and bot (recruiter) messages.

**Session State Management**
Used st.session_state to keep track of current question index and the chat history, enabling smooth multi-turn interaction without losing context on reruns.

**Clean UI using Streamlit’s Inbuilt Components**
Leveraged Streamlit’s chat message blocks for a neat conversational UI without any external CSS or frontend frameworks.

## Challenges Faced
- Handling Streamlit’s rerun behavior to keep the chat UI in sync with user inputs.

- Avoiding glitches in webcam feed freezing by correctly setting up the webrtc_streamer.

- Managing bot question flow without duplicated or skipped questions.