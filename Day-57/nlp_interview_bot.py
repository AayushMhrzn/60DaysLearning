import streamlit as st
from streamlit_webrtc import webrtc_streamer
import speech_recognition as sr
import spacy
import av

st.set_page_config(page_title="Interview Bot", layout="wide")

st.title("🤖 Interview Bot — Webcam + Sidebar Chat")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --- Interview Questions ---
questions = [
    "Tell me about yourself.",
    "What are your strengths?",
    "Why do you want this job?",
    "Describe a challenge you faced and how you overcame it."
]

# --- Session State ---
if "q_idx" not in st.session_state:
    st.session_state.q_idx = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Webcam Feed (Main Area) ---
st.markdown("### 📷 Live Webcam Feed")
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="interview_cam",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# --- Sidebar Chat ---
with st.sidebar:
    st.header("💬 Interview Chat")

    # Show previous chat messages
    for msg in st.session_state.chat_history:
        role = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f"**{role} {msg['role'].capitalize()}:** {msg['content']}")

    # Inject first question only once
    if st.session_state.q_idx == 0 and not st.session_state.chat_history:
        first_q = questions[0]
        st.session_state.chat_history.append({"role": "assistant", "content": first_q})

    # Text input
    prompt = st.text_input("📝 Type your answer:")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Ask next question if any left
        st.session_state.q_idx += 1
        if st.session_state.q_idx < len(questions):
            next_q = questions[st.session_state.q_idx]
            st.session_state.chat_history.append({"role": "assistant", "content": next_q})

        st.rerun()

    # Voice input
    if st.button("🎙 Speak Answer"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎧 Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            transcribed = recognizer.recognize_google(audio)
            st.success(f"📝 Transcribed: {transcribed}")
            st.session_state.chat_history.append({"role": "user", "content": transcribed})

            # Ask next question if any left
            st.session_state.q_idx += 1
            if st.session_state.q_idx < len(questions):
                next_q = questions[st.session_state.q_idx]
                st.session_state.chat_history.append({"role": "assistant", "content": next_q})

            st.rerun()
        except Exception as e:
            st.error(f"❌ Could not transcribe: {e}")

    # Optional: Manual skip
    if st.button("➡️ Next Question"):
        st.session_state.q_idx += 1
        if st.session_state.q_idx < len(questions):
            next_q = questions[st.session_state.q_idx]
            st.session_state.chat_history.append({"role": "assistant", "content": next_q})
        st.rerun()
