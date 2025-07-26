import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

st.title("Interview Bot — Using Streamlit Chat UI")

questions = [
    "Tell me about yourself.",
    "What are your strengths?",
    "Why do you want this job?",
    "Describe a challenge you faced and how you overcame it."
]

if "q_idx" not in st.session_state:
    st.session_state.q_idx = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Handle new user input first
prompt = st.chat_input("Your answer:")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.q_idx += 1

# Then show chat history including latest user input
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Show next bot question if available and last message isn't already from bot
if st.session_state.q_idx < len(questions):
    if len(st.session_state.chat_history) == 0 or st.session_state.chat_history[-1]["role"] != "assistant":
        bot_msg = questions[st.session_state.q_idx]
        st.session_state.chat_history.append({"role": "assistant", "content": bot_msg})
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

# Webcam feed
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.write("---")
webrtc_streamer(
    key="interview_bot",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
