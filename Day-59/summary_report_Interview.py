import streamlit as st
from streamlit_webrtc import webrtc_streamer
import speech_recognition as sr
import spacy
import av
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="Interview Bot", layout="wide")
st.title("🤖 Interview Bot — Webcam + Voice + NLP Scoring")

# --- Load NLP Model ---
nlp = spacy.load("en_core_web_sm")

# --- Interview Questions ---
questions = [
    "Tell me about yourself.",
    "What are your strengths?",
    "Why do you want this job?",
    "Describe a challenge you faced and how you overcame it."
]

# --- Session State Initialization ---
if "q_idx" not in st.session_state:
    st.session_state.q_idx = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "nlp_feedback" not in st.session_state:
    st.session_state.nlp_feedback = []

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

# --- NLP Analysis Function ---
def analyze_answer(answer, question):
    doc = nlp(answer)
    sentiment = TextBlob(answer).sentiment
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    keywords = [chunk.text for chunk in doc.noun_chunks]

    # Relevance score using cosine similarity (TF-IDF)
    vect = TfidfVectorizer().fit([question, answer])
    vectors = vect.transform([question, answer])
    score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100

    return {
        "sentiment": sentiment,
        "entities": entities,
        "keywords": keywords,
        "relevance_score": score
    }

# --- Sidebar Chat Interface ---
with st.sidebar:
    st.header("💬 Interview Chat")

    # Display chat history with NLP analysis
    for idx, msg in enumerate(st.session_state.chat_history):
        role = "🧑" if msg["role"] == "user" else "🤖"
        st.markdown(f"**{role} {msg['role'].capitalize()}:** {msg['content']}")

        if msg["role"] == "user":
            i = idx // 2
            if i < len(st.session_state.nlp_feedback):
                analysis = st.session_state.nlp_feedback[i]
                st.markdown("🔍 **NLP Analysis:**")
                st.markdown(f"- **Relevance Score:** `{analysis['relevance_score']:.2f}%`")
                st.markdown(f"- **Sentiment Polarity:** `{analysis['sentiment'].polarity:.2f}`")
                st.markdown(f"- **Subjectivity:** `{analysis['sentiment'].subjectivity:.2f}`")

                st.markdown("- **Named Entities:**")
                if analysis["entities"]:
                    for ent, label in analysis["entities"]:
                        st.markdown(f"  - `{ent}` ({label})")
                else:
                    st.markdown("  - None")

                st.markdown("- **Keywords:**")
                if analysis["keywords"]:
                    for kw in analysis["keywords"]:
                        st.markdown(f"  - `{kw}`")
                else:
                    st.markdown("  - None")

    # Ask next question if needed
    if st.session_state.q_idx < len(questions):
        if len(st.session_state.chat_history) == 0 or st.session_state.chat_history[-1]["role"] != "assistant":
            next_q = questions[st.session_state.q_idx]
            st.session_state.chat_history.append({"role": "assistant", "content": next_q})

    # Text input
    prompt = st.text_input("📝 Type your answer:")
    if prompt:
        q_idx = st.session_state.q_idx
        question = questions[q_idx] if q_idx < len(questions) else ""
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.nlp_feedback.append(analyze_answer(prompt, question))
        st.session_state.q_idx += 1
        st.rerun()

    # Voice input
    if st.button("🎙 Speak Answer"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎧 Listening... Please speak your full answer.")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)
                transcribed = recognizer.recognize_google(audio)
                st.success(f"📝 Transcribed: {transcribed}")
                q_idx = st.session_state.q_idx
                question = questions[q_idx] if q_idx < len(questions) else ""
                st.session_state.chat_history.append({"role": "user", "content": transcribed})
                st.session_state.nlp_feedback.append(analyze_answer(transcribed, question))
                st.session_state.q_idx += 1
                st.rerun()
            except Exception as e:
                st.error(f"❌ Could not transcribe: {e}")

    # Optional: Skip Question
    if st.button("➡️ Next Question"):
        st.session_state.q_idx += 1
        st.rerun()

# --- Final Summary Report ---
def generate_interview_report(chat_history, nlp_feedback, questions):
    report = ""
    total_score = 0
    answered = 0

    for i, feedback in enumerate(nlp_feedback):
        try:
            question = questions[i]
            user_msg = chat_history[i * 2 + 1]["content"]
        except IndexError:
            continue

        score = feedback["relevance_score"]
        total_score += score
        answered += 1

        report += f"### Question {i+1}: {question}\n"
        report += f"**User Answer:** {user_msg}\n"
        report += f"**Relevance Score:** {score:.2f}%\n"
        report += f"**Sentiment Polarity:** {feedback['sentiment'].polarity:.2f}\n"
        report += f"**Subjectivity:** {feedback['sentiment'].subjectivity:.2f}\n\n"

    if answered > 0:
        avg_score = total_score / answered
        report += "---\n"
        report += f"## 🧠 Final Relevance Score: {avg_score:.2f}%\n"
        if avg_score > 50:
            report += "✅ **Verdict:** Likely Fit for the Role\n"
        else:
            report += "❌ **Verdict:** Needs Improvement\n"
    else:
        report += "No valid answers provided."

    return report

# Show report at end
if st.session_state.q_idx >= len(questions):
    st.markdown("---")
    st.subheader("📋 Interview Summary Report")
    final_report = generate_interview_report(
        st.session_state.chat_history,
        st.session_state.nlp_feedback,
        questions
    )
    st.markdown(final_report)
