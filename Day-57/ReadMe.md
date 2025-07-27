# Day 57 – Upgrading the Interview Bot App with Webcam + Voice + Chat Sidebar

Today, I enhanced my Interview Bot built with Streamlit by integrating multiple interactive modalities—webcam feed, voice input with speech recognition, and a persistent sidebar chat UI. These upgrades significantly improve the realism and interactivity of the interview simulation experience.

## Key Upgrades & Features:

### 1. Live Webcam Feed Integration (Non-Interruptive)
Used streamlit-webrtc to display a live webcam feed in the main app area.

Solved the webcam reset issue by:

- Moving the chat interface to the sidebar, which avoids triggering re-runs that interfere with the camera.
- Isolating the camera logic from interactive inputs (text/audio).

### 2. Speech Input with Transcription
Added a "🎙 Speak Answer" button to allow candidates to speak their answers.

Used the speech_recognition library with Google Speech Recognition API for live speech-to-text transcription.

The transcribed answer is appended to chat history and immediately triggers the next interview question.

### 3. Persistent Sidebar Chat UI
Relocated the chat interface to the sidebar (st.sidebar) to separate it from the main page rendering.

Displays:

- Interviewer questions (🤖 Assistant)
- User responses (typed or spoken) (🧑 User)
- Automatically asks the next question after each answer.
- Added a manual ➡️ Next Question button to optionally skip ahead.

### 4 Improved User Flow
Ensures the interview flow follows a question → answer → next question cycle.

Fixed a logic bug where the next question was not shown immediately after an answer.

Used st.rerun() to refresh the sidebar UI after input, while preserving webcam continuity.

## Technologies & Libraries Used:
Streamlit: UI and layout
streamlit-webrtc: Live webcam feed
speech_recognition: Microphone input + Google transcription
spaCy: (Loaded for future NLP analysis of answers)
AV: Frame handling in webcam callback

## What I Learned:

- Real-time input handling with both st.text_input and microphone.

- State management with st.session_state for clean, step-by-step interview logic.

- Layout tricks like moving the chat to st.sidebar to decouple camera rendering from input events.

