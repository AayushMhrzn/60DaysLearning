# Day 58 – Enhancing Interview Bot with NLP analysis for Every Answer + Audio Fix

Today, I upgraded the Interview Bot with deeper NLP analysis and improved the speech input flow. The app now analyzes *every* user response using spaCy and TextBlob, and no longer cuts off early during voice input. These upgrades bring the app one step closer to an intelligent, natural interview simulation.

---

## Key Upgrades & Features:**

### 1.  Improved Audio Input (No Early Cut-Off)

* Previously, the `speech_recognition` listener timed out too quickly, cutting off user speech mid-answer.
* **Fix:** Increased the `timeout` and `phrase_time_limit` to allow for longer, more complete responses.

  ```python
  recognizer.listen(source, timeout=10, phrase_time_limit=20)
  ```
* Now users can take their time to answer fully without worrying about being cut off.

### 2.  NLP Analysis for Every User Answer

* Integrated `spaCy` for Named Entity Recognition (NER) and noun keyword extraction.
* Integrated `TextBlob` for sentiment polarity and subjectivity analysis.
* Each typed or spoken answer is now followed by:

  * Sentiment metrics
  * Named Entities (like names, dates, organizations)
  * Key noun phrases

### 3. Persistent Chat History + NLP Feedback

* Chat sidebar shows:

  * All past questions and user answers
  * NLP feedback underneath each user response
* This makes it feel like a real interview feedback session!

### 4.  Technologies Used:

* `streamlit` – UI Framework
* `streamlit-webrtc` – Webcam feed integration
* `speech_recognition` – Microphone input + Google Speech API
* `spaCy` – Entity recognition and keyword extraction
* `textblob` – Sentiment analysis
* `AV` – Video frame callback handling

---

🔍 **What I Learned:**

* How to analyze NLP on dynamic, step-based chat messages
* How to control rerun behavior without breaking webcam feed
* Best practices for speech recognition and timeout tuning

---
