# Day 59 – Smarter Interview Bot with Relevance Scoring & Final Summary

- Semantic Relevance Scoring using SpaCy’s medium model (en_core_web_md) combined with keyword overlap to measure how closely the user answers relate to the question.

- Sentiment Analysis using TextBlob to show emotional tone (polarity and subjectivity) for each answer.

-  Enhanced NLP Breakdown: Noun phrases and named entities are extracted and displayed.

- Final Interview Report generated at the end:

Shows each question, the user’s answer, relevance score, sentiment, and final verdict.

| Feature            | Description                                                                        |
| ------------------ | ---------------------------------------------------------------------------------- |
| Semantic Scoring   | Combines SpaCy similarity score and keyword overlap for relevance percentage       |
| Sentiment Feedback | TextBlob provides how positive or negative the tone is                             |
| In-depth Feedback  | Named entities & noun phrases break down answer quality                            |
| Summary Report     | Full overview with average score and verdict (“Likely Fit” or “Needs Improvement”) |

## What I Learned:
Using en_core_web_md greatly improves semantic comparison accuracy.

Combining different NLP signals (similarity, overlap, sentiment) gives richer feedback.

Building a dynamic interview flow with chat history, input handling, and final summary.

Formatting chat and results into a sidebar achieves a cleaner, more intuitive UI.