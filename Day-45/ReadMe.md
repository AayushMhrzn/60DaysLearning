# NLP - Sentiment Analysis with Hugging Face Transformers - Day45

## Overview

 a simple yet powerful application of modern Natural Language Processing (NLP) using the **Hugging Face Transformers** library. It performs **sentiment analysis** — classifying text as *positive* or *negative* — using a pretrained transformer model fine-tuned on the **SST-2 dataset**.

---

##  How It Works

### 1. Installation of Required Libraries

We install the following libraries:

* `transformers`: Provides pretrained transformer models for NLP tasks
* `datasets`: For accessing benchmark NLP datasets
* `torch`: PyTorch, the deep learning framework used by the models

```python
!pip install -q transformers datasets torch
```

---

### 2. Loading the Pretrained Model and Tokenizer

We load:

* **Tokenizer**: Converts raw text into subword tokens
* **Model**: Pretrained `distilbert-base-uncased-finetuned-sst-2-english` model for binary sentiment classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

---

### 3. Tokenization and Encoding

* Converts sentences to tokens
* Pads and truncates for uniform input size
* Converts input into PyTorch tensors

```python
texts = [
    "I love learning NLP with Hugging Face!",
    "This is the worst movie I have ever seen.",
    "Transformers models are so powerful and easy to use.",
    "I'm not sure if I like this new phone."
]

encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

---

### 4. Model Inference

We feed the encoded inputs into the model to obtain **logits** (raw prediction scores).

```python
import torch
with torch.no_grad():
    outputs = model(**encoded_inputs)
logits = outputs.logits
```

---

### 5. Calculating Probabilities

We apply **softmax** to convert logits into normalized probabilities.

```python
import torch.nn.functional as F
probs = F.softmax(logits, dim=1)
```

---

### 6. Output Prediction

We interpret the highest probability class and print the results with confidence.

```python
labels = ["Negative", "Positive"]

for text, prob in zip(texts, probs):
    pred_label = labels[prob.argmax()]
    confidence = prob.max().item()
    print(f"Text: {text}")
    print(f"Predicted sentiment: {pred_label} (confidence: {confidence:.2f})")
    print("-" * 50)
```

---

## Summary

This project demonstrates:

* Use of pretrained models for NLP tasks
* Efficient tokenization and batching of input
* Performing inference with transformers
* Mapping outputs to human-readable sentiment labels

---
