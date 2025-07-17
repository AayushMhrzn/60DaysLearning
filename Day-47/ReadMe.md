#  Day 47: Fine-Tuning a Lightweight Question Answering Model using DistilBERT 

Today, I built and fine-tuned a **Question Answering (QA)** model using the **DistilBERT** architecture, trained on a subset of the **SQuAD** dataset. implemented a **custom training loop using PyTorch**, allowing for deeper control and understanding of the training process. 

---

## What is Question Answering?

Question Answering (QA) is a core NLP task where a model receives:
- A **context paragraph**
- A **question**
And is expected to predict the **start and end position** of the answer in the given context.

---

##  Model Used

- **DistilBERT**: A distilled (compressed) version of BERT that is 40% smaller and 60% faster while retaining ~97% of performance.
- **Pretrained checkpoint**: `distilbert-base-uncased`
- **Task head**: Question Answering (`DistilBertForQuestionAnswering` adds start and end span heads on top of the encoder).

---

##  Dataset

-  **SQuAD v1.1**: Stanford Question Answering Dataset.
-  Only used the **first 1%** (`train[:1%]`) for speed.
- Further split into 90% train and 10% test.

Each example contains:
- `context`: paragraph
- `question`: question about the paragraph
- `answers`: correct answer text and its character-level `start_position`

---

##  Key Steps

###  Tokenization and Preprocessing

Used `DistilBertTokenizerFast` to:
- Tokenize `question` + `context`
- Pad to `max_length=384`
- Compute **token-to-char offset mappings** to align character-level answer spans with token-level positions

###  Encoding and Dataloader

For each example:
- Convert `answer_start` and `answer_end` (in characters) to **token indices**
- Store:
  - `input_ids`
  - `attention_mask`
  - `start_positions`
  - `end_positions`
- Loaded using PyTorch `DataLoader` with batch size = 4

### Model & Optimizer

- Model: `DistilBertForQuestionAnswering`
- Optimizer: `AdamW` with `lr=3e-5`

###  Custom Training Loop

- Trained for 2 epochs(just for learning)
- For each batch:
  - Forward pass
  - Compute span prediction loss
  - Backward + optimizer step
- Printed total loss per epoch


`Note: With only 1% of the dataset, accuracy is expectedly lower. More training data/epochs = better performance.`

---

##  What I Learned

- How to **fine-tune a transformer model manually** without relying on high-level wrappers like `Trainer`
- How to **align character-based answers to token positions**
- How to **evaluate span-based QA tasks**
- Why **tokenization strategy and offset mapping** are crucial in extractive QA
- Practical trade-offs when training on a **tiny subset** of a large dataset

---
