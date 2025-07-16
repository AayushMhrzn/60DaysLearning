**Day 46 - Dependency Parsing & Constituency Parsing - Day46**

Learnt how to analyze sentence **structure and grammar** using two powerful parsing techniques in NLP:

* **Dependency Parsing**: Understand grammatical relationships
* **Constituency Parsing**: Understand sentence phrase structure

---

### DEPENDENCY PARSING

**Goal**: Identify head-dependent relations between words in a sentence.

**Example**:
Sentence: *"The cat sat on the mat."*

| Word | Head | Relation    |
| ---- | ---- | ----------- |
| The  | cat  | determiner  |
| cat  | sat  | subject     |
| sat  | ROOT | root verb   |
| on   | sat  | preposition |
| the  | mat  | determiner  |
| mat  | on   | object      |

**Use case**: Grammar analysis, relation extraction

---

### CONSTITUENCY PARSING

**Goal**: Break down sentence into nested **phrases** using a tree structure.

**Tree View**:

```
(S
  (NP (DT The) (NN cat))
  (VP (VBD sat)
      (PP (IN on)
          (NP (DT the) (NN mat)))))
```

* S: Sentence
* NP: Noun Phrase
* VP: Verb Phrase
* PP: Prepositional Phrase

**Use case**: Text summarization, question answering

---

### LIBRARIES USED

* **spaCy** — for dependency parsing
* **Stanza** — for constituency parsing

```bash
pip install spacy stanza
python -m spacy download en_core_web_sm
```

---

### USING spaCy (Dependency)

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps over the lazy dog.")
for token in doc:
    print(token.text, "->", token.dep_, "->", token.head.text)
```

---

### USING STANZA (Constituency)

```python
import stanza
stanza.download("en")
nlp = stanza.Pipeline("en", processors="tokenize,pos,constituency")
doc = nlp("The quick brown fox jumps over the lazy dog.")
doc.sentences[0].constituency.pretty_print()
```

---

### USE CASE COMPARISON

| Task                  | Dependency | Constituency |
| --------------------- | ---------- | ------------ |
| Grammatical structure | ✅ Yes      | ❌ No         |
| Phrase-level grouping | ❌ No       | ✅ Yes        |
| Tree interpretability | Moderate   | High         |

---

### SUMMARY

* **Dependency Parsing**: Who does what to whom?
* **Constituency Parsing**: How is the sentence built?
* Combined use helps with **deep syntactic and semantic NLP tasks**.

