# Day 18: Decision Tree from Scratch (Classification)

implemented a **Decision Tree Classifier** using only **NumPy and Python**. This helped me deeply understand how decision trees actually work — including **Gini impurity**, **recursive splitting**, and how predictions are made.

---

## What is a Decision Tree?

A Decision Tree is a supervised learning algorithm used for both classification and regression. It works by recursively splitting the dataset based on feature values that result in the **purest subsets**.
A tree structure consists of:

- Root Node: The top node containing the full dataset.
- Decision Nodes: Intermediate nodes where the dataset is split.
- Leaf Nodes: Terminal nodes representing the final output class.

Each internal node splits the data on a **feature + threshold**.  
Each leaf node holds a **class prediction**.

---

## Gini Impurity vs Entropy in Decision Trees

In decision trees, the **splitting criterion** is used to decide where to split the data at each node. Two common criteria are:

---

### Gini Impurity

Gini impurity measures the probability of **misclassifying** a randomly chosen element if it was randomly labeled according to the distribution of labels in the subset.

`Gini(D) = 1 - ∑(p_k)^2`

Where:
- \( p_k \) is the proportion of class \( k \) in the dataset \( D \)
- \( K \) is the number of classes

**Example:**  
If we have 3 classes with proportions 0.5, 0.3, and 0.2:

\[
Gini = 1 - (0.5^2 + 0.3^2 + 0.2^2) = 1 - (0.25 + 0.09 + 0.04) = 0.62
\]

---

### Entropy (Information Gain)

Entropy is based on **information theory**. It measures the expected amount of **information (surprise)** in a randomly drawn sample.

`Entropy(D) = - ∑ p_k * log2(p_k)`

Where:
- \( p_k \) is the proportion of class \( k \) in the dataset \( D \)

**Example:**  
If we have class probabilities 0.5, 0.3, 0.2:

\[
Entropy = -(0.5 \log_2 0.5 + 0.3 \log_2 0.3 + 0.2 \log_2 0.2) \approx 1.485
\]

---

### Key Differences

| Criteria       | Gini Impurity                    | Entropy (Information Gain)              |
|----------------|----------------------------------|------------------------------------------|
| Formula        | \( 1 - \sum p_k^2 \)             | \( -\sum p_k \log_2(p_k) \)              |
| Range          | 0 (pure) to ~0.67 (for 3 classes)| 0 (pure) to \( \log_2(K) \)              |
| Interpretation | Probability of misclassification| Average info needed to identify a class  |
| Performance    | Slightly faster (no log)         | Slightly more precise in some cases      |

---


## Key Concepts

### 1. Gini Impurity (used as split criterion)
Gini measures how mixed/pure the classes are in a node. Lower is better.

\[
\text{Gini}(D) = 1 - \sum_{k=1}^{K} p_k^2
\]

Where:
- \( p_k \) is the proportion of samples belonging to class \( k \).
- Gini = 0 → pure node (all samples same class).

---

### 2. Information Gain (Best Split)

To find the best feature and threshold to split, we compute:

\[
\text{Gain} = \text{Gini}_{\text{parent}} - \left( \frac{n_L}{n} \cdot \text{Gini}_L + \frac{n_R}{n} \cdot \text{Gini}_R \right)
\]

We choose the split that **maximizes gain** (i.e., gives purer children).

---

### 3. Tree Building (Top-down)

- **Start at the root**.
- At each node, compute the **best feature & threshold** to split.
- Recurse to left/right children.
- Stop if:
  - Max depth is reached.
  - All labels are the same.

---

## Step-by-Step Implementation

- Gini Impurity
- Dataset Splitting
- Best Split Finder
- Node Class & Recursive Tree Builder
- Predict and Evaluate
