# Day 23 – Gaussian Naive Bayes from Scratch (with Visualization)

Understand and implement the Gaussian Naive Bayes algorithm from scratch, including how it uses Bayes' Theorem, why it assumes feature independence, and how it handles continuous features using Gaussian distribution.

## What is Naive Bayes?
Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem, with a naive assumption that all features are independent given the class label.

Despite this unrealistic assumption, it performs surprisingly well in many real-world scenarios, especially:
- Spam detection
- Sentiment analysis
- Document classification

## Bayes' Theorem Refresher
We want to compute the posterior probability of a class `C` given input features `X = (x₁, x₂, ..., xₙ):`

``P(y | x) = (P(x | y) * P(y)) / P(x)``

- `P(y | x)` → Probability of class `y` given input `x`
- `P(x | y)` → Likelihood of feature vector under class `y`
- `P(y)` → Prior probability of class `y`
- `P(x)` → Marginal probability of `x` (same for all classes, ignored in classification)




In Naive Bayes, we assume features are independent given the class, so `P(x)` is same for all classes:
``log P(y | x) = log P(y) + Σ log P(x_i | y)``
- In log space (to avoid underflow):
We choose the class with the maximum posterior probability.

## Why called Gaussian Naive Bayes
When features are continuous (not categorical), we model `P(xᵢ | C)` using a Gaussian (Normal) Distribution:
``P(xᵢ | C) = (1 / sqrt(2πσ²)) * exp(- (xᵢ - μ)² / (2σ²))``
Where:

- μ = mean of feature for class C
- σ² = variance of feature for class C
each feature is independent given the class. In reality, this isn't always true, but it works surprisingly well.This is why it’s called Gaussian Naive Bayes.

## Where Bayes' Theorem Is Used
```python
prior = np.log(self.priors[c])
likelihood = np.sum(np.log(self._gaussian(x, self.mean[c], self.var[c])))
posterior = prior + likelihood
```

## Why Do We Separate Classes?
Because the goal of any classification algorithm is to separate classes in feature space. A hyperplane is just a boundary between different classes — and in case of Naive Bayes, the separation happens via probability distributions instead of strict lines.


