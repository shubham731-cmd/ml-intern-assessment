# Evaluation
<!-- 
Please provide a 1-page summary of your design choices for the Trigram Language Model.

This should include:

- How you chose to store the n-gram counts.
- How you handled text cleaning, padding, and unknown words.
- How you implemented the `generate` function and the probabilistic sampling.
- Any other design decisions you made and why you made them. -->

# Trigram Language Model – Design Overview

## 1. Problem & High-Level Design

The goal of this assignment was to implement a trigram (N=3) language model from scratch that can:
- Learn word continuation statistics from a text corpus.
- Handle text cleaning, padding, and unknown words.
- Generate new text probabilistically based on the learned distribution.

I implemented a `TrigramModel` class with three main responsibilities:

1. **Preprocessing & vocabulary building**
2. **Trigram counting / model training**
3. **Probabilistic text generation**

The core data structure for the model is:

- `trigram_counts[(w1, w2)][w3] = count`
- `bigram_counts[(w1, w2)] = total number of times (w1, w2) appears`

This makes it straightforward to estimate conditional probabilities:

\[
P(w_3 \mid w_1, w_2) \approx \frac{\text{count}(w_1, w_2, w_3)}{\text{count}(w_1, w_2)}.
\]

---

## 2. Data & Preprocessing

For the corpus, I used **“Alice’s Adventures in Wonderland” by Lewis Carroll** from Project Gutenberg. The raw text is saved as:

```text
ml-assignment/data/alice_in_wonderland.txt
