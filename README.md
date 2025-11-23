# 🧠 Trigram Language Model (N = 3)

This project implements a **Trigram Language Model** from scratch as part of the **AI/ML Intern Assignment**.  
The model learns trigram probabilities from text and generates new text using **probabilistic sampling** instead of deterministic word selection.

---

## 📂 Project Structure

```
ml-assignment/
│
├── data/
│   └── alice_in_wonderland.txt        # Training corpus
│
├── src/
│   ├── nagram_model.py                 # Core model implementation
│   ├── generate.py                    # Script to train & generate text
│   └── download_clean_alice.py        # Optional auto-downloader
│
├── tests/
│   └── test_ngram.py                  # Basic correctness tests
│
└── evaluation.md                      # Design choices (1-page summary)
```

---

## ✅ Setup Instructions

### 1. Install dependencies

```
pip install -r requirements.txt
```

If using the optional download script:

```
pip install requests
```

---

## 📥 Download the Training Corpus

### Option A — Automatic (recommended)

```
python ml-assignment/src/download_clean_alice.py
```

### Option B — Manual

1. Download **"Alice’s Adventures in Wonderland"** (plain text) from Project Gutenberg  
2. Save it as:

```
ml-assignment/data/alice_in_wonderland.txt
```

---

## 🚀 Train Model & Generate Text

Run the generator script:

```
python ml-assignment/src/generate.py
```

This will:

✅ load the cleaned book  
✅ train the trigram model  
✅ generate 3 example text samples  

Example output:

```
=== Sample #1 ===
the project gutenberg license 1 e 4 do not charge anything for copies of this agreement and any <unk> format must <unk> the rattling teacups would change to <unk> to notice this question but hurriedly went on that begins with

=== Sample #2 ===
the project gutenberg electronic work within 90 days of receipt of the trees behind him or next day maybe the footman s head with great <unk> and had been before she got to grow up any more and here alice

=== Sample #3 ===
the project gutenberg license for all that said the mouse in the sea cried the mock turtle yawned and shut his note book <unk> out a box of comfits luckily the salt water had not a bit afraid of them
```

---

## 🧪 Run Tests

From project root:

```
pytest -q
```

Expected output:

```
3 passed
```

---

## 🛠️ Model Features

✅ Trigram count dictionary  
✅ Probabilistic sampling using normalized frequencies  
✅ Text preprocessing  
✅ Unknown word handling via `<unk>` token  
✅ Start and end padding tokens (`<s1> <s2> </s>`)  
✅ Generates varied outputs across runs  

---

## 🧠 Summary of Design Choices (short version)

### ✅ Preprocessing
- convert text to lowercase  
- remove punctuation via regex  
- tokenize by whitespace  

### ✅ Vocabulary + Unknown Tokens
- word counts collected
- rare words (≤ threshold) mapped to `<unk>`

### ✅ Trigram Storage Structure
```
(w1, w2) -> { w3: count }
```

### ✅ Probability Estimation
```
P(w3 | w1, w2) = count(w1,w2,w3) / total(w1,w2)
```

### ✅ Generation Strategy
- start from `<s1>, <s2>`
- repeatedly sample next word
- stop on `</s>` or max length or unseen context

### ✅ Trade-offs
✔ simple, clean, readable  
✔ suitable for assignment  
✖ no smoothing  
✖ no true sentence boundary detection  
✖ basic tokenizer  

---

## 🔧 Possible Extensions (if evaluated further)

✅ Laplace or Kneser–Ney smoothing  
✅ Perplexity computation  
✅ Backoff or interpolation  
✅ Sentence segmentation  
✅ Better tokenizer  

---

## 📌 Purpose of This Submission

This project demonstrates:

✅ understanding of probabilistic language models  
✅ ability to implement core NLP logic without libraries  
✅ clean Python coding & organization  
✅ reasoning about design trade-offs  

---

## 🏁 Final Notes

You can now:

✅ run the model  
✅ generate text  
✅ run tests  
✅ submit confidently

