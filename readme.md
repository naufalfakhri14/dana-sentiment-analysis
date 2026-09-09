# DANA App Sentiment Analysis

An end-to-end Natural Language Processing (NLP) project for classifying user reviews of the DANA mobile application into **positive** and **negative** sentiment using a **Bidirectional GRU (BiGRU) with Attention**.

This project covers the complete workflow from data collection and text preprocessing to deep learning, model evaluation, inference, and model saving.

---

## Project Overview

User reviews contain valuable feedback about application quality, usability, reliability, and customer experience. However, analyzing thousands of reviews manually can be time-consuming.

This project uses NLP and deep learning to automatically classify DANA app reviews based on their sentiment.

### Objectives

- Collect a large set of DANA user reviews from Google Play Store.
- Clean and prepare the review text for NLP modeling.
- Convert review text into numerical sequences.
- Build a sentiment classification model using BiGRU and Attention.
- Evaluate model performance on unseen test data.
- Demonstrate sentiment prediction on new reviews.
- Save the trained model and preprocessing objects for reuse.

---

## Data Collection

The reviews were collected from the DANA application on Google Play Store using `google-play-scraper`.

**Application ID:**

```text
id.dana
```

A total of **50,000 raw reviews** were collected.

The raw data is saved as:

```text
dana_raw_50000.csv
```

The scraping process is also available in:

```text
scraping.py
```

---

## Dataset Preparation

After cleaning, deduplication, and sentiment filtering, the final dataset used for modeling contains **29,688 reviews**.

Sentiment labels were determined from the review rating:

| Rating | Sentiment |
|---|---|
| 1–2 | Negative |
| 3 | Excluded |
| 4–5 | Positive |

### Final Sentiment Distribution

| Sentiment | Reviews |
|---|---:|
| Positive | 17,060 |
| Negative | 12,628 |
| **Total** | **29,688** |

Reviews with a rating of 3 were excluded because they were treated as ambiguous for binary sentiment classification.

The processed dataset is saved as:

```text
dana_sentiment_dataset.csv
```

---

## Dataset Split

The final dataset is divided using stratified sampling:

| Dataset | Samples |
|---|---:|
| Training | 23,750 |
| Validation | 2,969 |
| Testing | 2,969 |
| **Total** | **29,688** |

The split ratio is:

- **80% Training**
- **10% Validation**
- **10% Testing**

Stratification is used to maintain the sentiment distribution across the three subsets.

---

## Text Preprocessing

The review text goes through several preprocessing stages before being used by the model.

The text is:

1. Cleaned using a custom text cleaning function.
2. Tokenized using Keras `Tokenizer`.
3. Converted into integer sequences.
4. Padded or truncated to a fixed sequence length.

### Tokenization Configuration

| Parameter | Value |
|---|---:|
| Maximum Vocabulary | 20,000 words |
| Maximum Sequence Length | 50 |
| Padding | `post` |
| Truncating | `post` |
| OOV Token | `<OOV>` |

The tokenizer is fitted only on the training data.

The resulting vocabulary contains **15,474 words**.

---

## Model Architecture

The sentiment classifier uses a **Bidirectional GRU (BiGRU) with Attention**.

The model architecture is:

```text
Input Sequence (50 tokens)
        ↓
Embedding
        ↓
Bidirectional GRU (128 units)
        ↓
Attention
        ↓
GlobalAveragePooling1D
        ↓
Dense (64, ReLU)
        ↓
Dropout (0.4)
        ↓
Dense (2, Softmax)
```

### Main Components

- **Embedding** — converts token IDs into dense vector representations.
- **Bidirectional GRU** — captures contextual information from both directions of the text.
- **Attention** — allows the model to focus on important parts of the sequence.
- **GlobalAveragePooling1D** — reduces the sequence representation into a fixed-size vector.
- **Dense + Dropout** — performs classification while helping reduce overfitting.
- **Softmax** — produces probabilities for the two sentiment classes.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Embedding Dimension | 128 |
| BiGRU Units | 128 |
| Dense Units | 64 |
| Dropout | 0.4 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 32 |
| Maximum Epochs | 15 |
| Callback | EarlyStopping |

Training was stopped early when validation performance stopped improving.

---

## Model Performance

The final model achieved:

| Metric | Accuracy |
|---|---:|
| Training Accuracy | **91.33%** |
| Validation Accuracy | **86.02%** |
| Testing Accuracy | **85.65%** |

The model achieved **85.65% accuracy on the hold-out test set**.

---

## Classification Report

Performance on the test set:

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Negative | 0.82 | 0.86 | 0.84 |
| Positive | 0.89 | 0.86 | 0.87 |
| **Accuracy** | | | **0.86** |

The model shows relatively balanced recall between positive and negative sentiment on the test set.

---

## Confusion Matrix

A confusion matrix is included in the notebook to visualize correct and incorrect predictions for both sentiment classes.

The model evaluates:

- Actual Negative vs. Predicted Negative
- Actual Negative vs. Predicted Positive
- Actual Positive vs. Predicted Negative
- Actual Positive vs. Predicted Positive

---

## Inference

The project includes an inference function that accepts a new review and returns:

- Predicted sentiment
- Prediction confidence

Example predictions from the project:

```text
"aplikasinya sangat bagus dan mudah digunakan"
→ Positive
→ Confidence: 0.9979
```

```text
"dana saya sering error dan tidak bisa transfer"
→ Negative
→ Confidence: 0.8117
```

```text
"transaksi di dana sangat cepat dan praktis"
→ Positive
→ Confidence: 0.9965
```

```text
"aplikasinya sering gangguan dan tidak bisa dipakai"
→ Negative
→ Confidence: 0.8657
```

---

## Model and Preprocessing Artifacts

The trained model and preprocessing objects are saved for reuse:

```text
dana_sentiment_model.keras
dana_tokenizer.pkl
dana_label_encoder.pkl
```

These files make it possible to reuse the trained model without rebuilding the preprocessing pipeline from scratch.

---

## Project Structure

```text
dana-sentiment-analysis/
│
├── dana_sentiment_model.keras
├── dana_tokenizer.pkl
├── dana_label_encoder.pkl
│
├── dana_sentiment_dataset.csv
├── dana_raw_50000.csv
│
├── scraping.py
├── kode2.ipynb
├── requirements.txt
└── README.md
```

> Large raw or processed datasets can be excluded from the public GitHub repository when repository size is a concern. The notebook and scraping script document how the data was collected and prepared.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Play Scraper
- NLP
- Deep Learning
- GRU
- Attention Mechanism

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd dana-sentiment-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Open:

```text
kode2.ipynb
```

Run the notebook from the beginning to reproduce the data preparation, model training, evaluation, and inference workflow.

### 4. Collect new reviews

To collect new DANA reviews, run:

```bash
python scraping.py
```

The script uses the DANA application ID:

```text
id.dana
```

---

## Key Takeaways

This project demonstrates an end-to-end NLP workflow for analyzing mobile application reviews:

```text
Google Play Reviews
        ↓
Data Collection
        ↓
Text Cleaning
        ↓
Rating-Based Sentiment Labeling
        ↓
Tokenization & Padding
        ↓
BiGRU + Attention
        ↓
Model Evaluation
        ↓
Sentiment Inference
```

The final model achieved **85.65% testing accuracy** on 2,969 unseen reviews.

---


## Author

**Naufal Fakhri**

Data Science | Machine Learning | Artificial Intelligence

---
