# Fake News Detection in Urdu and Roman Urdu

## Overview
This project develops machine learning models to detect fake news in Pakistani 
social media text across two writing systems — Urdu script and Roman Urdu. 
Two independent classifiers are trained and evaluated, followed by cross-script 
evaluation to assess generalization across writing systems.
## Motivation
Pakistan ranks among the top countries affected by misinformation on social 
media. Unlike English fake news detection — which has thousands of labeled 
datasets and pre-trained models — Urdu and Roman Urdu remain severely 
under-resourced languages in NLP research.

A critical challenge unique to Pakistani social media is script inconsistency: 
users freely switch between Urdu script and Roman Urdu (Urdu written in Latin 
characters) within the same platform and sometimes the same conversation. 
This project investigates whether models trained on one script can generalize 
to the other — a question with direct implications for real-world fake news 
detection systems in Pakistan.
## Dataset
Two publicly available datasets were used:

| Dataset | Source | Language | Rows | Labels |
|---|---|---|---|---|
| RUFND | Kaggle (Zainab Noor) | Roman Urdu | 3,800 | fake/real |
| Urdu News Dataset | Kaggle | Urdu Script | 10,440 | fake/real |

**Note:** A third dataset (Amjad et al., 2023) was excluded after inspection 
revealed structural data quality issues — multiple concatenated news articles 
assigned a single label per row, making individual story classification 
unreliable.

All labels were standardized to lowercase fake/real. Four missing values in 
the Urdu News dataset were dropped.
## Project Structure
urdu-fake-news-detection/
│
├── data/
│   ├── raw/                  # Original downloaded files
│   └── processed/            # Cleaned and feature-engineered data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_urdu.ipynb
│   ├── 05_model_roman_urdu.ipynb
│   └── 06_cross_evaluation.ipynb
│
├── models/                   # Saved vectorizers and trained models
├── results/
│   └── figures/              # All generated charts
├── src/
│   └── preprocessing.py
├── requirements.txt
└── README.md
## Methodology
### 1. Data Exploration
Loaded and inspected all three datasets. Identified missing values, label 
inconsistencies, class imbalance, and structural data quality issues.
### 2. Preprocessing
- Standardized column names across datasets
- Standardized labels to lowercase fake/real
- Removed leading noise characters (semicolons, quotes, newlines)
- Dropped 4 missing rows from Urdu News dataset
- Excluded Amjad et al. dataset due to structural quality issues
### 3. Feature Engineering
- Added text length and word count features
- Applied TF-IDF vectorization with unigrams and bigrams (max 5000 features)
- Separate vectorizers fitted for each script to preserve script-specific patterns
### 4. Modelling
Four classifiers trained on each dataset independently:
- Logistic Regression (baseline)
- Linear SVM (best for sparse high-dimensional text data)
- Random Forest (ensemble, non-linear patterns)
- Naive Bayes (benchmark against prior Roman Urdu sentiment work)

Class imbalance addressed using class_weight='balanced' parameter.
Models evaluated using F1 score, precision, and recall — not just accuracy.
### 5. Cross Evaluation
Each best model tested on the other script's data to measure 
cross-script generalization.

## Results

### Model 1: Urdu News Dataset (Urdu Script)

| Model | F1 (fake) | Accuracy |
|---|---|---|
| Linear SVM | 0.9217 | 0.9377 |
| Random Forest | 0.9114 | 0.9301 |
| Logistic Regression | 0.9068 | 0.9248 |
| Naive Bayes | 0.8444 | 0.8793 |

### Model 2: RUFND Roman Urdu

| Model | F1 (fake) | Accuracy |
|---|---|---|
| Linear SVM | 0.9537 | 0.9513 |
| Logistic Regression | 0.9440 | 0.9421 |
| Naive Bayes | 0.9407 | 0.9368 |
| Random Forest | 0.9195 | 0.9145 |

**Linear SVM was the best performing model on both datasets.**

### Cross Evaluation

| Model | Own Test F1 | Cross Dataset F1 | Performance Drop |
|---|---|---|---|
| Urdu News Model | 0.9217 | 0.6803 | 0.2414 |
| Roman Urdu Model | 0.9537 | 0.0793 | 0.8744 |
## Key Finding
Both models perform strongly within their training script. However, 
cross-script evaluation reveals a critical generalization failure.

The Urdu script model drops from F1 0.92 to 0.68 on Roman Urdu data, 
defaulting to predicting fake for most inputs. The Roman Urdu model 
collapses from F1 0.95 to 0.08 on Urdu script data, becoming nearly 
blind to fake news in the other script.

This confirms that script-specific vocabulary patterns dominate learned 
representations, and that Pakistani fake news detection requires dedicated 
multilingual or script-agnostic approaches. A single-script model leaves 
a significant detection gap given that Pakistani social media users 
freely switch between both writing systems.
## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/mrahmadtalha/urdu-fake-news-detection.git
cd urdu-fake-news-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download datasets
- RUFND: https://www.kaggle.com/datasets/zainabnoor02/rufnd-dataset
- Urdu News: https://www.kaggle.com/datasets/hannanbuttdev/urdu-news-articles-original-and-translated

Place downloaded files in `data/raw/` and rename:
- `rufnd_roman_urdu.csv`
- `Urdu_News_Data.csv`

### 4. Run notebooks in order
```
01_data_exploration.ipynb
02_preprocessing.ipynb
03_feature_engineering.ipynb
04_model_urdu.ipynb
05_model_roman_urdu.ipynb
06_cross_evaluation.ipynb
```
## Live Demo
A Streamlit web application is included with two features:

**Dashboard Tab**
- Dataset distribution charts
- Model performance comparison tables
- Cross-script evaluation results and key finding

**Live Prediction Tab**
- Enter any news text in Urdu script or Roman Urdu
- Real-time fake news detection using trained Linear SVM models
- Tip: Use 2-3 sentences minimum for accurate predictions

### Run the app locally
```bash
streamlit run app.py
```

## Technologies Used

- **Python 3.12**
- **pandas** — data manipulation and cleaning
- **numpy** — numerical operations
- **scikit-learn** — TF-IDF vectorization, model training, evaluation
- **matplotlib / seaborn** — visualization
- **Jupyter Notebook** — interactive development
- **Git / GitHub** — version control
## Limitations
- TF-IDF does not capture semantic meaning or word context — 
  word embeddings or transformer models would likely perform better
- Roman Urdu has no standardized spelling — the same word can be 
  written multiple ways, introducing noise the model cannot handle
- Dataset sizes are imbalanced between scripts — 10,440 Urdu script 
  vs 3,800 Roman Urdu rows
- Models were trained on news articles — performance on short social 
  media posts may differ
- No cross-validation was performed due to computational constraints

## Future Work
- Explore multilingual transformer models (XLM-R, mBERT) for 
  script-agnostic fake news detection
- Collect larger Roman Urdu datasets from Pakistani social media platforms
- Develop a unified model trained on both scripts simultaneously
- Build a real-time fake news detection API for Pakistani social media
- Investigate code-switching detection — posts mixing Urdu and English
## Author
**Ahmad Talha Abid**  
BS Computer Science — Virtual University of Pakistan (2024)  
GitHub: https://github.com/mrahmadtalha  

---
*This project was developed as a portfolio piece demonstrating end-to-end 
data science and NLP skills — from data acquisition and preprocessing 
through modelling, evaluation, and cross-script analysis.*