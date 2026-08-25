# Fake News Detection in Urdu and Roman Urdu: A Cross-Script Analysis

**Ahmad Talha Abid**  
BS Computer Science, Virtual University of Pakistan  
GitHub: https://github.com/mrahmadtalha

---

## Abstract

Misinformation on social media poses a significant challenge in Pakistan,
where users communicate in both Urdu script and Roman Urdu — the same
language written in two different scripts. While fake news detection has
been extensively studied for English, Urdu and Roman Urdu remain severely
under-resourced in NLP research. This project develops two independent
machine learning classifiers for fake news detection — one trained on Urdu
script news articles (10,440 samples) and one on Roman Urdu news
(3,800 samples). Both models achieve strong within-script performance,
with Linear SVM reaching F1 scores of 0.92 and 0.95 respectively.
However, cross-script evaluation reveals a critical generalization failure —
the Roman Urdu model collapses to F1 0.08 on Urdu script data, and the
Urdu script model drops to F1 0.68 on Roman Urdu data. These findings
confirm that script-specific vocabulary dominates learned representations,
and that effective Pakistani fake news detection requires multilingual
or script-agnostic approaches.

---

## 1. Introduction

Pakistan ranks among the top countries affected by social media
misinformation. Platforms like Twitter, Facebook, and WhatsApp are primary
news sources for millions of Pakistanis, making fake news detection
critically important. Unlike English, where large labeled datasets and
pre-trained transformer models are readily available, Urdu and Roman Urdu
remain low-resource languages with limited NLP tooling.

A unique challenge in Pakistani social media is script inconsistency.
Pakistani users freely switch between Urdu script (Arabic-derived) and
Roman Urdu (Urdu written in Latin characters) within the same platform
and sometimes the same conversation. This raises a critical research
question: can a fake news detection model trained on one script generalize
to the other?

This project addresses this question by training separate models for each
script and conducting systematic cross-script evaluation. The findings have
direct implications for real-world fake news detection systems targeting
Pakistani social media.

---

## 2. Related Work

Several studies have addressed fake news detection in low-resource languages.
Amjad et al. (2023) developed a fake news detection dataset for Urdu,
demonstrating that classical ML approaches achieve competitive performance
on Urdu text classification. Hanif et al. (2022) explored sentiment
analysis in Roman Urdu, highlighting the challenges of non-standardized
spelling and code-switching. Zubiaga et al. (2018) provided a comprehensive
survey of rumor detection on social media, establishing evaluation
frameworks applicable to multilingual settings. Rahman et al. (2021)
demonstrated that TF-IDF with Linear SVM consistently outperforms other
classical approaches on high-dimensional sparse text data. Unlike prior
work, this project specifically investigates cross-script generalization —
an understudied problem directly relevant to Pakistani social media.

---

## 3. Dataset

Two publicly available datasets were used:

| Dataset | Source | Language | Samples | Fake | Real |
|---|---|---|---|---|---|
| RUFND | Kaggle (Zainab Noor) | Roman Urdu | 3,800 | 2,000 | 1,800 |
| Urdu News Dataset | Kaggle | Urdu Script | 10,440 | 4,101 | 6,339 |

A third dataset (Amjad et al., 2023) was excluded after inspection revealed
structural data quality issues — multiple concatenated news articles were
assigned a single label per row, making individual story classification
unreliable. This decision reflects the importance of data quality over
dataset size.

The RUFND dataset shows mild class imbalance (53% fake, 47% real). The
Urdu News dataset shows moderate imbalance (61% real, 39% fake), addressed
using class_weight='balanced' during model training.

---

## 4. Methodology

### 4.1 Preprocessing

Raw datasets underwent the following preprocessing steps:
- Column names standardized across datasets
- Labels standardized to lowercase fake/real
- Leading noise characters removed (semicolons, quotes, newlines)
- Four missing rows dropped from Urdu News dataset
- Trailing whitespace stripped from all text and label fields

### 4.2 Feature Engineering

Two text-derived features were added prior to vectorization:
- **text_length** — character count of each news item
- **word_count** — token count of each news item

Analysis revealed an interesting cross-dataset pattern: in RUFND, fake
news articles are longer than real ones (95 vs 77 characters average),
while in Urdu News, real articles are longer than fake ones (1,799 vs
1,413 characters average). This suggests fake news length patterns
differ by source and context.

TF-IDF vectorization was applied separately to each dataset:
- Unigrams and bigrams (ngram_range=(1,2))
- Maximum 5,000 features
- Minimum document frequency of 2
- Maximum document frequency of 95%

Separate vectorizers were fitted for each script to preserve
script-specific vocabulary patterns.

### 4.3 Models

Four classifiers were trained on each dataset independently:

| Model | Justification |
|---|---|
| Logistic Regression | Interpretable baseline |
| Linear SVM | Theoretically optimal for sparse high-dimensional text |
| Random Forest | Ensemble method, captures non-linear patterns |
| Naive Bayes | Benchmark against prior Roman Urdu classification work |

All models used an 80/20 train/test split with stratification.
Class imbalance was addressed using class_weight='balanced'.
Models were evaluated using F1 score, precision, and recall — not
just accuracy — with particular focus on the fake class, as missing
fake news carries greater real-world risk than false alarms.

---

## 5. Results

### 5.1 Urdu News Dataset (Urdu Script)

| Model | F1 (fake) | Precision (fake) | Recall (fake) | Accuracy |
|---|---|---|---|---|
| Linear SVM | 0.9217 | 0.91 | 0.93 | 0.9377 |
| Random Forest | 0.9114 | 0.91 | 0.92 | 0.9301 |
| Logistic Regression | 0.9068 | 0.88 | 0.93 | 0.9248 |
| Naive Bayes | 0.8444 | 0.85 | 0.83 | 0.8793 |

### 5.2 RUFND Roman Urdu Dataset

| Model | F1 (fake) | Precision (fake) | Recall (fake) | Accuracy |
|---|---|---|---|---|
| Linear SVM | 0.9537 | 0.95 | 0.95 | 0.9513 |
| Logistic Regression | 0.9440 | 0.96 | 0.93 | 0.9421 |
| Naive Bayes | 0.9407 | 0.93 | 0.95 | 0.9368 |
| Random Forest | 0.9195 | 0.91 | 0.93 | 0.9145 |

Linear SVM was the best performing model on both datasets, confirming
its suitability for sparse TF-IDF feature spaces.

### 5.3 Cross-Script Evaluation

| Model | Own Test F1 | Cross Dataset F1 | Performance Drop |
|---|---|---|---|
| Urdu News Model | 0.9217 | 0.6803 | 0.2414 |
| Roman Urdu Model | 0.9537 | 0.0793 | 0.8744 |

---

## 6. Discussion

### 6.1 Within-Script Performance
Both models achieve strong results within their training script. Linear SVM
consistently outperforms other classifiers, supporting theoretical
expectations for high-dimensional sparse text classification. The Roman Urdu
model marginally outperforms the Urdu script model (0.95 vs 0.92 F1),
likely due to better class balance in the RUFND dataset.

### 6.2 Cross-Script Generalization Failure
Cross-script evaluation reveals a critical finding. The Urdu script model
drops from F1 0.92 to 0.68 on Roman Urdu data, defaulting to predicting
fake for most inputs (recall for real class drops to 0.04). The Roman Urdu
model collapses from F1 0.95 to 0.08 on Urdu script data, becoming nearly
blind to fake news in the other script.

This failure confirms that TF-IDF representations are dominated by
script-specific surface features rather than semantic content. The models
learn vocabulary patterns tied to script, not meaning. When tested on an
unseen script, the learned vocabulary is almost entirely absent from the
TF-IDF feature space, causing the model to default to majority-class
predictions.

### 6.3 Implications
Pakistani social media users switch freely between Urdu script and Roman
Urdu. A deployment-ready fake news detection system cannot rely on
single-script models — it would miss large volumes of misinformation in
the untrained script. This finding motivates future work on multilingual
embeddings and script-agnostic approaches for Pakistani NLP.

---

## 7. Limitations

- TF-IDF does not capture semantic meaning or word context
- Roman Urdu has no standardized spelling — the same word can appear
  in multiple forms, introducing noise
- Dataset sizes differ significantly between scripts (10,440 vs 3,800)
- Models trained on news articles may not generalize to short social
  media posts
- No cross-validation was performed due to computational constraints
- Results reflect dataset characteristics and may not generalize to
  all Pakistani fake news domains

---

## 8. Conclusion

This project developed and evaluated fake news detection models for Urdu
script and Roman Urdu, achieving strong within-script performance (F1 0.92
and 0.95). Cross-script evaluation revealed a critical generalization
failure, with the Roman Urdu model collapsing to F1 0.08 on Urdu script
data. This finding confirms that script-specific vocabulary patterns
dominate TF-IDF representations, and that effective Pakistani fake news
detection requires multilingual or script-agnostic approaches. Future work
should explore transformer-based models such as XLM-R or mBERT, which
operate on subword tokens and are less sensitive to script differences.

---

## 9. References

1. Amjad, M., et al. (2023). Fake news detection for Urdu language.
   PeerJ Computer Science.
2. Hanif, M., et al. (2022). Sentiment analysis for Roman Urdu text
   on social media.
3. Zubiaga, A., et al. (2018). Detection and resolution of rumours in
   social media: A survey. ACM Computing Surveys.
4. Rahman, M., et al. (2021). Text classification using TF-IDF and
   machine learning approaches for low-resource languages.
5. Zainab Noor. (2023). RUFND: Roman Urdu Fake News Dataset. Kaggle.

---

*This report accompanies the full project repository at:*  
*https://github.com/mrahmadtalha/urdu-fake-news-detection*