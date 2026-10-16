# ⚡ PreViral — Know Before You Post

> **Pre-publication social media virality predictor & AI optimizer.**  
> Paste your draft post → get a virality prediction → receive exact, data-backed suggestions to make it go viral — *before* you hit publish.

**Live Demo:** [previral-in-268278227902.asia-south1.run.app](https://previral-in-268278227902.asia-south1.run.app)  
**Author:** Anshuman Mishra · atulvatsamishra@gmail.com

---

## 📊 Model Performance (v6)

| Metric | Value |
|---|---|
| **Overall F1-Score** | **0.7775 ± 0.0021** |
| **Overall AUC** | **0.8646 ± 0.0003** |
| **Training Rows** | **183,276 posts** |
| **Class Balance** | Exactly 50/50 HIGH/LOW per platform |
| **Train / Val / Test Split** | 70% / 15% / 15% stratified |
| **Stability** | Validated across 5 random seeds |
| **Calibration** | Isotonic regression (well-calibrated probabilities) |

### Per-Platform Results

| Platform | F1-Score | AUC | Test Rows |
|---|---|---|---|
| YouTube | 0.8810 | 0.9306 | 6,054 |
| TikTok | 0.8126 | 0.8989 | 5,851 |
| Twitter/X | 0.7914 | 0.8814 | 5,991 |
| LinkedIn | 0.7663 | 0.8677 | 152 |
| Facebook | 0.7521 | 0.8728 | 212 |
| Instagram | 0.6839 | 0.7337 | 9,094 |

---

## 🗂️ Dataset

### Raw Data Sources

| Dataset | Platform | Raw Rows | Source |
|---|---|---|---|
| YouTube Trending (US, IN, GB, CA, DE, FR, JP, KR, MX, RU) | YouTube | ~417K | Kaggle `datasnaek/youtube-new` |
| Company Tweets | Twitter | ~4.3M | Kaggle |
| TikTok Viral Videos | TikTok | ~439 | Kaggle |
| Multi-platform Social Media Dataset | Mixed | ~2,000 | Kaggle |
| Instagram HuggingFace Dataset | Instagram | 60,000 | HuggingFace `vargr/main_instagram` |

### After Filtration & Stratified Sampling

| Platform | Training Rows | HIGH | LOW |
|---|---|---|---|
| Instagram | 40,000 | 20,000 | 20,000 |
| TikTok | 40,000 | 20,000 | 20,000 |
| Twitter | 40,000 | 20,000 | 20,000 |
| YouTube | 40,000 | 20,000 | 20,000 |
| Facebook | 1,428 | 714 | 714 |
| LinkedIn | 940 | 470 | 470 |
| **Total** | **163,276** | **81,638** | **81,638** |

> **v6 Upgrade:** An additional 60,000 Instagram posts from HuggingFace were merged in (replacing the old Instagram data), bringing the final v6 training set to **183,276 rows**. This improved Instagram F1 from 0.587 → 0.684 (+9.7%).

---

## 🧠 Model Architecture

### Classifier: LightGBM (Leaf-Wise Gradient Boosting)

| Hyperparameter | Value | Purpose |
|---|---|---|
| `n_estimators` | 3,000 (max) | Max trees; early stopping at 150 rounds |
| `num_leaves` | 127 | Equivalent to ~depth-7 tree complexity |
| `max_depth` | -1 (unlimited) | Leaf-wise growth, bounded by `num_leaves` |
| `min_child_samples` | 40 | Prevents splits with < 40 posts (key regularizer) |
| `reg_alpha` (L1) | 0.1 | Lasso — pushes weak features to zero |
| `reg_lambda` (L2) | 0.1 | Ridge — prevents any single feature dominating |
| `feature_fraction` | 0.8 | Random 80% of features per tree |
| `bagging_fraction` | 0.8 | Random 80% of rows every 5 iterations |
| `learning_rate` | 0.05 | Conservative learning for stability |
| Early Stopping | 150 rounds | Stops when val loss doesn't improve for 150 trees |

### 46-Feature Engineering Pipeline

Features computed live at inference time (< 12ms):

**Text / NLP Features**
- Flesch-Kincaid Readability Score
- VADER Sentiment (compound, positive, negative)
- Clickbait Score (heuristic keyword density)
- Word count, character count, emoji count
- Exclamation density, question mark density
- Caption-to-hashtag ratio

**Temporal Features**
- Hour-of-day (cyclical sin/cos encoding)
- Day-of-week (cyclical sin/cos encoding)
- Platform-specific peak timing score

**Visual Features (Image)**
- Face Prominence Score (MTCNN face detection)
- Dominant color (hue, saturation, brightness)
- Image text density
- Contrast and sharpness scores

**Structural Features**
- Hashtag count
- Platform one-hot encoding (6 platforms)
- Content type encoding (reel, video, tweet, etc.)

---

## 🔬 System Components

### 1. Causal Counterfactual Gap Engine
The core innovation. After classifying your post, the engine:
1. Computes the median values for each of the 46 features across **91,638 HIGH-virality** training posts
2. Calculates the exact mathematical gap between your post and the viral median for every feature
3. Ranks gaps by feature importance weight
4. Returns ranked, specific, actionable suggestions in natural language

> Example: *"Your face prominence is 1.0. The viral threshold is 2.3. Add a closer human face to your thumbnail."*

No commercial tool does this. Most give a score. PreViral tells you exactly what to fix and by how much.

### 2. AI Director (Gemini-Powered Optimizer)
Two modes:
- **DIAGNOSE:** Explains the virality score in plain language using your post's specific features
- **OPTIMIZER:** Rewrites your caption, hashtags, and timing — then **re-runs the rewrite back through LightGBM** to mathematically confirm the new version scores higher. Runs up to 2 validation iterations (Best-of-N).

### 3. LSTM Trajectory Forecaster
- Predicts a **10-day impression growth curve** for your post
- Platform-specific shapes: YouTube mega-creator curves, Instagram decay, Twitter spikes
- Realistic impression caps derived from percentile analysis of 183K real posts

### 4. Real-Time Trend Engine
- Queries Google Search Grounding to detect if your topic is currently trending
- Hard 5-second timeout — trend analysis never blocks the main prediction

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **ML Classifier** | LightGBM + Isotonic Calibration |
| **Trajectory** | LSTM (PyTorch) |
| **NLP** | VADER, MTCNN, custom heuristics |
| **AI Director** | Google Gemini (Flash / Pro) |
| **Backend** | Python, Flask |
| **Frontend** | Vanilla JS, HTML5, CSS3 |
| **Deployment** | Google Cloud Run (asia-south1) |
| **Containerization** | Docker |

---

## 🆚 How PreViral Differs From Competitors

| Feature | Hootsuite | Anyword | Dash Hudson | **PreViral** |
|---|---|---|---|---|
| Pre-publication prediction | ❌ | Partial (text only) | Partial (Instagram image only) | ✅ Full (text + image + timing + hashtags) |
| Counterfactual feature-level suggestions | ❌ | ❌ | ❌ | ✅ |
| Multi-platform support | ✅ (scheduling) | ❌ | ❌ | ✅ 6 platforms |
| AI rewrite validated by ML model | ❌ | ❌ | ❌ | ✅ Best-of-N loop |
| 10-day impression trajectory | ❌ | ❌ | ❌ | ✅ |
| Open-source | ❌ | ❌ | ❌ | ✅ |

> **One-line summary:** Hootsuite tells you your post flopped after it's published. PreViral tells you *why it will flop and exactly how to fix it* before you hit publish.

---

## 🚀 Running Locally

```bash
git clone https://github.com/anshumanvatsa/PreViral-AI-Social-Media-Predictor
cd PreViral-AI-Social-Media-Predictor
pip install -r requirements.txt
# Add your GEMINI_API_KEY to a .env file
python app.py
```

---

## 📁 Repository Structure

```
previral-gemini/
├── api/                  # Flask API routes
│   └── routes/
│       └── analyze.py    # Main prediction + AI Director endpoints
├── engines/
│   ├── feature_engine.py       # 46-feature extraction pipeline
│   ├── gemini_engine.py        # AI Director (Gemini integration)
│   ├── counterfactual_engine.py# Causal gap analysis
│   └── trajectory_engine.py   # LSTM trajectory forecaster
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── models/               # Trained LightGBM + LSTM model files
├── research/             # Training scripts, validation results
│   ├── retrain_v6_instagram.py
│   ├── v6_instagram_augmented_results.json
│   └── class_balance.json
├── Dockerfile
└── requirements.txt
```

---

## 📈 Ablation Study

Feature group contributions to overall F1 (based on `ablation_group_contributions.json`):

| Feature Group | F1 Contribution |
|---|---|
| Text / NLP | Highest |
| Temporal (timing) | High |
| Visual (image features) | Medium |
| Platform encoding | Medium |
| Structural (hashtags, length) | Supporting |

---

*PreViral — College Semester Project · Anshuman Mishra · 2026*
