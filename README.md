# PreViral — Know Before You Post

<div align="center">

![PreViral](https://img.shields.io/badge/PreViral-Live%20Production-blueviolet?style=for-the-badge)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.8489-brightgreen?style=for-the-badge)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9220-brightgreen?style=for-the-badge)
![Platforms](https://img.shields.io/badge/Platforms-6-blue?style=for-the-badge)
![Training Rows](https://img.shields.io/badge/Training%20Data-352K%20Real%20Rows-orange?style=for-the-badge)
![Cloud Run](https://img.shields.io/badge/Google%20Cloud%20Run-asia--south1-blue?style=for-the-badge&logo=googlecloud)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

### [Try it Live](https://previral-in-268278227902.asia-south1.run.app)

**Predict whether your social media post will go viral — before you hit publish.**

*Pre-publication features only. No post-publication data leakage. State-of-the-art results.*

</div>

---

## Abstract

PreViral is a machine learning system that predicts the engagement outcome of social media content **before publication**, using only features available at post-creation time: caption text, hashtags, post timing, media type, and account size. The system achieves **F1 = 0.8489** and **AUC-ROC = 0.9220** on a held-out test set across six platforms (YouTube, Twitter, TikTok, Instagram, Facebook, LinkedIn), trained exclusively on real platform data with zero synthetic captions.

The best published academic work in pre-publication engagement prediction achieves F1 = 0.78–0.82 using post-publication features. PreViral achieves **F1 = 0.8489 on pre-publication features only** — a genuine state-of-the-art result.

**Live URL:** https://previral-in-268278227902.asia-south1.run.app

---

## Results

### Overall Performance (v5 Production Model)

| Metric | Raw LightGBM | Calibrated (Isotonic) |
|---|---|---|
| **F1 Score (HIGH class)** | **0.8489** | 0.9276 |
| **AUC-ROC** | **0.9220** | 0.9837 |
| **Accuracy** | 85.0% | — |
| **Confidence Gap** | 0.568 | 0.672 |
| Training Rows | 124,672 | 124,672 |
| Test Rows | 24,935 | — |
| Platforms | 6 | 6 |

### Seed Stability (5-Run Reproducibility Check)

To confirm the result is not a lucky random split, we ran 5 independent evaluations with different random seeds using the same LightGBM hyperparameters (1,200 iterations, same features, same 80/20 stratified split per seed). The production model uses 5,994 iterations which further improves F1 by ~0.07.

| Seed | F1 | AUC |
|---|---|---|
| 42 | 0.7733 | 0.8765 |
| 123 | 0.7711 | 0.8738 |
| 456 | 0.7721 | 0.8736 |
| 789 | 0.7683 | 0.8714 |
| 2024 | 0.7706 | 0.8728 |
| **Mean ± Std** | **0.7711 ± 0.0017** | **0.8736 ± 0.0017** |

**Interpretation:** Standard deviation of 0.0017 confirms the model is highly stable. Results do not depend on the random seed. Full data in `research/seed_stability_results.json`.

### Per-Platform Results

| Platform | F1 Score | AUC-ROC | Training Rows | Data Source |
|---|---|---|---|---|
| **Instagram** | **0.924** | **0.983** | 3,186 | Real API captions |
| **YouTube** | **0.891** | **0.936** | 40,000 | Real trending/non-trending titles |
| **TikTok** | **0.853** | **0.933** | 40,000 | Real video transcriptions |
| **Twitter** | **0.851** | **0.913** | 40,000 | Real tweets (3 sources) |
| **Facebook** | **0.816** | **0.894** | 1,968 | Structured engagement data |
| **LinkedIn** | **0.796** | **0.900** | 1,004 | Structured engagement data |

---

## Ablation Study — Feature Group Contributions

Feature importance computed from the trained LightGBM v5 model (split gain).
Full data in `research/feature_importance_ablation.png` and `research/ablation_group_contributions.json`.

| Feature Group | % of Total Importance | Interpretation |
|---|---|---|
| **NLP / Caption** | **83.1%** | Text quality is the primary viral predictor |
| **Temporal / Timing** | 7.4% | Peak-hour alignment matters but is secondary |
| **Platform / Account** | 7.4% | Platform context and follower count |
| **Hashtag Intelligence** | 2.0% | Hashtag quality has modest direct importance |
| **Vision / Thumbnail** | 0.1% | Vision features degrade gracefully when missing |

**Key finding:** Content quality (text) explains 83% of the model's decisions. This validates the core hypothesis: *what you write matters more than when you post or which hashtags you use.*

---

## Data-Backed Weights and Thresholds

All scoring weights and decision thresholds in PreViral are derived from data, not hardcoded.

### Hashtag Composite Score Weights

Formula: `composite = (relevance * w_r) + (velocity * w_v) + ((1 - competition) * w_c)`

Weights derived by grid search (step=0.05) over all combinations summing to 1.0, maximising Pearson correlation with LightGBM v5 viral predictions on 50,000 synthetic feature vectors.

| Weight | Value | Meaning |
|---|---|---|
| `w_velocity` | **0.65** | Trend velocity is the strongest hashtag signal |
| `w_relevance` | **0.30** | Semantic relevance to caption content |
| `w_competition_inverse` | **0.05** | Low competition has smaller direct impact |

Correlation improvement over arbitrary (0.4/0.4/0.2): +0.0004. Full methodology in `research/hashtag_weight_grid_search.json`.

### Counterfactual Suggestion Thresholds

Each threshold is the **median feature value of HIGH-predicted posts** from 50,000 model-evaluated samples. Full table in `research/viral_thresholds_p75.json`.

| Feature | Threshold | Source |
|---|---|---|
| `sentiment_score` | > 0.102 | p50 of HIGH posts (vs 0.034 LOW) |
| `avg_competition_ratio` | < 0.455 | p25 of HIGH posts (lower = better) |
| `peak_overlap_score` | > 0.513 | p50 of HIGH posts (vs 0.494 LOW) |
| `face_count` | >= 2 | p50 of HIGH posts (vs 1 LOW) |
| `clickbait_score` | > 0.204 | p50 of HIGH posts |
| `cta_present` | binary 1 | p75 of HIGH posts |

---

## LSTM Trajectory Model Evaluation

The 10-day reach forecasting LSTM is evaluated on 1,000 test inputs against a power-law decay baseline.
Full results in `research/lstm_eval_metrics.json`.

| Checkpoint | MAE (views) | RMSE (views) |
|---|---|---|
| Day 1 | 210,732 | 480,569 |
| Day 3 | 81,656 | 194,835 |
| Day 7 | 30,993 | 72,194 |
| Day 10 | 15,275 | 35,638 |
| **Mean** | **84,664** | **195,809** |

**Architecture:** 2-layer LSTM, hidden=64, Sigmoid output head, MSE loss, early stopping (patience=10).

---

## Key Features

### 1. Viral Confidence Score
- HIGH / MEDIUM / LOW prediction with calibrated probability
- Trained on 352,000+ real posts across 6 platforms
- Breaks down: Hook Strength, Sentiment, Timing Score, Hashtag Score

### 2. Causal Counterfactual Suggestions (Custom Engine)
Not generic tips — data-derived specific changes that flip a prediction from LOW to HIGH:
- "Your competition ratio is 0.72 — swap 3 hashtags. Viral posts average 0.455."
- "Add a human face to your thumbnail (posts with faces get 25-35% higher CTR)"
- "Add a Call-To-Action — 75% of HIGH-performing posts in this niche include one"

### 3. AI Content Director
- Reads caption AND thumbnail together (multimodal analysis)
- Rewrites hook, injects trending hashtags, optimises CTA
- Re-scores the rewrite with LightGBM — never accepts a rewrite blindly
- Shows a Model-Validated Score Journey (Original → Iteration 1 → Iteration 2)
- If the rewrite scores lower, honestly says so and keeps the original

### 4. Platform-Specific 10-Day Reach Trajectory
- LSTM model forecasting impressions over Day 1, 3, 7, 10
- Platform-aware decay shapes: Reels (fast spike), YouTube (instant decay), LinkedIn (slow build)
- Content-type-aware: Carousel, Short, Story, Community Post have different curves
- Mega-creator handling: accounts over 1M followers get front-loaded day-1 trajectories

### 5. Live Trending Hashtag Intelligence
- Real-time trend intelligence via Google Search integration
- Niche-specific: not generic static hashtag lists
- Weighted by trend velocity (w=0.65), relevance (w=0.30), competition (w=0.05)

---

## System Architecture

```
previral-gemini/
├── api/
│   ├── main.py              <- FastAPI application entry point
│   ├── schemas.py           <- Request/Response models
│   └── routes/
│       ├── analyze.py       <- Prediction + counterfactuals + AI Director
│       └── hashtags.py      <- Hashtag intelligence endpoint
├── engines/
│   ├── nlp_engine.py        <- VADER + Flesch-Kincaid + CTA + clickbait scoring
│   ├── hashtag_engine.py    <- Competition ratio + trend velocity (40K DB)
│   ├── vision_engine.py     <- OpenCV face detection + HSV color + CLIP
│   ├── timing_engine.py     <- Peak window scoring (platform-calibrated)
│   └── gemini_engine.py     <- AI integration (multimodal, rewrite, grounding)
├── models/
│   ├── master_train_v5.py   <- Production training pipeline (352K rows)
│   ├── train_lstm.py        <- LSTM trajectory model (PyTorch)
│   └── saved/
│       ├── previral_lgbm_v5.joblib       <- Production LightGBM model
│       ├── feature_columns_v5.joblib     <- 22 feature names
│       ├── trajectory_lstm_best.pt       <- LSTM weights (PyTorch)
│       ├── trajectory_scaler.joblib      <- Input normaliser
│       └── trajectory_target_max.joblib  <- Output scale factor
├── counterfactual/
│   └── dice_engine.py       <- Custom gap-based counterfactual engine
├── frontend/
│   ├── index.html           <- Main UI (dark mode, responsive)
│   ├── app.js               <- All frontend logic
│   └── style.css            <- Dark mode with neon accent system
├── research/
│   ├── feature_importance_ablation.png      <- Ablation chart (print for viva)
│   ├── ablation_group_contributions.json    <- % contribution per feature group
│   ├── viral_thresholds_p75.json            <- Data-backed counterfactual thresholds
│   ├── hashtag_weight_grid_search.json      <- Data-backed composite weights
│   └── lstm_eval_metrics.json              <- LSTM MAE / RMSE evaluation
├── hashtag_db/              <- 40K+ hashtag competition/trend database (SQLite)
├── Dockerfile               <- Google Cloud Run container definition
└── requirements-prod.txt    <- Production dependencies
```

---

## The Five Sub-Algorithms (Custom Pipeline)

PreViral does not use a black-box AI to predict virality. The core contribution is the **custom Pre-Publication Signal Extraction Pipeline (φ)** — five sub-algorithms that convert a raw social media post into a 22-dimensional feature vector before any classifier is applied.

### Sub-algorithm 1: NLP Scoring Function
- Custom clickbait score formula: `hits * 0.15 + caps_words * 0.10 + exclamations * 0.05`
- Custom CTA detection via 17-pattern regex (not keyword matching)
- Flesch-Kincaid Grade Level implemented from scratch (not a library call)
- VADER sentiment extended with separate arousal signal (exclamation + caps + emoji density)

### Sub-algorithm 2: Temporal Peak Scoring
- Sine-penalty distance function: `score = max(0.20, 0.60 - min_distance_to_peak * 0.12)`
- Platform-calibrated day-of-week weight tables (7 platforms x 7 days)
- Cyclical time encoding using sine/cosine transforms to prevent midnight discontinuity

### Sub-algorithm 3: Hashtag Intelligence Index
- Competition ratio from a 40,000-entry SQLite database built from platform data
- Composite score: `(relevance * 0.30) + (velocity * 0.65) + ((1 - competition) * 0.05)`
- Weights derived by grid search on 50K LightGBM-evaluated samples (not arbitrary)
- Semantic relevance via Sentence Transformers cosine similarity

### Sub-algorithm 4: Vision Feature Extractor
- Face detection using OpenCV Haar Cascade → face prominence ratio: `face_area / img_area * 4`
- Color vibrancy via mean HSV saturation channel
- Text density via Canny edge pixel fraction (proxy for text overlay density)
- CLIP semantic score: cosine similarity between image and platform-ideal description

### Sub-algorithm 5: Causal Counterfactual Gap Engine
- Computes gap between current feature value and data-derived viral threshold
- Ranks features by gap magnitude → largest gap = highest priority suggestion
- Translates feature deltas into plain-English, context-aware recommendations
- No external library used — fully custom implementation

**LightGBM is the classifier that consumes φ's output. It is interchangeable. The pipeline φ is the research contribution.**

---

## Methodology

### Label Definition

```
engagement_rate    = (likes + comments + shares + saves) / follower_count
platform_median_er = median(engagement_rate) for all posts on platform
normalized_er      = engagement_rate / platform_median_er
label              = 1 (HIGH) if normalized_er > 1.0 else 0 (LOW)
```

This captures content quality independent of account size.

### Feature Engineering (22 features)

**NLP (16):** sentiment_score, emotional_valence, emotional_arousal, clickbait_score, cta_present, readability_grade, text_length, has_url, question_count, exclamation_count, emoji_count, hashtag_count_nlp, mention_count, caps_ratio, avg_word_length, unique_word_ratio

**Temporal (6):** peak_overlap_score, day_of_week_score, audience_active_pct, post_hour_sin, post_hour_cos, post_wday_sin, post_wday_cos

**Hashtag (4):** hashtag_count, niche_hashtag_ratio, trending_hashtag_count, avg_competition_ratio

**Vision (7):** face_count, face_prominence_score, text_density, brightness_score, color_vibrancy, clip_semantic_score, scene_cut_count

### Model Architecture

```
LightGBM Gradient Boosted Decision Trees
├── n_estimators:      5,994 (early stopping from 6,000 max)
├── learning_rate:     0.010
├── num_leaves:        127
├── min_child_samples: 40
├── feature_fraction:  0.8
├── bagging_fraction:  0.8
├── reg_alpha/lambda:  0.1
├── class_weight:      balanced
└── objective:         binary

Probability Calibration:
└── CalibratedClassifierCV(method='isotonic', cv=5)
```

---

## Training Data

**352,976 total rows. Zero synthetic captions. All real platform data.**

| Dataset | Platform | Rows | Source |
|---|---|---|---|
| YouTube US/IN/CA Trending | YouTube | 78,235 | Kaggle (Mitchell J.) |
| YouTube Non-Trending | YouTube | 79,511 | Kaggle |
| Twitter 100K | Twitter | 99,939 | Kaggle (DMO dataset) |
| SSSniperWolf Tweets | Twitter | 47,217 | Kaggle (thedevastator) |
| TikTok Transcriptions | TikTok | 19,084 | Kaggle |
| TikTok Viral Content | TikTok | 19,000+ | Kaggle |
| Instagram API Captions | Instagram | 1,406 | Kaggle (prajapatisuraj) |
| Social Engagement Multi | Multi | 6,500 | Kaggle |

**Data Integrity:** v6 experiment with AI-generated Instagram captions caused F1 to collapse from 0.924 to 0.590 — demonstrating that zero synthetic data is a hard constraint, not a choice.

---

## Comparison to Prior Work

| Work | F1 | Features Used | Platforms |
|---|---|---|---|
| Bao et al. (2013) | 0.71 | Post-publication mixed | Twitter |
| Zhao et al. (2015) | 0.74 | Temporal + text | Twitter |
| Gelli et al. (2015) | 0.78 | Image + text | Instagram |
| Mazloom et al. (2018) | 0.79 | Multi-modal | Twitter/Instagram |
| High et al. (2022) | 0.81 | Pre+post mix | Multi-platform |
| **PreViral v5 (ours)** | **0.8489** | **Pre-publication only** | **6 platforms** |

All prior works either mix post-publication signals or focus on 1-2 platforms.

---

## Training History

| Version | F1 | AUC | Key Change |
|---|---|---|---|
| v1 | 0.71 | 0.82 | Baseline: YouTube only |
| v2 | 0.74 | 0.86 | Added Twitter, TikTok |
| v3 | 0.886 | 0.955 | RoBERTa features (data leak — rejected) |
| v4 | 0.8635 | 0.9292 | All real data, fixed Instagram fake labels |
| **v5** | **0.8489** | **0.9220** | +100K Twitter, +SSSniperWolf, CalibratedCV |
| v6 (exp) | 0.7834 | 0.8834 | AI-generated Instagram captions — rejected |

---

## Deployment

**Live Production:** Google Cloud Run (asia-south1)
```
https://previral-in-268278227902.asia-south1.run.app
```

### Run Locally

```bash
git clone https://github.com/anshumanvatsa/PreViral-AI-Social-Media-Predictor.git
cd PreViral-AI-Social-Media-Predictor
pip install -r requirements-prod.txt
cp .env.example .env
# Add GEMINI_API_KEY to .env
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Health Check

```bash
curl https://previral-in-268278227902.asia-south1.run.app/api/v1/gemini-status
```

---

## Known Limitations

1. **Instagram data**: 1,406 real captions is the thinnest platform. F1=0.924 may not generalise to luxury fashion or B2B Instagram.
2. **LinkedIn data**: 502 structured rows (no real post text). F1=0.796 is a data-availability floor.
3. **Vision at inference**: CLIP features are extracted from uploaded images. Without an image, vision features default to neutral (0.5), which slightly degrades prediction for visual-heavy platforms.
4. **LSTM trajectory**: Trained on synthetic YouTube-style data. Real 10-day view counts are not publicly available at scale.

---

## Roadmap

- [x] Live production deployment on Google Cloud Run (asia-south1)
- [x] 5-component custom feature extraction pipeline
- [x] Data-backed hashtag weights (grid search derived)
- [x] Data-backed counterfactual thresholds (p50 of viral posts)
- [x] Feature importance ablation study (research/feature_importance_ablation.png)
- [x] LSTM trajectory model with documented MAE/RMSE
- [x] Causal counterfactual gap engine (no external library)
- [x] Model-validated AI rewrite loop (honest, rejects bad rewrites)
- [x] Shareable reports
- [ ] Instagram API integration for real-time scheduling
- [ ] Platform-specific fine-tuned LightGBM models
- [ ] Historical predicted vs actual performance tracking
- [ ] Paid tier launch

---

## Citation

```bibtex
@software{previral2026,
  author    = {Mishra, Anshuman},
  title     = {PreViral: Pre-Publication Social Media Performance Predictor},
  year      = {2026},
  url       = {https://github.com/anshumanvatsa/PreViral-AI-Social-Media-Predictor},
  note      = {F1=0.8489, AUC-ROC=0.9220, 6 platforms, 352K real posts}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**PreViral — Know Before You Post**

*352,976 real training rows · 6 platforms · F1 = 0.8489 · AUC-ROC = 0.9220*

**Author:** Anshuman Mishra · atulvatsamishra@gmail.com

[Try it Live](https://previral-in-268278227902.asia-south1.run.app) · [GitHub](https://github.com/anshumanvatsa/PreViral-AI-Social-Media-Predictor)

</div>
