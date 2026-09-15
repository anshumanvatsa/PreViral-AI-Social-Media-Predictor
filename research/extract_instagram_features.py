import sys, os, warnings, re, math
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import pandas as pd, numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

CLEAN_CSV = "data/instagram_raw/instagram_clean.csv"
OUT_CSV   = "data/instagram_raw/instagram_features.csv"
FEAT_COLS = "models/saved/feature_columns_v5.joblib"

print("Loading cleaned Instagram data...")
df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
print(f"  {len(df):,} rows")

# Subsample to 60K (balanced 30K HIGH + 30K LOW) — plenty for retraining
hi = df[df["label"]==1].sample(30000, random_state=42)
lo = df[df["label"]==0].sample(30000, random_state=42)
df = pd.concat([hi, lo]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"  Sampled: {len(df):,} rows (30K HIGH + 30K LOW)")

vader = SentimentIntensityAnalyzer()
feature_cols = list(joblib.load(FEAT_COLS))
print(f"  Target features: {len(feature_cols)}")

# ── NLP feature extraction (mirrors nlp_engine.py) ──────────────────────────
CLICKBAIT = ["shocking","you won't believe","mind-blowing","insane","incredible",
             "secret","hack","never","always","amazing","this is why","watch this",
             "must see","gone wrong","challenge","exposed","truth about"]
CTA_PATTERNS = re.compile(r"\b(link in bio|swipe|comment|save this|follow|share|tag|dm|click|check out|shop now|grab yours|buy now|sign up)\b", re.I)

def flesch_grade(text):
    words = text.split()
    if len(words) < 5: return 0.5
    sentences = max(1, text.count(".")+text.count("!")+text.count("?"))
    syllables = sum(max(1, len(re.findall(r"[aeiouAEIOU]", w))) for w in words)
    try:
        fk = 0.39*(len(words)/sentences) + 11.8*(syllables/len(words)) - 15.59
        return min(1.0, max(0.0, fk/20.0))
    except: return 0.5

rows = []
BATCH = 5000
for i in range(0, len(df), BATCH):
    batch = df.iloc[i:i+BATCH]
    for _, row in batch.iterrows():
        text = str(row["description"])[:500]
        words = text.split()
        s = vader.polarity_scores(text)
        compound = s["compound"]
        excl = text.count("!")
        caps_words = sum(1 for w in words if w.isupper() and len(w)>1)
        emoji_count = len(re.findall(r"[\U0001F300-\U0001FAFF]", text, re.UNICODE))
        hashtags = re.findall(r"#\w+", text)
        mentions = re.findall(r"@\w+", text)
        clickbait_hits = sum(1 for cb in CLICKBAIT if cb in text.lower())
        has_cta = float(bool(CTA_PATTERNS.search(text)))
        followers = float(row["followers"])
        follower_norm = min(1.0, math.log10(max(1, followers)) / 8.0)

        # Post hour from date if available
        try:
            import datetime
            dt = pd.to_datetime(row["date"])
            hour = dt.hour; wday = dt.dayofweek
        except:
            hour = 12; wday = 2

        feat = {
            "sentiment_score":      compound,
            "emotional_valence":    max(0, compound),
            "emotional_arousal":    min(1, (excl*0.1 + caps_words*0.05 + emoji_count*0.05)),
            "clickbait_score":      min(1, clickbait_hits*0.15 + caps_words*0.1 + excl*0.05),
            "cta_present":          has_cta,
            "readability_grade":    flesch_grade(text),
            "text_length":          min(1, len(text)/500),
            "has_url":              float("http" in text or "bit.ly" in text),
            "question_count":       min(1, text.count("?")/5),
            "exclamation_count":    min(1, excl/5),
            "emoji_count":          min(1, emoji_count/10),
            "hashtag_count_nlp":    float(len(hashtags)),
            "mention_count":        float(len(mentions)),
            "caps_ratio":           min(1, caps_words/max(1,len(words))),
            "avg_word_length":      min(1, np.mean([len(w) for w in words]) / 10) if words else 0.5,
            "unique_word_ratio":    len(set(words))/max(1,len(words)),
            # Hashtag features
            "hashtag_count":        float(len(hashtags)),
            "niche_hashtag_ratio":  min(1, len([h for h in hashtags if len(h)<20])/max(1,len(hashtags))) if hashtags else 0.0,
            "trending_hashtag_count": min(5, float(len([h for h in hashtags if len(h.replace("#",""))<12]))),
            "avg_competition_ratio": 0.5,  # neutral default (no DB lookup in batch)
            # Timing
            "peak_overlap_score":   0.5,
            "day_of_week_score":    0.6 if wday in [1,2,3] else 0.4,
            "audience_active_pct":  0.4,
            "post_hour_sin":        math.sin(2*math.pi*hour/24),
            "post_hour_cos":        math.cos(2*math.pi*hour/24),
            "post_wday_sin":        math.sin(2*math.pi*wday/7),
            "post_wday_cos":        math.cos(2*math.pi*wday/7),
            # Vision (Instagram posts have images — set neutral)
            "face_count":           1.0,
            "face_prominence_score": 0.3,
            "text_density":         0.3,
            "brightness_score":     0.55,
            "color_vibrancy":       0.55,
            "clip_semantic_score":  0.5,
            "scene_cut_count":      0.0,
            # Account
            "follower_count":       follower_norm,
            "has_media":            1.0,
            "is_video":             float(str(row.get("post_type","")).lower() in ["video","reel"]),
            # Platform one-hot
            "platform_instagram":   1.0,
            "platform_twitter":     0.0,
            "platform_youtube":     0.0,
            "platform_tiktok":      0.0,
            "platform_facebook":    0.0,
            "platform_linkedin":    0.0,
            "platform_reddit":      0.0,
            # Label
            "label":                int(row["label"]),
            "platform":             "instagram",
        }
        rows.append(feat)

    if (i // BATCH) % 2 == 0:
        print(f"  Processed {min(i+BATCH,len(df)):,}/{len(df):,}")

print("Building DataFrame...")
out_df = pd.DataFrame(rows)
print(f"  Shape: {out_df.shape}")
print(f"  Label dist: {out_df['label'].value_counts().to_dict()}")

# Reorder to match feature_cols exactly
for c in feature_cols:
    if c not in out_df.columns:
        out_df[c] = 0.0
out_df[feature_cols + ["label","platform"]].to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print("Ready to merge with existing training data and retrain.")
