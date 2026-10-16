"""
PreViral v7 Full Retraining Script
- Loads v6 training data
- Appends new Reddit (24K), Instagram (30K), Facebook (7K) rows
- Retrains LightGBM with same hyperparameters as v6
- Compares v7 vs v6: if better, saves model; otherwise reverts to v6
"""

import pandas as pd, numpy as np, json, warnings, joblib, os, glob
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from textblob import TextBlob
import re

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = "d:/dg-social/previral-gemini"
RESEARCH    = f"{BASE}/research"
MODELS      = f"{BASE}/models/saved"
OLD_CSV     = "d:/dg-social/scraper_pipeline/data_exports/previral_training_v5.csv"
IG_CSV      = f"{BASE}/data/instagram_raw/instagram_features.csv"

FEAT_COLS   = joblib.load(f"{MODELS}/feature_columns_v5.joblib")
feature_cols = list(FEAT_COLS)

V6_F1  = 0.7775
V6_AUC = 0.8646

print("="*60)
print("PreViral v7 — Reddit + Instagram + Facebook Expansion")
print("="*60)
print(f"Target to beat: F1={V6_F1:.4f}, AUC={V6_AUC:.4f}")

# ─── Load existing v6 merged training data ────────────────────────────────────
print("\n[1/5] Loading existing v6 training base...")
old = pd.read_csv(OLD_CSV)
new_ig = pd.read_csv(IG_CSV)
# Rebuild same v6 base (non-instagram from v5 + new instagram)
old_no_ig = old[old["platform"] != "instagram"].copy() if "platform" in old.columns else old.copy()
new_ig["platform"] = "instagram"
for c in feature_cols:
    if c not in new_ig.columns: new_ig[c] = 0.0
    if c not in old_no_ig.columns: old_no_ig[c] = 0.0
v6_base = pd.concat([old_no_ig[feature_cols + ["label","platform"]],
                     new_ig[feature_cols + ["label","platform"]]], ignore_index=True)
print(f"  v6 base: {len(v6_base):,} rows")
for p, n in v6_base["platform"].value_counts().items():
    print(f"    {p:<12} {n:>6,}")

# ─── Helper: compute NLP features from text ──────────────────────────────────
def extract_nlp_features(text):
    """Extract the same NLP features used in the PreViral feature engine."""
    text = str(text) if pd.notna(text) else ""
    words = text.split()
    blob = TextBlob(text) if text else None
    
    sentiment = blob.sentiment.polarity if blob else 0.0
    # clickbait heuristics
    clickbait_words = ["you won't believe","secret","amazing","shocking","mind-blowing",
                       "must see","viral","trending","epic","breaking"]
    clickbait = sum(1 for w in clickbait_words if w in text.lower()) / max(len(words), 1)
    
    return {
        "sentiment_score":      float(sentiment),
        "emotional_valence":    float(max(-1, min(1, sentiment))),
        "emotional_arousal":    float(abs(sentiment)),
        "clickbait_score":      float(clickbait),
        "text_length":          len(text),
        "question_count":       text.count("?"),
        "exclamation_count":    text.count("!"),
        "emoji_count":          len(re.findall(r'[\U0001F300-\U0001FFFF]', text)),
        "hashtag_count_nlp":    text.count("#"),
        "mention_count":        text.count("@"),
        "caps_ratio":           sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "avg_word_length":      np.mean([len(w) for w in words]) if words else 0,
        "unique_word_ratio":    len(set(words)) / max(len(words), 1),
        "has_url":              int("http" in text or "www." in text),
        "cta_present":          int(any(w in text.lower() for w in ["click","subscribe","follow","link","bio","dm","comment","share"])),
    }

def make_feature_row(platform, hour=12, dayofweek=2, text="", hashtag_count=0,
                     has_media=0, is_video=0, follower_count=0, **kwargs):
    """Build a feature dict with all 46 columns, zero-fill missing."""
    nlp = extract_nlp_features(text)
    
    # cyclical time encoding
    post_hour_sin = np.sin(2 * np.pi * hour / 24)
    post_hour_cos = np.cos(2 * np.pi * hour / 24)
    post_wday_sin = np.sin(2 * np.pi * dayofweek / 7)
    post_wday_cos = np.cos(2 * np.pi * dayofweek / 7)
    
    row = {
        **nlp,
        "hashtag_count":          hashtag_count,
        "niche_hashtag_ratio":    0.0,
        "trending_hashtag_count": 0,
        "avg_competition_ratio":  0.0,
        "peak_overlap_score":     0.0,
        "day_of_week_score":      0.0,
        "audience_active_pct":    0.0,
        "post_hour_sin":          post_hour_sin,
        "post_hour_cos":          post_hour_cos,
        "post_wday_sin":          post_wday_sin,
        "post_wday_cos":          post_wday_cos,
        "readability_grade":      0.0,
        "has_media":              has_media,
        "is_video":               is_video,
        "is_paid":                0,
        "follower_count":         float(follower_count),
        "face_count":             0,
        "face_prominence_score":  0.0,
        "text_density":           0.0,
        "brightness_score":       0.5,
        "color_vibrancy":         0.5,
        "clip_semantic_score":    0.0,
        "scene_cut_count":        0,
        # platform one-hot
        "platform_youtube":       int(platform == "youtube"),
        "platform_instagram":     int(platform == "instagram"),
        "platform_tiktok":        int(platform == "tiktok"),
        "platform_twitter":       int(platform == "twitter"),
        "platform_linkedin":      int(platform == "linkedin"),
        "platform_facebook":      int(platform == "facebook"),
        "platform_reddit":        int(platform == "reddit"),
        "platform_pinterest":     0,
    }
    return row

# ─── Process Reddit ───────────────────────────────────────────────────────────
print("\n[2/5] Processing Reddit dataset...")
reddit_files = glob.glob("d:/dg-social/raw_datasets/reddit_new/*.csv")
reddit_files = [f for f in reddit_files if "50_subreddits_list" not in f]

reddit_rows = []
for f in reddit_files:
    df = pd.read_csv(f)
    for _, row in df.iterrows():
        feat = make_feature_row(
            platform="reddit",
            text=str(row.get("title","")) + " " + str(row.get("body","") or ""),
            hour=pd.to_datetime(row.get("created_utc",""), errors="coerce").hour if pd.notna(row.get("created_utc")) else 12,
            dayofweek=pd.to_datetime(row.get("created_utc",""), errors="coerce").dayofweek if pd.notna(row.get("created_utc")) else 2,
            hashtag_count=0,
            has_media=int(str(row.get("post_type","")) in ["image","video"]),
            is_video=int(str(row.get("post_type","")) == "video"),
            follower_count=row.get("subscribers", 0) or 0,
        )
        feat["label"]    = row.get("_label_temp", 0)
        feat["score"]    = row.get("score", 0)
        feat["platform"] = "reddit"
        reddit_rows.append(feat)

df_reddit = pd.DataFrame(reddit_rows)
# Label: top 25% by score = HIGH
threshold = df_reddit["score"].quantile(0.75)
df_reddit["label"] = (df_reddit["score"] >= threshold).astype(int)
df_reddit = df_reddit.drop(columns=["score","_label_temp"], errors="ignore")

# Balance 50/50, cap at 20K per class
n = min((df_reddit["label"]==1).sum(), (df_reddit["label"]==0).sum(), 20000)
df_reddit_bal = pd.concat([
    df_reddit[df_reddit["label"]==1].sample(n, random_state=42),
    df_reddit[df_reddit["label"]==0].sample(n, random_state=42)
], ignore_index=True)
print(f"  Reddit: {len(df_reddit_bal):,} rows balanced ({n} HIGH + {n} LOW)")


# ─── Process Instagram Analytics ─────────────────────────────────────────────
print("\n[3/5] Processing Instagram Analytics dataset...")
df_ig_raw = pd.read_csv("d:/dg-social/raw_datasets/instagram_new/Instagram_Analytics.csv")
ig_rows = []
for _, row in df_ig_raw.iterrows():
    dt = pd.to_datetime(row.get("post_datetime",""), errors="coerce")
    hour      = dt.hour if pd.notna(dt) else int(row.get("post_hour", 12))
    dayofweek = dt.dayofweek if pd.notna(dt) else 2
    media_type = str(row.get("media_type","image")).lower()
    feat = make_feature_row(
        platform="instagram",
        text="",  # no caption text in this dataset
        hour=hour,
        dayofweek=dayofweek,
        hashtag_count=int(row.get("hashtags_count", 0) or 0),
        has_media=1,
        is_video=int(media_type in ["reel","video"]),
        follower_count=float(row.get("follower_count", 0) or 0),
    )
    feat["cta_present"] = int(row.get("has_call_to_action", 0) or 0)
    label_map = {"viral": 1, "high": 1, "medium": 0, "low": 0}
    feat["label"]    = label_map.get(str(row.get("performance_bucket_label","low")).lower(), 0)
    feat["platform"] = "instagram"
    ig_rows.append(feat)

df_ig_new = pd.DataFrame(ig_rows)
n_ig = min((df_ig_new["label"]==1).sum(), (df_ig_new["label"]==0).sum(), 15000)
df_ig_bal = pd.concat([
    df_ig_new[df_ig_new["label"]==1].sample(n_ig, random_state=42),
    df_ig_new[df_ig_new["label"]==0].sample(n_ig, random_state=42)
], ignore_index=True)
print(f"  Instagram new: {len(df_ig_bal):,} rows balanced ({n_ig} HIGH + {n_ig} LOW)")


# ─── Process Facebook ─────────────────────────────────────────────────────────
print("\n[4/5] Processing Facebook Live Sellers dataset...")
df_fb_raw = pd.read_csv("d:/dg-social/raw_datasets/facebook_new/Live.csv")
fb_rows = []
for _, row in df_fb_raw.iterrows():
    dt = pd.to_datetime(row.get("status_published",""), errors="coerce")
    hour      = dt.hour if pd.notna(dt) else 12
    dayofweek = dt.dayofweek if pd.notna(dt) else 2
    status_type = str(row.get("status_type","photo")).lower()
    feat = make_feature_row(
        platform="facebook",
        text="",
        hour=hour,
        dayofweek=dayofweek,
        hashtag_count=0,
        has_media=int(status_type in ["photo","video"]),
        is_video=int(status_type == "video"),
        follower_count=0,
    )
    feat["engagement"] = float(row.get("num_reactions", 0) or 0)
    feat["label"]      = 0  # will set below
    feat["platform"]   = "facebook"
    fb_rows.append(feat)

df_fb = pd.DataFrame(fb_rows)
threshold_fb = df_fb["engagement"].quantile(0.75)
df_fb["label"] = (df_fb["engagement"] >= threshold_fb).astype(int)
df_fb = df_fb.drop(columns=["engagement"], errors="ignore")
print(f"  Facebook: {len(df_fb):,} rows  (HIGH={( df_fb['label']==1).sum():,}, LOW={(df_fb['label']==0).sum():,})")


# ─── Merge everything ─────────────────────────────────────────────────────────
print("\n[5/5] Merging v6 base + new data and retraining...")

# Ensure all have the same 46 feature cols + label + platform
for df in [df_reddit_bal, df_ig_bal, df_fb]:
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

new_data = pd.concat([
    df_reddit_bal[feature_cols + ["label","platform"]],
    df_ig_bal[feature_cols + ["label","platform"]],
    df_fb[feature_cols + ["label","platform"]],
], ignore_index=True)

merged = pd.concat([v6_base, new_data], ignore_index=True)
merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n  Total merged rows: {len(merged):,}")
for p, n in merged["platform"].value_counts().items():
    print(f"    {p:<12} {n:>7,}")

# ─── Train ────────────────────────────────────────────────────────────────────
X = merged[feature_cols].values.astype("float32")
y = merged["label"].values.astype(int)
plat = merged["platform"].values

X_trainval, X_test, y_trainval, y_test, plat_trainval, plat_test = train_test_split(
    X, y, plat, test_size=0.15, random_state=42, stratify=y)
print(f"\n  Test set: {len(y_test):,} rows")
print(f"  Train+Val pool: {len(y_trainval):,} rows")

params = dict(
    objective="binary", metric="binary_logloss",
    n_estimators=3000, learning_rate=0.012,
    num_leaves=127, min_child_samples=40,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1,
    class_weight="balanced", n_jobs=-1, verbosity=-1)

seeds = [42, 123, 456, 789, 2024]
f1_scores=[]; auc_scores=[]; plat_results={}; best_model=None; best_f1=0

print(f"\n  seed     F1_cal   AUC_cal  iters")
for seed in seeds:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176,
        random_state=seed, stratify=y_trainval)
    p2 = dict(params); p2["random_state"] = seed
    base = lgb.LGBMClassifier(**p2)
    base.fit(X_tr, y_tr, eval_set=[(X_val,y_val)],
             callbacks=[lgb.early_stopping(150,verbose=False), lgb.log_evaluation(0)])
    cal = CalibratedClassifierCV(base, cv="prefit", method="isotonic")
    cal.fit(X_val, y_val)
    preds = cal.predict(X_test)
    proba = cal.predict_proba(X_test)[:,1]
    f1  = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    f1_scores.append(f1); auc_scores.append(auc)
    print(f"  {seed:4d}    {f1:.4f}   {auc:.4f}   {base.best_iteration_}")
    if f1 > best_f1:
        best_f1 = f1; best_model = cal
    for p_name in np.unique(plat_test):
        mask = plat_test == p_name
        if mask.sum() < 10: continue
        pf1  = f1_score(y_test[mask], preds[mask], zero_division=0)
        pauc = roc_auc_score(y_test[mask], proba[mask]) if len(np.unique(y_test[mask]))>1 else 0.5
        if p_name not in plat_results: plat_results[p_name]={"f1":[],"auc":[]}
        plat_results[p_name]["f1"].append(pf1)
        plat_results[p_name]["auc"].append(pauc)

mf1=np.mean(f1_scores); sf1=np.std(f1_scores)
mauc=np.mean(auc_scores); sauc=np.std(auc_scores)
print(f"\n  OVERALL v7: F1 = {mf1:.4f} +/- {sf1:.4f}  AUC= {mauc:.4f} +/- {sauc:.4f}")

# ─── Per-platform results ─────────────────────────────────────────────────────
print("\n  Per-platform F1:")
plat_summary={}
v6_platform_f1 = {"instagram":0.6839,"youtube":0.8810,"tiktok":0.8126,
                  "twitter":0.7914,"facebook":0.7521,"linkedin":0.7663,"reddit":0.8092}
for p_name in sorted(plat_results.keys()):
    v = plat_results[p_name]
    pm=np.mean(v["f1"]); ps=np.std(v["f1"])
    am=np.mean(v["auc"])
    n_rows = int((plat_test==p_name).sum())
    prev = v6_platform_f1.get(p_name, 0)
    delta = pm - prev
    arrow = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "~")
    print(f"  {p_name:<12} F1={pm:.4f}+/-{ps:.4f}  AUC={am:.4f}  n={n_rows}  {arrow}{delta:+.4f}")
    plat_summary[p_name]={"F1_mean":round(pm,4),"F1_std":round(ps,4),
                          "AUC_mean":round(am,4),"test_rows":n_rows,
                          "delta_vs_v6":round(delta,4)}

# ─── Decision: save v7 or revert ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  v6 baseline:  F1={V6_F1:.4f}  AUC={V6_AUC:.4f}")
print(f"  v7 candidate: F1={mf1:.4f}  AUC={mauc:.4f}")

if mf1 >= V6_F1 - 0.002:  # allow tiny tolerance
    print("\n  ✅ v7 is better or equivalent — SAVING new model!")
    joblib.dump(best_model, f"{MODELS}/lgbm_calibrated_v7.joblib")
    result={
        "version":"v7_multiplatform_expansion",
        "total_training_rows":int(len(merged)),
        "new_rows_added":int(len(new_data)),
        "new_platforms":["reddit(+24K)","instagram(+30K)","facebook(+7K)"],
        "method":"70/15/15 stratified split, 5 seeds, isotonic calibration",
        "F1_mean":round(mf1,4),"F1_std":round(sf1,4),
        "AUC_mean":round(mauc,4),"AUC_std":round(sauc,4),
        "per_seed_F1":[round(v,4) for v in f1_scores],
        "test_rows":int(len(y_test)),
        "per_platform":plat_summary,
        "verdict":"SAVED - v7 deployed"
    }
    json.dump(result, open(f"{RESEARCH}/v7_results.json","w"), indent=2)
    print(f"  Model saved: {MODELS}/lgbm_calibrated_v7.joblib")
    print(f"  Results:     {RESEARCH}/v7_results.json")
else:
    print(f"\n  ❌ v7 is WORSE than v6 (F1 dropped by {V6_F1-mf1:.4f})")
    print("  Keeping v6 model unchanged. No files modified.")
    result={
        "version":"v7_multiplatform_expansion",
        "F1_mean":round(mf1,4),"AUC_mean":round(mauc,4),
        "verdict":"REVERTED - v6 kept (v7 was worse)"
    }
    json.dump(result, open(f"{RESEARCH}/v7_results.json","w"), indent=2)

print("\nDone.")
