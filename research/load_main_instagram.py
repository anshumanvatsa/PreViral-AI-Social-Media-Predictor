import warnings; warnings.filterwarnings("ignore")
from datasets import load_dataset
import pandas as pd, numpy as np, os, re, json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUT = "d:/dg-social/previral-gemini/data/instagram_raw"
os.makedirs(OUT, exist_ok=True)

print("Loading vargr/main_instagram (605K rows)...")
ds = load_dataset("vargr/main_instagram", split="train")
df = ds.to_pandas()
print(f"  Loaded: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Save raw with UTF-8
df.to_csv(f"{OUT}/main_instagram_raw.csv", index=False, encoding="utf-8-sig")
print(f"  Raw saved: {OUT}/main_instagram_raw.csv")

# ── CLEANING ──────────────────────────────────────────────────
print("\nCleaning...")

# 1. English only
if "lang" in df.columns:
    df_en = df[df["lang"] == "en"].copy()
    print(f"  English only: {len(df_en):,} rows (from {len(df):,})")
else:
    df_en = df.copy()

# 2. Drop nulls in critical columns
df_en = df_en.dropna(subset=["description", "likes", "followers"])
df_en = df_en[df_en["description"].str.strip().str.len() > 10]
df_en = df_en[df_en["followers"] > 100]
df_en = df_en[df_en["likes"] >= 0]
print(f"  After quality filter: {len(df_en):,} rows")

# 3. Engagement rate
df_en["engagement_rate"] = (df_en["likes"] + df_en["comments"].fillna(0)) / df_en["followers"].clip(lower=1)

# 4. Label using platform median (same methodology as rest of project)
median_er = df_en["engagement_rate"].median()
df_en["label"] = (df_en["engagement_rate"] > median_er).astype(int)
print(f"  Median ER: {median_er:.4f}")
print(f"  HIGH: {df_en['label'].sum():,} ({df_en['label'].mean()*100:.1f}%)")

# 5. Hashtag count from caption
df_en["hashtag_count_raw"] = df_en["description"].str.count(r"#\w+")
df_en["mention_count_raw"] = df_en["description"].str.count(r"@\w+")

# Save cleaned
df_en.to_csv(f"{OUT}/instagram_clean.csv", index=False, encoding="utf-8-sig")
print(f"\nCleaned saved: {len(df_en):,} rows -> {OUT}/instagram_clean.csv")

# Summary stats
print("\nSample rows:")
print(df_en[["description","likes","comments","followers","engagement_rate","label"]].head(3).to_string())

# Platform stats
print(f"\nPost type distribution:")
if "post_type" in df_en.columns:
    print(df_en["post_type"].value_counts().head(5).to_string())

print(f"\nFollower tier distribution:")
tiers = pd.cut(df_en["followers"], bins=[0,1e3,1e4,1e5,1e6,1e9],
               labels=["<1K","1K-10K","10K-100K","100K-1M","1M+"])
print(tiers.value_counts().sort_index().to_string())

print("\nDone. Ready for feature extraction.")
