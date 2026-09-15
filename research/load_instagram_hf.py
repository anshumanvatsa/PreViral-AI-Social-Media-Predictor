import warnings; warnings.filterwarnings("ignore")
from datasets import load_dataset
import pandas as pd, os, json

OUT = "d:/dg-social/previral-gemini/data/instagram_raw"
os.makedirs(OUT, exist_ok=True)

results = {}

# Dataset 1: vargr/main_instagram - tabular with engagement data
print("\n[1/4] Loading vargr/main_instagram ...")
try:
    ds = load_dataset("vargr/main_instagram", split="train")
    df = ds.to_pandas()
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample:\n{df.head(2).to_string()}")
    df.to_csv(f"{OUT}/main_instagram.csv", index=False)
    results["main_instagram"] = {"rows": len(df), "cols": list(df.columns)}
except Exception as e:
    print(f"  FAILED: {e}")

# Dataset 2: prakhars/instagram_captions - captions text
print("\n[2/4] Loading prakhars/instagram_captions ...")
try:
    ds = load_dataset("prakhars/instagram_captions", split="train")
    df2 = ds.to_pandas()
    print(f"  Shape: {df2.shape}")
    print(f"  Columns: {list(df2.columns)}")
    print(f"  Sample:\n{df2.head(2).to_string()}")
    df2.to_csv(f"{OUT}/instagram_captions.csv", index=False)
    results["instagram_captions"] = {"rows": len(df2), "cols": list(df2.columns)}
except Exception as e:
    print(f"  FAILED: {e}")

# Dataset 3: AzrilFahmiardi captions config
print("\n[3/4] Loading AzrilFahmiardi/instagram_influencer_and_brand (captions) ...")
try:
    ds3 = load_dataset("AzrilFahmiardi/instagram_influencer_and_brand", "captions", split="train")
    df3 = ds3.to_pandas()
    print(f"  Shape: {df3.shape}")
    print(f"  Columns: {list(df3.columns)}")
    print(f"  Sample:\n{df3.head(2).to_string()}")
    df3.to_csv(f"{OUT}/influencer_captions.csv", index=False)
    results["influencer_captions"] = {"rows": len(df3), "cols": list(df3.columns)}
except Exception as e:
    print(f"  FAILED: {e}")

# Dataset 4: AzrilFahmiardi instagram_influencers
print("\n[4/4] Loading AzrilFahmiardi/instagram_influencer_and_brand (influencers) ...")
try:
    ds4 = load_dataset("AzrilFahmiardi/instagram_influencer_and_brand", "instagram_influencers", split="train")
    df4 = ds4.to_pandas()
    print(f"  Shape: {df4.shape}")
    print(f"  Columns: {list(df4.columns)}")
    print(f"  Sample:\n{df4.head(2).to_string()}")
    df4.to_csv(f"{OUT}/instagram_influencers.csv", index=False)
    results["instagram_influencers"] = {"rows": len(df4), "cols": list(df4.columns)}
except Exception as e:
    print(f"  FAILED: {e}")

print("\n\nSUMMARY:")
for k,v in results.items():
    print(f"  {k}: {v['rows']:,} rows | cols: {v['cols']}")
json.dump(results, open(f"{OUT}/dataset_probe.json","w"), indent=2)
