import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
import joblib, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

FEAT_COLS  = "models/saved/feature_columns_v5.joblib"
OLD_CSV    = "d:/dg-social/scraper_pipeline/data_exports/previral_training_v5.csv"
NEW_CSV    = "data/instagram_raw/instagram_features.csv"
MERGED_CSV = "data/instagram_raw/training_v6_with_instagram.csv"
OUT        = "research"

feature_cols = list(joblib.load(FEAT_COLS))

print("="*60)
print("PreViral v6 — Instagram-Augmented Retraining")
print("="*60)

# ── Merge ──────────────────────────────────────────────────────
print("\n[1/3] Merging datasets...")
old = pd.read_csv(OLD_CSV)
new = pd.read_csv(NEW_CSV)
print(f"  Old training: {len(old):,} rows (existing)")
print(f"  New Instagram: {len(new):,} rows (fresh 60K)")

# Remove old instagram rows from old CSV (replace with better data)
if "platform" in old.columns:
    old_no_ig = old[old["platform"] != "instagram"].copy()
    print(f"  Old non-Instagram rows kept: {len(old_no_ig):,}")
else:
    old_no_ig = old.copy()

# Align columns
for c in feature_cols:
    if c not in new.columns: new[c] = 0.0
    if c not in old_no_ig.columns: old_no_ig[c] = 0.0

new["platform"] = "instagram"
merged = pd.concat([old_no_ig[feature_cols + ["label","platform"]],
                    new[feature_cols + ["label","platform"]]], ignore_index=True)
merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
merged.to_csv(MERGED_CSV, index=False)
print(f"  Merged: {len(merged):,} rows total")
print(f"  Platform breakdown:")
if "platform" in merged.columns:
    for p,n in merged["platform"].value_counts().items():
        print(f"    {p:<12} {n:>6,}")

# ── Train/Val/Test Split ──────────────────────────────────────
print("\n[2/3] 70/15/15 Split + 5-Seed Validation with Calibration...")
X = merged[feature_cols].values.astype("float32")
y = merged["label"].values.astype(int)
plat = merged["platform"].values

X_trainval, X_test, y_trainval, y_test, plat_trainval, plat_test = train_test_split(
    X, y, plat, test_size=0.15, random_state=42, stratify=y)
print(f"  Fixed test set: {len(y_test):,} rows (held out from all seeds)")
print(f"  Train+Val pool: {len(y_trainval):,} rows")

params = dict(
    objective="binary", metric="binary_logloss",
    n_estimators=3000, learning_rate=0.012,
    num_leaves=127, min_child_samples=40,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1,
    class_weight="balanced", n_jobs=-1, verbosity=-1)

seeds = [42, 123, 456, 789, 2024]
f1_scores=[]; auc_scores=[]; plat_results={}

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
print(f"\n  OVERALL: F1 = {mf1:.4f} +/- {sf1:.4f}  AUC= {mauc:.4f} +/- {sauc:.4f}")

print("\n  Per-platform F1 (mean +/- std, 5 seeds, held-out test):")
plat_summary={}
for p_name in sorted(plat_results.keys()):
    v = plat_results[p_name]
    pm=np.mean(v["f1"]); ps=np.std(v["f1"])
    am=np.mean(v["auc"]); as_=np.std(v["auc"])
    n_rows = int((plat_test==p_name).sum())
    prev_f1 = {"instagram":0.5868,"youtube":0.8945,"tiktok":0.8266,
               "twitter":0.7917,"facebook":0.7993,"linkedin":0.7888}.get(p_name, 0)
    delta = pm - prev_f1
    arrow = "^" if delta > 0.005 else ("v" if delta < -0.005 else "~")
    print(f"  {p_name:<12} F1={pm:.4f}+/-{ps:.4f}  AUC={am:.4f}  n={n_rows}  {arrow}{delta:+.4f} vs before")
    plat_summary[p_name]={"F1_mean":round(pm,4),"F1_std":round(ps,4),
                          "AUC_mean":round(am,4),"test_rows":n_rows,
                          "delta_vs_v5":round(delta,4)}

result={
    "version":"v6_instagram_augmented",
    "total_training_rows":int(len(merged)),
    "new_instagram_rows":60000,
    "method":"70/15/15 stratified split, 5 seeds, isotonic calibration",
    "F1_mean":round(mf1,4),"F1_std":round(sf1,4),
    "AUC_mean":round(mauc,4),"AUC_std":round(sauc,4),
    "per_seed_F1":[round(v,4) for v in f1_scores],
    "test_rows":int(len(y_test)),
    "per_platform":plat_summary
}
json.dump(result, open(f"{OUT}/v6_instagram_augmented_results.json","w"), indent=2)
print(f"\n[3/3] Saved: {OUT}/v6_instagram_augmented_results.json")
print("Done.")
