import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
import joblib, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

CSV  = "d:/dg-social/scraper_pipeline/data_exports/previral_training_v5.csv"
OUT  = "research"
FEAT = "models/saved/feature_columns_v5.joblib"

print("="*60)
print("PreViral Research -- Complete Model Validation")
print("="*60)

df = pd.read_csv(CSV)
feature_cols = list(joblib.load(FEAT))
X = df[feature_cols].values.astype("float32")
y = df["label"].values.astype(int)
plat = df["platform"].values
print(f"Loaded {len(df):,} rows, {len(feature_cols)} features")

# GAP 4: Class Balance
print("\n[1/4] Class Balance Documentation...")
cb = {}
for p in sorted(df["platform"].unique()):
    sub = df[df["platform"]==p]
    hi = int(sub["label"].sum()); tot = len(sub)
    cb[p] = {"total_rows":tot,"HIGH":hi,"LOW":tot-hi,
              "HIGH_pct":round(hi/tot*100,1),"LOW_pct":round((tot-hi)/tot*100,1)}
    print(f"  {p:<12} total={tot:>6,}  HIGH={hi:>5,} ({hi/tot*100:.1f}%)  LOW={tot-hi:>5,}")
cb["overall"] = {"total_rows":len(df),"HIGH":int(y.sum()),"LOW":int((y==0).sum()),
    "HIGH_pct":round(y.mean()*100,1),
    "note":"50/50 balance enforced via stratified sampling per platform during training"}
json.dump(cb, open(f"{OUT}/class_balance.json","w"), indent=2)
print(f"  Saved: {OUT}/class_balance.json")

# GAP 1+2+3: 70/15/15 split + calibration + per-platform
print("\n[2/4] 70/15/15 Split + Calibration + Per-platform seed stability...")
X_trainval, X_test, y_trainval, y_test, plat_trainval, plat_test = train_test_split(
    X, y, plat, test_size=0.15, random_state=42, stratify=y)
print(f"  Fixed test set: {len(y_test):,} rows (held out from ALL seeds)")
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
print(f"\n  FINAL RESULT: F1 = {mf1:.4f} +/- {sf1:.4f}  (min={min(f1_scores):.4f}  max={max(f1_scores):.4f})")
print(f"               AUC= {mauc:.4f} +/- {sauc:.4f}")

print("\n  Per-platform F1 (mean +/- std, 5 seeds, calibrated, held-out):")
plat_summary={}
for p_name in sorted(plat_results.keys()):
    v = plat_results[p_name]
    pm=np.mean(v["f1"]); ps=np.std(v["f1"])
    am=np.mean(v["auc"]); as_=np.std(v["auc"])
    n_rows = int((plat_test==p_name).sum())
    print(f"  {p_name:<12} F1={pm:.4f}+/-{ps:.4f}  AUC={am:.4f}+/-{as_:.4f}  n={n_rows}")
    plat_summary[p_name]={"F1_mean":round(pm,4),"F1_std":round(ps,4),
                          "AUC_mean":round(am,4),"AUC_std":round(as_,4),"test_rows":n_rows}

result={
    "method":"70/15/15 stratified split. 15% test set held out from all seeds. Isotonic calibration applied (matches production). Same hyperparams as production except 3K iters for speed.",
    "seeds":seeds,
    "per_seed_F1":[round(v,4) for v in f1_scores],
    "per_seed_AUC":[round(v,4) for v in auc_scores],
    "F1_mean":round(mf1,4),"F1_std":round(sf1,4),
    "AUC_mean":round(mauc,4),"AUC_std":round(sauc,4),
    "F1_min":round(min(f1_scores),4),"F1_max":round(max(f1_scores),4),
    "test_rows":int(len(y_test)),
    "per_platform":plat_summary
}
json.dump(result, open(f"{OUT}/seed_stability_final.json","w"), indent=2)
print(f"\n  Saved: {OUT}/seed_stability_final.json")
print("\nAll gaps closed. Report F1 = mean+/-std from seed_stability_final.json")
