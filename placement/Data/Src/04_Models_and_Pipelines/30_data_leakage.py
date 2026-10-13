# Data Leakage
# Leakage is any route by which information the model would NOT have at
# real prediction time reaches training, so the score you measure is
# better than the score you will actually get. It is not the same thing
# as splitting the data (Session 5): a correct train/validation/test
# split is the main DEFENCE, but several kinds of leakage survive a
# perfectly executed split. This session builds five, measures the
# inflation each one causes on this project's data, and audits the
# project's own splits for the ones that can be checked.
#
#   1. Target leakage      - a feature that encodes the answer
#   2. Preprocessing leakage - imputer/scaler fitted before splitting
#   3. Feature-selection leakage - choosing features using ALL the labels
#   4. Duplicate / overlap leakage - the same row on both sides of a split
#   5. Test-set reuse      - choosing a model by its score on the same data
#                            you then quote as its performance
#
# Two of these (2 and 5) turn out to be small on this dataset, and the
# script reports that plainly. Leakage is worth understanding because
# its size depends on the situation, not because every case is dramatic.
import sys, os
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_curr_dir) if os.path.basename(_curr_dir) == "04_Models_and_Pipelines" else os.path.dirname(os.path.dirname(_curr_dir))
_app_dir = os.path.join(_src_dir, "05_Web_Application")
for _p in [
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    _src_dir,
    _app_dir,
    _curr_dir
]:
    if _p and os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
import config

os.makedirs(config.LEAKAGE_DIR, exist_ok=True)
RS = config.RANDOM_STATE
TARGET = config.TARGET_COLUMN

raw = {n: pd.read_csv(f"{config.SPLITS_DIR}/{n}.csv") for n in ["train", "val", "test"]}
rdy = {n: pd.read_csv(f"{config.SPLITS_DIR}/{n}_ready.csv") for n in ["train", "val", "test"]}
for n in raw:
    assert (raw[n].StudentID.values == rdy[n].StudentID.values).all()
drop_cols = ["StudentID", "IsAnomaly", "Salary Package", TARGET]
clean_cols = [c for c in rdy["train"].columns if c not in drop_cols]
y = {n: rdy[n][TARGET].values for n in rdy}
summary = []   # (label, reported_score, honest_score)


def lgbm():
    return LGBMClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, random_state=RS, verbose=-1)


# =====================================================================
# AUDIT A: are the project's own splits clean?
# =====================================================================
print("=" * 70)
print("AUDIT A - Split integrity")
print("=" * 70)
ids = {n: set(raw[n].StudentID) for n in raw}
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    print(f"  StudentID overlap {a}/{b}: {len(ids[a] & ids[b])}")
feat_cols_raw = [c for c in raw["train"].columns if c not in ("StudentID",)]
sig = {n: pd.util.hash_pandas_object(raw[n][feat_cols_raw], index=False) for n in raw}   # one hash per row; NaN-safe
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    print(f"  Identical feature rows (ignoring StudentID) {a}/{b}: {len(set(sig[a]) & set(sig[b]))}")
print(f"  Identical rows within train: {int(sig['train'].duplicated().sum())}")
audit_a = {f"{a}/{b}": len(ids[a] & ids[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}

# =====================================================================
# AUDIT B: which single feature looks 'too good'?
# =====================================================================
print("\n" + "=" * 70)
print("AUDIT B - Single-feature AUC scan (a leakage detector)")
print("=" * 70)
audit_cols = [c for c in rdy["train"].columns if c not in ("StudentID", TARGET)]
single_auc = {}
for c in audit_cols:
    a = roc_auc_score(y["train"], rdy["train"][c])
    single_auc[c] = max(a, 1 - a)
top = sorted(single_auc.items(), key=lambda kv: kv[1], reverse=True)[:8]
for c, a in top:
    flag = "  <-- SUSPICIOUS (>0.95)" if a > 0.95 else ""
    print(f"  {c:<22} AUC {a:.4f}{flag}")
suspicious = [c for c, a in single_auc.items() if a > 0.95]
print(f"Features flagged: {suspicious}")
print("A real pre-placement feature rarely separates the outcome this cleanly on its own.")

# =====================================================================
# DEMO 1: target leakage
# =====================================================================
print("\n" + "=" * 70)
print("DEMO 1 - Target leakage: add 'Salary Package' as a feature")
print("=" * 70)
res1 = {}
for name, cols in [("clean features", clean_cols), ("+ Salary Package", clean_cols + ["Salary Package"])]:
    m = lgbm().fit(rdy["train"][cols], y["train"])
    p = m.predict_proba(rdy["val"][cols])[:, 1]
    res1[name] = (accuracy_score(y["val"], p >= 0.5), roc_auc_score(y["val"], p))
    print(f"  {name:<20} val accuracy {res1[name][0]:.4f}   val AUC {res1[name][1]:.4f}")
print("Salary Package exists only AFTER a student is placed (mean {:.2f} placed vs {:.2f} not placed),".format(
    raw["train"].loc[raw["train"][TARGET] == 1, "Salary Package"].mean(),
    raw["train"].loc[raw["train"][TARGET] == 0, "Salary Package"].mean()))
print("so at prediction time it would not be available. The 'great' score is an artefact.")
summary.append(("1. Target leakage\n(+ Salary Package)", res1["+ Salary Package"][0], res1["clean features"][0]))

# =====================================================================
# DEMO 2: preprocessing leakage
# =====================================================================
print("\n" + "=" * 70)
print("DEMO 2 - Preprocessing leakage: imputer + scaler fitted on ALL data vs TRAIN only")
print("=" * 70)
nc = config.NUMERIC_COLUMNS
res2 = {}
for name in ["fit on train only (correct)", "fit on train+val+test (leaky)"]:
    fit_df = raw["train"][nc] if "correct" in name else pd.concat([raw[n][nc] for n in raw])
    imp = SimpleImputer(strategy="median").fit(fit_df)
    sc = StandardScaler().fit(imp.transform(fit_df))
    tf = lambda d: sc.transform(imp.transform(d[nc]))
    m = LogisticRegression(max_iter=2000, random_state=RS).fit(tf(raw["train"]), y["train"])
    p = m.predict_proba(tf(raw["val"]))[:, 1]
    res2[name] = (accuracy_score(y["val"], p >= 0.5), roc_auc_score(y["val"], p))
    print(f"  {name:<32} val accuracy {res2[name][0]:.4f}   val AUC {res2[name][1]:.6f}")
d2 = res2["fit on train+val+test (leaky)"][0] - res2["fit on train only (correct)"][0]
print(f"Difference in accuracy: {d2:+.4f}")
print("With 35,000 training rows the mean/median/std barely move when validation rows are added,")
print("so the inflation is negligible HERE. The habit still matters: on small datasets, or with")
print("target-based encodings, the same mistake is large (see Demo 3).")
summary.append(("2. Preprocessing leakage\n(scaler fit on all data)", res2["fit on train+val+test (leaky)"][0],
                res2["fit on train only (correct)"][0]))

# =====================================================================
# DEMO 3: feature-selection leakage (pure noise)
# =====================================================================
print("\n" + "=" * 70)
print("DEMO 3 - Feature-selection leakage: 200 rows, 2000 pure-noise features, random labels")
print("=" * 70)
print("True accuracy is 50% by construction - there is nothing to learn.")
R3, N3, P3, K3 = 30, 200, 2000, 20
wrong, right = [], []
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
for r in range(R3):
    rng = np.random.RandomState(1000 + r)
    Xn, yn = rng.randn(N3, P3), rng.randint(0, 2, N3)
    Xsel = SelectKBest(f_classif, k=K3).fit_transform(Xn, yn)          # uses ALL labels, before CV
    wrong.append(cross_val_score(LogisticRegression(max_iter=1000), Xsel, yn, cv=cv).mean())
    pipe = Pipeline([("select", SelectKBest(f_classif, k=K3)), ("clf", LogisticRegression(max_iter=1000))])
    right.append(cross_val_score(pipe, Xn, yn, cv=cv).mean())         # selection refit inside each fold
wrong, right = np.array(wrong), np.array(right)
print(f"  Select-then-CV (leaky):         mean CV accuracy {wrong.mean():.4f}  (sd {wrong.std():.4f})")
print(f"  Selection inside CV (correct):  mean CV accuracy {right.mean():.4f}  (sd {right.std():.4f})")
print(f"  Inflation: {wrong.mean() - right.mean():+.4f}  -> the leaky pipeline 'discovers' signal in pure noise")
summary.append(("3. Feature-selection leakage\n(noise data, chance = 0.50)", wrong.mean(), right.mean()))

# =====================================================================
# DEMO 4: duplicate / overlap leakage
# =====================================================================
print("\n" + "=" * 70)
print("DEMO 4 - Duplicate leakage: training rows copied into the validation set")
print("=" * 70)
rng = np.random.RandomState(RS)
sub = rng.choice(len(rdy["train"]), 10000, replace=False)
Xtr_s, ytr_s = rdy["train"][clean_cols].iloc[sub], y["train"][sub]
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=RS, n_jobs=1).fit(Xtr_s, ytr_s)
Xva, yva = rdy["val"][clean_cols], y["val"]
clean_acc = accuracy_score(yva, rf.predict(Xva))
n_dup = int(0.30 * len(Xva) / 0.70)         # copies make up 30% of the leaky validation set
dup_idx = rng.choice(len(Xtr_s), n_dup, replace=False)
Xleaky = pd.concat([Xva, Xtr_s.iloc[dup_idx]])
yleaky = np.concatenate([yva, ytr_s[dup_idx]])
leaky_acc = accuracy_score(yleaky, rf.predict(Xleaky))
dup_only_acc = accuracy_score(ytr_s[dup_idx], rf.predict(Xtr_s.iloc[dup_idx]))
print(f"  Unconstrained random forest, clean validation set:            {clean_acc:.4f}")
print(f"  Same model, validation set with {n_dup:,} copied training rows:  {leaky_acc:.4f}")
print(f"  Accuracy on the copied rows alone (memorised):                 {dup_only_acc:.4f}")
print("A model that can memorise (deep trees, k-NN) is rewarded for the copies, not for generalising.")
summary.append(("4. Duplicate leakage\n(30% copies in val)", leaky_acc, clean_acc))

# =====================================================================
# DEMO 5: reusing the selection set as the performance estimate
# =====================================================================
print("\n" + "=" * 70)
print("DEMO 5 - Test-set reuse: pick the best of many models by validation score")
print("=" * 70)
print("Untouched test set is used only to check what the winner REALLY scores.")
N_ROUNDS, N_CFG = 8, 40
Xtr_all, ytr_all = rdy["train"][clean_cols], y["train"]
Xte, yte = rdy["test"][clean_cols], y["test"]
Xva, yva = rdy["val"][clean_cols], y["val"]
sel_sizes = [500, 5000]
opt = {s: [] for s in sel_sizes}          # selection effect: winner's gap minus the average gap of ALL models
opt_raw = {s: [] for s in sel_sizes}      # winner's raw (selection score - test score)
offsets = {s: [] for s in sel_sizes}      # average (selection score - test score) over ALL models
gap_rows = []
for rd in range(N_ROUNDS):
    rng = np.random.RandomState(500 + rd)
    tr_idx = rng.choice(len(Xtr_all), 2000, replace=False)
    val_sub = {500: rng.choice(len(Xva), 500, replace=False), 5000: np.arange(len(Xva))}
    val_pred, test_acc = [], []
    for _ in range(N_CFG):
        m = LGBMClassifier(n_estimators=int(rng.randint(20, 200)), learning_rate=float(10 ** rng.uniform(-2, -0.7)),
                           num_leaves=int(rng.randint(3, 48)), min_child_samples=int(rng.randint(5, 80)),
                           subsample=0.8, subsample_freq=1, random_state=int(rng.randint(1e6)), verbose=-1)
        m.fit(Xtr_all.iloc[tr_idx], ytr_all[tr_idx])
        val_pred.append(m.predict(Xva))
        test_acc.append(accuracy_score(yte, m.predict(Xte)))
    tests = np.array(test_acc)
    for s in sel_sizes:
        scores_v = np.array([accuracy_score(yva[val_sub[s]], vp[val_sub[s]]) for vp in val_pred])
        b = int(np.argmax(scores_v))
        raw_gap = scores_v[b] - tests[b]
        # The selection set can be systematically easier or harder than the test set for EVERY model
        # (pure sampling difference). The average gap over all models measures that offset; subtracting
        # it leaves the part caused by picking the winner.
        offset = (scores_v - tests).mean()
        opt_raw[s].append(raw_gap); offsets[s].append(offset); opt[s].append(raw_gap - offset)
        if s == 500:
            gap_rows.append((scores_v[b] - offset, tests[b]))
for s in sel_sizes:
    print(f"  Selection set of {s:>4} rows:")
    print(f"     winner's raw gap (selection score - test score):        {np.mean(opt_raw[s]):+.4f}")
    print(f"     average gap over ALL {N_CFG} models (set-difficulty offset): {np.mean(offsets[s]):+.4f}")
    print(f"     => selection effect (raw gap minus offset):             {np.mean(opt[s]):+.4f}  "
          f"(sd {np.std(opt[s]):.4f}, {N_ROUNDS} rounds)")
print("The offset is a fixed sampling difference between the validation and test sets - it affects every")
print("model equally, so it is NOT selection optimism and is removed. What remains is the optimism caused")
print("by picking the best of many models on the same data. It is larger on the smaller selection set.")
print("The fix is a separate, untouched test set (Session 40) or nested cross-validation (Session 32).")
sel500 = np.mean([g[0] for g in gap_rows])
tst500 = np.mean([g[1] for g in gap_rows])
summary.append(("5. Selection-set reuse\n(best of 40, 500-row set,\noffset-corrected)", sel500, tst500))

# =====================================================================
# Summary table + plots
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY - reported (leaky) score vs honest score")
print("=" * 70)
print(f"{'Case':<48}{'Leaky':>8}{'Honest':>8}{'Inflation':>11}")
for label, leaky, honest in summary:
    print(f"{label.replace(chr(10), ' '):<48}{leaky:>8.4f}{honest:>8.4f}{leaky - honest:>+11.4f}")

fig, ax = plt.subplots(figsize=(11, 6))
ypos = np.arange(len(summary))[::-1]
for yp, (label, leaky, honest) in zip(ypos, summary):
    ax.barh(yp + 0.2, honest, 0.38, color="#4C72B0", label="Honest score" if yp == ypos[0] else None)
    ax.barh(yp - 0.2, leaky, 0.38, color="#C0392B", label="Leaky (reported) score" if yp == ypos[0] else None)
    ax.text(max(leaky, honest) + 0.008, yp, f"{leaky - honest:+.4f}", va="center", fontsize=9, fontweight="bold")
ax.set_yticks(ypos)
ax.set_yticklabels([s[0] for s in summary], fontsize=9)
ax.set_xlim(0.4, 1.08)
ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Accuracy")
ax.set_title("Five Kinds of Leakage: How Much Each Inflates the Reported Score")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
plt.tight_layout()
plt.savefig(f"{config.LEAKAGE_DIR}/leakage_inflation_summary.png", dpi=130)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(right, bins=12, alpha=0.7, color="#4C72B0", label=f"Selection inside CV (mean {right.mean():.3f})")
axes[0].hist(wrong, bins=12, alpha=0.7, color="#C0392B", label=f"Select-then-CV (mean {wrong.mean():.3f})")
axes[0].axvline(0.5, color="black", linestyle="--", label="True accuracy (chance)")
axes[0].set_xlabel("Cross-validated accuracy on pure noise")
axes[0].set_ylabel("Count of 30 repeats")
axes[0].set_title("Feature-Selection Leakage Finds 'Signal' in Noise")
axes[0].legend(fontsize=8)
axes[1].bar([str(s) for s in sel_sizes], [np.mean(opt[s]) for s in sel_sizes],
            yerr=[np.std(opt[s]) for s in sel_sizes], color="#DD8452", capsize=5)
axes[1].set_xlabel("Number of rows in the set used to pick the winner")
axes[1].set_ylabel("Selection effect: winner's gap minus average gap of all models")
axes[1].set_title("Selection Optimism Shrinks as the Selection Set Grows")
plt.tight_layout()
plt.savefig(f"{config.LEAKAGE_DIR}/selection_and_noise_demos.png", dpi=130)
plt.close()
print("\nPlots saved")

with open(f"{config.LEAKAGE_DIR}/leakage_report.txt", "w") as f:
    f.write("Data leakage: five cases measured on this project's data\n\n")
    f.write("Split integrity audit (StudentID overlap): " + ", ".join(f"{k}: {v}" for k, v in audit_a.items()) + "\n")
    f.write(f"Identical feature rows within train: {int(sig['train'].duplicated().sum())}\n\n")
    f.write("Single-feature AUC scan, top 8 (flag > 0.95):\n")
    for c, a in top:
        f.write(f"  {c:<22} {a:.4f}{'  <-- SUSPICIOUS' if a > 0.95 else ''}\n")
    f.write(f"\n{'Case':<48}{'Leaky':>8}{'Honest':>8}{'Inflation':>11}\n")
    for label, leaky, honest in summary:
        f.write(f"{label.replace(chr(10), ' '):<48}{leaky:>8.4f}{honest:>8.4f}{leaky - honest:>+11.4f}\n")
    f.write(f"\nDemo 5 detail (winner of {N_CFG} models, {N_ROUNDS} rounds), by size of selection set:\n")
    for s in sel_sizes:
        f.write(f"  {s:>4} rows: winner raw gap {np.mean(opt_raw[s]):+.4f}, set-difficulty offset {np.mean(offsets[s]):+.4f}, "
                f"selection effect {np.mean(opt[s]):+.4f} (sd {np.std(opt[s]):.4f})\n")
    f.write("  The offset is how much easier the validation set is than the test set for EVERY model - a sampling\n")
    f.write("  difference, not selection - so it is subtracted. The summary row uses the offset-corrected figure.\n")
    f.write("\nWhat the numbers say:\n")
    f.write("  - Target leakage and feature-selection leakage are LARGE and can make a useless model look excellent.\n")
    f.write("  - Duplicate leakage is large for models that can memorise (deep trees, k-NN).\n")
    f.write(f"  - Preprocessing leakage was {d2:+.4f} here: negligible at n=35,000, but it is a habit worth keeping\n")
    f.write("    because it grows as the dataset shrinks.\n")
    f.write(f"  - Selection-set reuse: after removing the val-vs-test difficulty offset, the winner's selection score\n")
    f.write(f"    exceeds its true score by {np.mean(opt[500]):+.4f} on a 500-row set and {np.mean(opt[5000]):+.4f} on a 5000-row set.\n")
    f.write("\nPrevention checklist:\n")
    f.write("  1. Ask of every feature: 'Would I have this value at the moment of prediction?'\n")
    f.write("  2. Scan single-feature AUCs; investigate anything near 1.0 before celebrating.\n")
    f.write("  3. Fit imputers, scalers, encoders, selectors and clusterers on training data only -\n")
    f.write("     put them inside a Pipeline so cross-validation refits them per fold.\n")
    f.write("  4. Check split integrity: no shared IDs, no duplicated rows across splits.\n")
    f.write("  5. Touch the test set once. Use validation or cross-validation to choose models.\n")
    f.write("  6. Be suspicious of any score that looks too good - the fix is usually upstream of the model.\n")

print(f"Report saved to {config.LEAKAGE_DIR}/leakage_report.txt")
