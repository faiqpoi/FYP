import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier
import joblib

np.random.seed(42)

# ── 1. GENERATE SYNTHETIC DATASET ─────────────────────────────────────────────
def generate_dataset(n=500):
    project_types   = ["Residential", "Commercial", "Infrastructure", "Industrial"]
    contract_types  = ["Lump Sum", "Design-Build", "Cost Plus", "Unit Price"]
    locations       = ["Selangor", "KL", "Johor", "Penang", "Sabah", "Sarawak"]
    funding_types   = ["Government", "Private", "PPP"]

    data = {
        "project_type":          np.random.choice(project_types, n),
        "contract_type":         np.random.choice(contract_types, n),
        "location":              np.random.choice(locations, n),
        "funding_type":          np.random.choice(funding_types, n),
        "planned_duration_days": np.random.randint(60, 1200, n),
        "project_size_m2":       np.random.randint(500, 50000, n),
        "num_floors":            np.random.randint(1, 40, n),
        "num_subcontractors":    np.random.randint(1, 20, n),
        "contractor_experience": np.random.randint(1, 30, n),   # years
        "planned_budget_RM":     np.random.randint(200000, 50000000, n),
        "budget_variance_pct":   np.round(np.random.uniform(-5, 40, n), 2),
        "payment_delay_days":    np.random.randint(0, 90, n),
        "approval_wait_days":    np.random.randint(1, 120, n),
        "material_lead_days":    np.random.randint(5, 90, n),
        "weather_risk_index":    np.round(np.random.uniform(0, 1, n), 2),
        "labour_supply_score":   np.round(np.random.uniform(0, 1, n), 2),
        "decision_making_speed": np.round(np.random.uniform(0, 1, n), 2),  # 1=fast
        "inspection_frequency":  np.random.randint(1, 20, n),
        "start_month":           np.random.randint(1, 13, n),
        "public_holidays_count": np.random.randint(0, 15, n),
        "past_delay_rate":       np.round(np.random.uniform(0, 1, n), 2),  # contractor history
    }

    df = pd.DataFrame(data)

    # ── Derived features ─────────────────────────────────────────
    df["budget_per_day"]     = df["planned_budget_RM"] / df["planned_duration_days"]
    df["size_duration_index"]= df["project_size_m2"] * df["planned_duration_days"] / 1e6
    df["subcontractor_ratio"]= df["num_subcontractors"] / df["planned_duration_days"].clip(lower=1)
    df["season_risk_flag"]   = df["start_month"].apply(
        lambda m: 1 if m in [11, 12, 1, 2] else 0   # northeast monsoon
    )
    df["complexity_score"]   = (
        df["num_floors"] * 0.3 +
        df["num_subcontractors"] * 0.4 +
        df["project_size_m2"] / 10000 * 0.3
    )

    # ── Target label: delay risk formula ─────────────────────────
    delay_score = (
        0.20 * (df["budget_variance_pct"] / 40) +
        0.15 * (df["payment_delay_days"] / 90) +
        0.15 * (df["approval_wait_days"] / 120) +
        0.10 * df["weather_risk_index"] +
        0.10 * df["past_delay_rate"] +
        0.10 * (1 - df["decision_making_speed"]) +
        0.08 * (df["material_lead_days"] / 90) +
        0.06 * (1 - df["labour_supply_score"]) +
        0.06 * df["season_risk_flag"]
    )
    noise = np.random.normal(0, 0.05, n)
    df["delayed"] = ((delay_score + noise) > 0.35).astype(int)

    return df


# ── 2. PREPROCESS ──────────────────────────────────────────────────────────────
def preprocess(df):
    cat_cols = ["project_type", "contract_type", "location", "funding_type"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df


# ── 3. TRAIN & EVALUATE ────────────────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc     = accuracy_score(y_test, y_pred)
        auc     = roc_auc_score(y_test, y_proba)
        cv      = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
        results[name] = {
            "model": model, "y_pred": y_pred, "y_proba": y_proba,
            "accuracy": acc, "auc": auc, "cv_accuracy": cv,
        }
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  Accuracy       : {acc:.4f}")
        print(f"  ROC-AUC        : {auc:.4f}")
        print(f"  CV Accuracy    : {cv:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Not Delayed','Delayed'])}")

    return results


# ── 4. PLOTS ───────────────────────────────────────────────────────────────────
def plot_results(results, X_test, y_test, feature_names):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Construction Project Delay Prediction — Model Evaluation",
                 fontsize=15, fontweight="bold", y=1.01)

    colors = {"Logistic Regression": "#534AB7",
               "Random Forest":       "#0F6E56",
               "XGBoost":             "#D85A30"}

    # ── ROC curves ───────────────────────────────────────────────
    ax = axes[0, 0]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})",
                color=colors[name], lw=2)
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_title("ROC Curves"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Accuracy comparison bar ───────────────────────────────────
    ax = axes[0, 1]
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    bars  = ax.bar(names, accs, color=[colors[n] for n in names], edgecolor="white")
    ax.set_ylim(0, 1); ax.set_title("Test Accuracy")
    ax.set_ylabel("Accuracy"); ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ── Confusion matrices ────────────────────────────────────────
    for i, (name, res) in enumerate(results.items()):
        ax = axes[1, i]
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Not Delayed","Delayed"],
                    yticklabels=["Not Delayed","Delayed"],
                    linewidths=0.5, cbar=False)
        ax.set_title(f"Confusion Matrix\n{name}", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # ── Feature importance (best model = RF or XGB) ───────────────
    ax = axes[0, 2]
    best_name = max(results, key=lambda n: results[n]["auc"])
    best_model = results[best_name]["model"]
    clf = best_model.named_steps["clf"] if hasattr(best_model, "named_steps") else best_model
    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=feature_names)
        top10 = importances.nlargest(10)
        top10.sort_values().plot(kind="barh", ax=ax, color=colors[best_name])
        ax.set_title(f"Top 10 Features\n({best_name})", fontsize=9)
        ax.set_xlabel("Importance"); ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/claude/model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Plot saved → model_evaluation.png")


# ── 5. PREDICT FUNCTION ────────────────────────────────────────────────────────
def predict_project(model, feature_names):
    """Demo single-project prediction."""
    sample = {
        "project_type": 0,           # Residential encoded
        "contract_type": 0,
        "location": 1,
        "funding_type": 1,
        "planned_duration_days": 365,
        "project_size_m2": 5000,
        "num_floors": 8,
        "num_subcontractors": 7,
        "contractor_experience": 10,
        "planned_budget_RM": 3000000,
        "budget_variance_pct": 15.0,
        "payment_delay_days": 30,
        "approval_wait_days": 45,
        "material_lead_days": 30,
        "weather_risk_index": 0.6,
        "labour_supply_score": 0.5,
        "decision_making_speed": 0.4,
        "inspection_frequency": 8,
        "start_month": 11,
        "public_holidays_count": 8,
        "past_delay_rate": 0.5,
        "budget_per_day": 3000000/365,
        "size_duration_index": 5000*365/1e6,
        "subcontractor_ratio": 7/365,
        "season_risk_flag": 1,
        "complexity_score": 8*0.3 + 7*0.4 + 5000/10000*0.3,
    }
    X_new = pd.DataFrame([sample])[feature_names]
    prob  = model.predict_proba(X_new)[0][1]
    label = model.predict(X_new)[0]
    print("\n" + "="*50)
    print("  SAMPLE PROJECT PREDICTION")
    print("="*50)
    print(f"  Delay risk probability : {prob*100:.1f}%")
    print(f"  Prediction             : {'DELAYED' if label==1 else 'NOT DELAYED'}")
    print("="*50)


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n>>> Generating dataset...")
    df = generate_dataset(500)
    df = preprocess(df)

    print(f"    Shape  : {df.shape}")
    print(f"    Delayed: {df['delayed'].sum()} / {len(df)} ({df['delayed'].mean()*100:.1f}%)")

    feature_cols = [c for c in df.columns if c != "delayed"]
    X = df[feature_cols]
    y = df["delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n>>> Training models...")
    results = train_models(X_train, X_test, y_train, y_test)

    print("\n>>> Generating evaluation plots...")
    plot_results(results, X_test, y_test, feature_cols)

    # Save best model by AUC
    best_name  = max(results, key=lambda n: results[n]["auc"])
    best_model = results[best_name]["model"]
    joblib.dump(best_model, "/home/claude/best_model.pkl")
    joblib.dump(feature_cols, "/home/claude/feature_cols.pkl")
    print(f"\n>>> Best model: {best_name} (AUC={results[best_name]['auc']:.4f})")
    print("    Saved → best_model.pkl")

    print("\n>>> Running sample prediction...")
    predict_project(best_model, feature_cols)

    # Summary table
    print("\n>>> Model Summary")
    print(f"{'Model':<25} {'Accuracy':>10} {'AUC':>10} {'CV Acc':>10}")
    print("-"*55)
    for name, res in results.items():
        marker = " <-- best" if name == best_name else ""
        print(f"{name:<25} {res['accuracy']:>10.4f} {res['auc']:>10.4f} {res['cv_accuracy']:>10.4f}{marker}")
