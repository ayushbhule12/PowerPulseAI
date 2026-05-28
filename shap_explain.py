# src/shap_explain.py
"""
SHAP explainability for Random Forest and XGBoost.
Works in plain Python scripts (no IPython/Jupyter required).
"""

import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def explain_model(model,
                  X_train: pd.DataFrame,
                  X_test:  pd.DataFrame,
                  model_name:   str = 'XGBoost',
                  max_display:  int = 20,
                  sample_size:  int = 500):
    """
    Compute SHAP values and save summary + bar plots to data/.
    No IPython / Jupyter needed.
    """
    print(f"\n  Computing SHAP values for {model_name} ...")

    # Sample test set for speed
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)

    # Choose explainer
    if model_name in ('XGBoost', 'Random Forest'):
        explainer  = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)   # returns ndarray
    else:
        background  = shap.maskers.Independent(
                          X_train.sample(min(200, len(X_train)), random_state=42))
        explainer   = shap.Explainer(model, background)
        expl_obj    = explainer(X_sample)
        shap_values = expl_obj.values

    slug = model_name.lower().replace(" ", "_")

    # ── 1. Bar plot (mean |SHAP|) ─────────────────────────────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.DataFrame({
        'feature':   X_sample.columns,
        'mean_shap': mean_abs
    }).sort_values('mean_shap', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    top = feat_imp.head(max_display)
    ax.barh(top['feature'][::-1], top['mean_shap'][::-1], color='steelblue')
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'{model_name} — Mean |SHAP| Feature Importance')
    plt.tight_layout()
    plt.savefig(f'data/shap_bar_{slug}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → data/shap_bar_{slug}.png")

    # ── 2. Beeswarm / dot summary plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        max_display=max_display,
        show=False,
        plot_type='dot'
    )
    plt.title(f'{model_name} — SHAP Beeswarm Summary')
    plt.tight_layout()
    plt.savefig(f'data/shap_beeswarm_{slug}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → data/shap_beeswarm_{slug}.png")

    # ── 3. Save importance CSV ────────────────────────────────────────────────
    feat_imp.to_csv(f'data/shap_importance_{slug}.csv', index=False)
    print(f"  Saved → data/shap_importance_{slug}.csv")

    print(f"\n  Top 5 features ({model_name}):")
    print(feat_imp.head().to_string(index=False))

    return feat_imp, shap_values