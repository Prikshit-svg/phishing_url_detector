"""Reproducible training/evaluation pipeline for the URL phishing detector.

Usage: python train.py --data dataset.csv
Outputs: best_model.pkl, artifacts/metrics.json, artifacts/model_comparison.csv,
         and evaluation PNGs. No URL is visited and no network access is used.
"""
from __future__ import annotations
import argparse, json, os, pickle, warnings
from urllib.parse import urlsplit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve,
    auc)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

from features import extract_features, feature_names

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, 'artifacts')
LABEL_NAMES = ['Legitimate', 'Phishing']
RANDOM_STATE = 42


def _host_group(url):
    value = str(url).strip()
    candidate = value if '://' in value else '//' + value
    try:
        host = (urlsplit(candidate).hostname or '').lower().rstrip('.')
    except ValueError:
        host = ''
    # Conservative grouping heuristic; without a Public Suffix List this is not
    # a perfect registrable-domain parser (e.g. co.uk). Group by hostname when unsure.
    parts = host.split('.')
    if len(parts) >= 2:
        if len(parts[-1]) == 2 and len(parts) >= 3 and parts[-2] in {'co','com','org','gov','ac'}:
            return '.'.join(parts[-3:])
        return '.'.join(parts[-2:])
    return host or value.lower()


def load_and_validate(path):
    df = pd.read_csv(path)
    if not {'url', 'label'}.issubset(df.columns):
        raise ValueError("Dataset must contain 'url' and 'label' columns.")
    df = df[['url', 'label']].copy()
    df['url'] = df['url'].fillna('').astype(str).str.strip()
    df = df[df['url'].str.len() > 3]
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df[df['label'].isin([0, 1])].copy()
    df['label'] = df['label'].astype(int)
    before = len(df)
    df = df.drop_duplicates(subset=['url'], keep='first').reset_index(drop=True)
    dropped_dupes = before - len(df)
    if df['label'].nunique() != 2:
        raise ValueError('Both labels 0 (legitimate) and 1 (phishing) must be present.')
    # Existing engineered columns are deliberately ignored and regenerated from URL.
    feature_frame = pd.DataFrame([extract_features(u) for u in df['url']], columns=feature_names())
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, feature_frame, {'rows_after_cleaning': int(len(df)), 'duplicate_urls_removed': int(dropped_dupes),
        'class_counts': {str(int(k)): int(v) for k, v in df['label'].value_counts().sort_index().items()},
        'label_convention_assumed': {'0': 'legitimate', '1': 'phishing'},
        'label_source_verified': False,
        'warning': 'The pipeline cannot verify ground-truth labels; confirm the dataset source and label convention.'}


def make_models():
    models = {
        'Logistic Regression': Pipeline([('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=RANDOM_STATE))]),
        'Random Forest': RandomForestClassifier(n_estimators=90, max_depth=16, min_samples_leaf=2,
            class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=90, max_depth=18, min_samples_leaf=2,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=70, max_depth=2,
            learning_rate=0.08, random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(n_estimators=80, max_depth=5, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, eval_metric='logloss', random_state=RANDOM_STATE,
            tree_method='hist', max_bin=128, n_jobs=2, verbosity=0)
    except Exception as exc:
        warnings.warn(f'XGBoost unavailable; skipping it: {exc}')
    return models


def _savefig(fig, name):
    path = os.path.join(ARTIFACT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def train(data_path=None):
    data_path = data_path or os.path.join(BASE_DIR, 'dataset.csv')
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df, X, audit = load_and_validate(data_path)
    y = df['label'].reset_index(drop=True)
    groups = df['url'].map(_host_group)
    # Domain-grouped holdout reduces leakage from repeated URLs on related hosts.
    try:
        split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
        train_idx, test_idx = next(split.split(X, y, groups=groups))
        if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
            raise ValueError('group split did not preserve both classes')
        split_method = 'GroupShuffleSplit by approximate registrable domain (20% test)'
    except Exception:
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.20,
            random_state=RANDOM_STATE, stratify=y)
        split_method = 'Stratified random split fallback (domain grouping unavailable/degenerate)'
    X_train_all, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_all, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Validation split is drawn from training only; test data is not used for selection.
    train_rel, val_rel = train_test_split(np.arange(len(y_train_all)), test_size=0.20,
        random_state=RANDOM_STATE, stratify=y_train_all)
    X_fit, y_fit = X_train_all.iloc[train_rel], y_train_all.iloc[train_rel]
    X_val, y_val = X_train_all.iloc[val_rel], y_train_all.iloc[val_rel]
    # Cap fitting rows for a practical local/Colab run while preserving the group holdout.
    if len(X_fit) > 35000:
        fit_idx, _ = train_test_split(np.arange(len(y_fit)), train_size=35000, random_state=RANDOM_STATE, stratify=y_fit)
        X_fit, y_fit = X_fit.iloc[fit_idx], y_fit.iloc[fit_idx]

    models = make_models()
    fitted, rows = {}, []
    for name, model in models.items():
        print(f'Training {name} on {len(X_fit):,} rows...', flush=True)
        model.fit(X_fit, y_fit)
        val_pred = model.predict(X_val)
        val_prob = model.predict_proba(X_val)[:, 1]
        rows.append({'Model': name, 'Validation Accuracy': accuracy_score(y_val, val_pred),
            'Validation Precision': precision_score(y_val, val_pred, zero_division=0),
            'Validation Recall': recall_score(y_val, val_pred, zero_division=0),
            'Validation F1': f1_score(y_val, val_pred, zero_division=0),
            'Validation ROC-AUC': roc_auc_score(y_val, val_prob),
            'Validation Average Precision': average_precision_score(y_val, val_prob),
            'Validation Brier Score': brier_score_loss(y_val, val_prob)})
        fitted[name] = model
    comparison = pd.DataFrame(rows).sort_values('Validation F1', ascending=False)
    comparison.to_csv(os.path.join(ARTIFACT_DIR, 'model_comparison.csv'), index=False)
    # Model selection uses validation F1 only. Refit chosen model on all training rows.
    best_name = str(comparison.iloc[0]['Model'])
    best_validation_model = fitted[best_name]
    best_val_prob = best_validation_model.predict_proba(X_val)[:, 1]
    # Refit every candidate on the complete training partition before test-only plots.
    X_final, y_final = X_train_all, y_train_all
    if len(X_final) > 45000:
        final_idx, _ = train_test_split(np.arange(len(y_final)), train_size=45000, random_state=RANDOM_STATE, stratify=y_final)
        X_final, y_final = X_final.iloc[final_idx], y_final.iloc[final_idx]
    for candidate in models.values():
        candidate.fit(X_final, y_final)
    best_model = models[best_name]
    pred = best_model.predict(X_test)
    prob = best_model.predict_proba(X_test)[:, 1]
    test_metrics = {
        'accuracy': float(accuracy_score(y_test, pred)),
        'precision': float(precision_score(y_test, pred, zero_division=0)),
        'recall': float(recall_score(y_test, pred, zero_division=0)),
        'f1': float(f1_score(y_test, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, prob)),
        'average_precision': float(average_precision_score(y_test, prob)),
        'brier_score': float(brier_score_loss(y_test, prob)),
        'classification_report': classification_report(y_test, pred, target_names=LABEL_NAMES, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, pred, labels=[0,1]).tolist(),
        'test_rows': int(len(y_test)), 'test_phishing_prevalence': float(y_test.mean())}

    # Plot 1: model comparison metrics (validation scores).
    metric_cols = ['Validation Precision','Validation Recall','Validation F1','Validation ROC-AUC','Validation Average Precision']
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison.set_index('Model')[metric_cols].plot(kind='bar', ax=ax)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score'); ax.set_title('Validation metrics by model')
    ax.tick_params(axis='x', rotation=20); ax.grid(axis='y', alpha=.25); ax.legend(loc='lower right', fontsize=8)
    _savefig(fig, 'model_comparison.png')

    # Plot 2: ROC curves, test set only.
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in fitted.items():
        p = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test,p):.3f})')
    ax.plot([0,1],[0,1],'k--',label='Chance'); ax.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curves on held-out test set')
    ax.legend(loc='lower right', fontsize=8); ax.grid(alpha=.25); _savefig(fig, 'roc_curves.png')

    # Plot 3: precision-recall curves, test set only.
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in fitted.items():
        p = model.predict_proba(X_test)[:,1]
        precision, recall, _ = precision_recall_curve(y_test, p)
        ax.plot(recall, precision, label=f'{name} (AP={average_precision_score(y_test,p):.3f})')
    ax.axhline(y_test.mean(), color='k', linestyle='--', label='Test prevalence baseline')
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision–recall curves on held-out test set'); ax.legend(loc='best', fontsize=8); ax.grid(alpha=.25)
    _savefig(fig, 'precision_recall_curves.png')

    # Plot 4: confusion matrix.
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=[0,1]), display_labels=LABEL_NAMES).plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title(f'Best model: {best_name} — held-out test'); _savefig(fig, 'confusion_matrix.png')

    # Plot 5: calibration curve.
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy='quantile')
    pd.DataFrame({'mean_predicted_probability': mean_pred, 'observed_phishing_frequency': frac_pos}).to_csv(
        os.path.join(ARTIFACT_DIR, 'calibration_points.csv'), index=False)
    fig, ax = plt.subplots(figsize=(7, 6)); ax.plot(mean_pred, frac_pos, marker='o', label=f'{best_name} selected model')
    ax.plot([0,1],[0,1],'k--',label='Perfect calibration'); ax.set(xlabel='Mean predicted phishing probability', ylabel='Observed phishing frequency', title='Probability calibration (test set)')
    ax.legend(); ax.grid(alpha=.25); _savefig(fig, 'calibration_curve.png')

    # Plot 6: threshold trade-off; thresholds are descriptive, not chosen on test.
    thresholds = np.linspace(0.05, 0.95, 37)
    threshold_rows = []
    # Threshold analysis uses validation predictions only; the final test set is not used to tune a threshold.
    for threshold in thresholds:
        p = (best_val_prob >= threshold).astype(int)
        threshold_rows.append({'threshold':float(threshold),'precision':float(precision_score(y_val,p,zero_division=0)),
            'recall':float(recall_score(y_val,p,zero_division=0)),'f1':float(f1_score(y_val,p,zero_division=0))})
    pd.DataFrame(threshold_rows).to_csv(os.path.join(ARTIFACT_DIR, 'threshold_analysis.csv'), index=False)
    fig, ax = plt.subplots(figsize=(8, 5)); trdf = pd.DataFrame(threshold_rows)
    for key in ['precision','recall','f1']: ax.plot(trdf.threshold, trdf[key], label=key.title())
    ax.set(xlabel='Decision threshold', ylabel='Score', ylim=(0,1.02), title='Threshold trade-offs (validation set only)'); ax.legend(); ax.grid(alpha=.25)
    _savefig(fig, 'threshold_analysis.png')

    # Plot 7: permutation importance on a bounded held-out sample.
    sample_n = min(4000, len(X_test))
    sample_idx = resample(np.arange(len(X_test)), n_samples=sample_n, random_state=RANDOM_STATE, replace=False)
    imp = permutation_importance(best_model, X_test.iloc[sample_idx], y_test.iloc[sample_idx],
        scoring='f1', n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = pd.DataFrame({'feature':X.columns, 'importance_mean':imp.importances_mean,
        'importance_std':imp.importances_std}).sort_values('importance_mean', ascending=False)
    imp_df.to_csv(os.path.join(ARTIFACT_DIR, 'permutation_importance.csv'), index=False)
    top = imp_df.head(15).sort_values('importance_mean')
    fig, ax = plt.subplots(figsize=(9, 7)); ax.barh(top.feature, top.importance_mean, xerr=top.importance_std)
    ax.set(xlabel='Decrease in F1 when shuffled', title='Permutation importance (held-out sample)'); ax.grid(axis='x', alpha=.25)
    _savefig(fig, 'permutation_importance.png')

    # Plot 8: class distribution.
    fig, ax = plt.subplots(figsize=(6,4)); counts = y.value_counts().reindex([0,1], fill_value=0)
    ax.bar(LABEL_NAMES, counts.values); ax.set(ylabel='Rows', title='Dataset label distribution'); ax.grid(axis='y', alpha=.25)
    _savefig(fig, 'class_distribution.png')

    # Subgroup performance analysis on the held-out test partition. Categories overlap
    # and describe lexical patterns, not mutually exclusive or ground-truth risk groups.
    subgroup_masks = {
        'IP-host URLs': X_test['has_ip'].astype(bool),
        'Suspicious-TLD URLs': X_test['is_suspicious_tld'].astype(bool),
        'Long URLs (>=75 chars)': X_test['url_length'] >= 75,
        'Many subdomains (>=3)': X_test['num_subdomains'] >= 3,
        'Multiple suspicious words': X_test['num_suspicious_words'] >= 2,
        'High URL entropy (>=4)': X_test['url_entropy'] >= 4.0,
    }
    subgroup_rows = []
    for subgroup_name, mask in subgroup_masks.items():
        if int(mask.sum()) < 20 or y_test[mask].nunique() < 2:
            continue
        yp, yt = pred[mask.to_numpy()], y_test[mask].to_numpy()
        subgroup_rows.append({'category': subgroup_name, 'n': int(mask.sum()),
            'precision': float(precision_score(yt, yp, zero_division=0)),
            'recall': float(recall_score(yt, yp, zero_division=0)),
            'f1': float(f1_score(yt, yp, zero_division=0))})
    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(os.path.join(ARTIFACT_DIR, 'subgroup_performance.csv'), index=False)
    if not subgroup_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        plotted = subgroup_df.set_index('category')[['precision', 'recall', 'f1']]
        plotted.plot(kind='bar', ax=ax)
        ax.set_ylim(0, 1.05); ax.set_ylabel('Score'); ax.set_title('Held-out performance by overlapping URL category')
        ax.tick_params(axis='x', rotation=25); ax.grid(axis='y', alpha=.25); ax.legend(loc='lower right')
        for idx, n in enumerate(subgroup_df['n']):
            ax.text(idx, 0.02, f'n={n}', ha='center', va='bottom', fontsize=8, rotation=90)
        _savefig(fig, 'subgroup_performance.png')

    # Domain-batch retraining experiment: no timestamps exist, so batches are a
    # deterministic group-disjoint simulation and must not be described as temporal drift.
    # Evaluate static initial model and refit model on growing earlier batches.
    rng = np.random.RandomState(RANDOM_STATE)
    unique_groups = np.array(groups.unique(), dtype=object); rng.shuffle(unique_groups)
    chunks = [list(x) for x in np.array_split(unique_groups, 4) if len(x)]
    adaptive_rows = []
    seen_groups = set(chunks[0]) if chunks else set()
    initial_mask = groups.isin(seen_groups)
    if initial_mask.sum() and y[initial_mask].nunique() == 2 and len(chunks) > 1:
        adaptive_model = RandomForestClassifier(n_estimators=45, max_depth=14, min_samples_leaf=2,
            class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1)
        init_rows = np.flatnonzero(initial_mask)
        if len(init_rows) > 18000:
            init_rows, _ = train_test_split(init_rows, train_size=18000, random_state=RANDOM_STATE, stratify=y.iloc[init_rows])
        adaptive_model.fit(X.iloc[init_rows], y.iloc[init_rows])
        for batch_no, chunk in enumerate(chunks[1:], start=2):
            test_mask = groups.isin(chunk)
            if test_mask.sum() == 0 or y[test_mask].nunique() < 2: continue
            test_rows = np.flatnonzero(test_mask)
            if len(test_rows) > 6000:
                test_rows, _ = train_test_split(test_rows, train_size=6000, random_state=RANDOM_STATE, stratify=y.iloc[test_rows])
            before = adaptive_model.predict(X.iloc[test_rows])
            adaptive_rows.append({'batch':batch_no,'approach':'Static initial model','f1':f1_score(y.iloc[test_rows],before,zero_division=0), 'rows':int(len(test_rows))})
            # Refit only on previously observed groups, then evaluate on the current
            # unseen batch. The current batch is added to history only afterward.
            history_mask = groups.isin(seen_groups)
            history_rows = np.flatnonzero(history_mask)
            if len(history_rows) > 22000:
                history_rows, _ = train_test_split(history_rows, train_size=22000, random_state=RANDOM_STATE, stratify=y.iloc[history_rows])
            if len(history_rows) and y.iloc[history_rows].nunique() == 2:
                updated = RandomForestClassifier(n_estimators=45, max_depth=14, min_samples_leaf=2,
                    class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1)
                updated.fit(X.iloc[history_rows], y.iloc[history_rows])
                after = updated.predict(X.iloc[test_rows])
                adaptive_rows.append({'batch':batch_no,'approach':'Refit on prior batches','f1':f1_score(y.iloc[test_rows],after,zero_division=0), 'rows':int(len(test_rows))})
            seen_groups.update(chunk)
    adf = pd.DataFrame(adaptive_rows)
    adf.to_csv(os.path.join(ARTIFACT_DIR, 'adaptive_learning_experiment.csv'), index=False)
    if not adf.empty:
        fig, ax = plt.subplots(figsize=(8,5))
        for label, part in adf.groupby('approach'):
            ax.plot(part.batch, part.f1, marker='o', label=label)
        ax.set(xlabel='Deterministic domain batch', ylabel='F1', ylim=(0,1.02), title='Retraining simulation (not a temporal evaluation)')
        ax.legend(fontsize=8); ax.grid(alpha=.25); _savefig(fig, 'adaptive_learning.png')
    else:
        with open(os.path.join(ARTIFACT_DIR, 'adaptive_learning_NOTE.txt'),'w') as f:
            f.write('Adaptive-learning chart not generated: at least one training batch lacked both labels. No timestamp field exists, so temporal drift cannot be evaluated from this dataset.\n')

    artifact = {'model':best_model, 'model_name':best_name, 'candidate_models':models, 'feature_cols':list(X.columns),
        'threshold':0.5, 'label_mapping':{'0':'Legitimate','1':'Phishing'},
        'training_rows':int(len(X_final)), 'available_training_partition_rows':int(len(X_train_all)), 'test_rows':int(len(X_test)),
        'split_method':split_method, 'metrics':test_metrics,
        'notes':['Predictions are based on lexical URL features only.',
                 'Predicted probabilities are not guaranteed to be calibrated.',
                 'Dataset ground-truth labels and label convention were not independently verified.',
                 'Threshold analysis is generated from validation predictions; the default threshold is 0.5.']}
    with open(os.path.join(BASE_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(artifact, f)
    summary = {'audit':audit, 'split_method':split_method, 'selected_model':best_name,
        'feature_count':len(X.columns), 'features':list(X.columns), 'model_fit_rows':int(len(X_final)), 'validation_comparison':comparison.to_dict(orient='records'),
        'test_metrics':test_metrics, 'adaptive_learning_note':'Synthetic group batches only; no timestamps were available.',
        'warnings':artifact['notes']}
    with open(os.path.join(ARTIFACT_DIR, 'metrics.json'),'w') as f:
        json.dump(summary, f, indent=2)
    print('\nValidation comparison:\n', comparison.to_string(index=False))
    print('\nSelected model:', best_name)
    print('Held-out test metrics:', {k:round(v,4) for k,v in test_metrics.items() if isinstance(v,float)})
    print('Artifacts saved to:', ARTIFACT_DIR)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.join(BASE_DIR, 'dataset.csv'))
    args = parser.parse_args()
    train(args.data)
