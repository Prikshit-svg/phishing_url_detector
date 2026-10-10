"""Lightweight tests for shared features and generated model artifacts."""
import json
import os
import pickle
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from features import extract_features, feature_names, full_analysis
from train import load_and_validate

ROOT = Path(__file__).resolve().parents[1]


class FeatureTests(unittest.TestCase):
    def test_feature_schema_is_stable(self):
        features = extract_features('https://example.com/login')
        self.assertEqual(list(features), feature_names())
        self.assertEqual(len(features), 30)
        self.assertTrue(all(np.isfinite(v) for v in features.values()))

    def test_ip_and_port_are_detected(self):
        features = extract_features('http://192.0.2.1:8080/login')
        self.assertEqual(features['has_ip'], 1)
        self.assertEqual(features['has_port'], 1)

    def test_full_analysis_discloses_heuristics(self):
        result = full_analysis('https://example.com')
        self.assertIn('features', result)
        self.assertIn('note', result['dns'])
        self.assertIn('not checked', result['ssl']['verdict'].lower())

    def test_dataset_features_are_recomputed(self):
        path = ROOT / 'dataset.csv'
        df, X, audit = load_and_validate(path)
        self.assertEqual(len(df), len(X))
        self.assertEqual(list(X.columns), feature_names())
        self.assertEqual(X.shape[1], 30)
        self.assertEqual(set(df['label'].unique()), {0, 1})
        self.assertIn('label_source_verified', audit)


class ArtifactTests(unittest.TestCase):
    def test_model_artifact_predicts(self):
        path = ROOT / 'best_model.pkl'
        self.assertTrue(path.exists(), 'Run python train.py before tests')
        with path.open('rb') as handle:
            artifact = pickle.load(handle)
        self.assertIn('model', artifact)
        self.assertIn('candidate_models', artifact)
        self.assertGreaterEqual(len(artifact['candidate_models']), 2)
        self.assertEqual(len(artifact['feature_cols']), 30)
        frame = pd.DataFrame([extract_features('example.com/login')]).reindex(columns=artifact['feature_cols'])
        proba = artifact['model'].predict_proba(frame)
        self.assertEqual(proba.shape, (1, 2))
        self.assertAlmostEqual(float(proba.sum()), 1.0, places=5)

    def test_evaluation_artifacts_exist(self):
        path = ROOT / 'artifacts' / 'metrics.json'
        self.assertTrue(path.exists())
        report = json.loads(path.read_text(encoding='utf-8'))
        for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']:
            self.assertIn(key, report['test_metrics'])
        for name in ['model_comparison.png', 'roc_curves.png', 'precision_recall_curves.png',
                     'confusion_matrix.png', 'calibration_curve.png', 'threshold_analysis.png',
                     'permutation_importance.png', 'class_distribution.png',
                     'subgroup_performance.png']:
            self.assertTrue((ROOT / 'artifacts' / name).exists(), f'Missing {name}')
        self.assertTrue((ROOT / 'artifacts' / 'calibration_points.csv').exists())
        self.assertTrue((ROOT / 'artifacts' / 'subgroup_performance.csv').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
