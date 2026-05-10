"""Tests unitaires pour le feature engineering."""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from anomaly_detection.features.windowing import generate_windows, assign_windows_to_df
from anomaly_detection.features.calculators import (
    compute_volume_features,
    compute_level_features,
    compute_http_features,
    compute_latency_features,
    compute_semantic_features,
    compute_temporal_features,
)


class TestWindowing(unittest.TestCase):
    def test_generate_basic(self):
        start = datetime(2026, 1, 1, 8, 0, 0)
        end = datetime(2026, 1, 1, 8, 10, 0)
        # window 5min, step 1min → on devrait avoir ~6 windows
        windows = generate_windows(start, end, 300, 60)
        self.assertGreaterEqual(len(windows), 5)
        self.assertEqual((windows[0][1] - windows[0][0]).total_seconds(), 300)
        self.assertEqual((windows[1][0] - windows[0][0]).total_seconds(), 60)

    def test_assign_windows_to_df(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-01-01 08:00:30",
                "2026-01-01 08:04:30",
                "2026-01-01 08:09:00",
            ], utc=True),
        })
        df_exp, windows = assign_windows_to_df(df, window_seconds=300, step_seconds=60)
        # 1er log dans 1 fenêtre, 2e dans 5 fenêtres, 3e dans 1 fenêtre
        self.assertGreater(len(df_exp), 3)
        self.assertIn("window_id", df_exp.columns)


class TestCalculators(unittest.TestCase):
    def _make_df(self, n=10, **overrides):
        base = {
            "timestamp": pd.to_datetime([f"2026-01-01 08:{i:02d}:00" for i in range(n)], utc=True),
            "programname": ["nova-api"] * n,
            "log_level": ["INFO"] * n,
            "severity": [1] * n,
            "log_kind": ["app_log"] * n,
            "groups": [["compute"]] * n,
            "http_method": [None] * n,
            "http_status": [None] * n,
            "http_status_class": ["none"] * n,
            "http_duration_ms": [None] * n,
            "endpoint_template": [None] * n,
            "payload_template_id": [0] * n,
            "request_id": [None] * n,
            "user_id": [None] * n,
            "tenant_id": [None] * n,
            "python_module": ["nova.api"] * n,
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_volume_features_empty(self):
        df = pd.DataFrame()
        feats = compute_volume_features(df, ["compute", "storage"])
        self.assertEqual(feats["count_total"], 0)

    def test_volume_features_basic(self):
        df = self._make_df(n=10)
        feats = compute_volume_features(df, ["compute", "storage"])
        self.assertEqual(feats["count_total"], 10)
        self.assertEqual(feats["count_group_compute"], 10)
        self.assertEqual(feats["count_group_storage"], 0)
        self.assertEqual(feats["unique_services"], 1)

    def test_level_features(self):
        df = self._make_df(n=10, severity=[1] * 8 + [3, 4])  # 8 INFO, 1 WARNING, 1 ERROR
        feats = compute_level_features(df)
        self.assertEqual(feats["count_severity_warning"], 1)
        self.assertEqual(feats["count_severity_error"], 1)
        self.assertAlmostEqual(feats["ratio_error"], 0.1)
        self.assertEqual(feats["max_severity"], 4)

    def test_http_features(self):
        n = 5
        df = self._make_df(
            n=n,
            log_kind=["http_access"] * n,
            http_method=["GET", "POST", "DELETE", "GET", "GET"],
            http_status=[200, 201, 404, 200, 500],
            http_status_class=["2xx", "2xx", "4xx", "2xx", "5xx"],
            http_duration_ms=[5.0, 200.0, 50.0, 10.0, 1500.0],
            endpoint_template=["GET /healthcheck", "POST /v3/auth/tokens",
                               "DELETE /v2.1/servers/<UUID>", "GET /healthcheck", "GET /"],
        )
        feats = compute_http_features(df)
        self.assertEqual(feats["count_http"], 5)
        self.assertEqual(feats["count_status_4xx"], 1)
        self.assertEqual(feats["count_status_5xx"], 1)
        self.assertEqual(feats["count_method_GET"], 3)
        self.assertEqual(feats["count_method_POST"], 1)
        self.assertEqual(feats["count_method_DELETE"], 1)
        # 2 modifying / 5 = 0.4
        self.assertAlmostEqual(feats["ratio_modifying"], 0.4)

    def test_latency_features(self):
        n = 4
        df = self._make_df(
            n=n,
            log_kind=["http_access"] * n,
            http_duration_ms=[1.0, 10.0, 100.0, 1500.0],
        )
        feats = compute_latency_features(df)
        self.assertEqual(feats["latency_max"], 1500.0)
        self.assertEqual(feats["count_latency_over_1s"], 1)

    def test_semantic_new_template(self):
        df = self._make_df(n=5, payload_template_id=[0, 0, 1, 2, 3])
        # On a déjà vu templates 0, 1
        feats = compute_semantic_features(df, known_template_ids={0, 1})
        # Templates 2, 3 sont nouveaux
        self.assertEqual(feats["new_templates_count"], 2)
        self.assertEqual(feats["unique_templates"], 4)

    def test_temporal_features(self):
        ts = pd.Timestamp("2026-04-27 14:30:00", tz="UTC")  # Lundi 14h30
        feats = compute_temporal_features(ts)
        self.assertEqual(feats["is_weekend"], 0)
        self.assertEqual(feats["is_business_hours"], 1)

        ts_sat = pd.Timestamp("2026-05-02 10:00:00", tz="UTC")  # Samedi
        feats_sat = compute_temporal_features(ts_sat)
        self.assertEqual(feats_sat["is_weekend"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
