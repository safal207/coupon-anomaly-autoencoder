import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coupon_autoencoder.data import FEATURES, generate, read_features, validate_features
from coupon_autoencoder.model import Detector, calibrate, score_bundle


class DataTests(unittest.TestCase):
    def test_temporal_splits_and_labels_are_separate(self):
        d = generate(seed=5)
        train, val, test = [d.split == s for s in ["train", "validation", "test"]]
        self.assertLess(d.day[train].max(), d.day[val].min())
        self.assertLess(d.day[val].max(), d.day[test].min())
        self.assertFalse(d.labels[train | val].any())
        self.assertTrue(d.labels[test].any())
        self.assertEqual(d.x.shape[1], 12)
        self.assertNotIn("label", FEATURES)
        self.assertNotIn("employee_id", FEATURES)
        self.assertNotIn("scenario", FEATURES)
        # Aggregated event counts remain counts, including injected anomalies.
        count_columns = (0, 1, 2, 3, 9, 11)
        np.testing.assert_array_equal(d.x[:, count_columns], np.floor(d.x[:, count_columns]))

    def test_reproducible_generation_and_legitimate_busy_samples(self):
        a, b = generate(seed=10), generate(seed=10)
        np.testing.assert_array_equal(a.x, b.x)
        self.assertTrue(((a.scenario == "legitimate_busy") & (a.split == "test")).any())
        self.assertFalse(a.labels[a.scenario == "legitimate_busy"].any())

    def test_invalid_features_fail_before_scoring(self):
        for bad in [np.ones((0, 12)), np.ones((3, 11)), np.full((2, 12), np.nan), np.full((2, 12), -1)]:
            with self.assertRaises(ValueError):
                validate_features(bad)
        x = np.ones((2, 12)); x[0, 5] = 1.1
        with self.assertRaises(ValueError):
            validate_features(x)

    def test_csv_requires_feature_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            p.write_text("label,employee_id\n1,E001\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_features(p)

    def test_calibration_ties_do_not_exceed_alert_budget(self):
        scores = np.array([1.] * 95 + [2.] * 5)
        t = calibrate(scores, .01)
        self.assertLessEqual((scores > t).mean(), .01)
        for invalid in [[], [np.nan], [np.inf]]:
            with self.assertRaises(ValueError):
                calibrate(invalid)
        with self.assertRaises(ValueError):
            calibrate([1, 2], 1)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate(seed=19, employees=12, days=40)
        cls.train = cls.data.x[cls.data.split == "train"]
        cls.val = cls.data.x[cls.data.split == "validation"]
        cls.test = cls.data.x[cls.data.split == "test"]
        cls.detector = Detector(seed=19).fit(cls.train, cls.val)

    def test_bottleneck_and_training_loss_reduction(self):
        model = self.detector.autoencoder_
        self.assertEqual([w.shape for w in model.coefs_], [(12, 24), (24, 6), (6, 24), (24, 12)])
        self.assertLess(model.loss_curve_[-1], model.loss_curve_[0])

    def test_preprocessing_and_threshold_do_not_fit_on_holdout(self):
        from coupon_autoencoder.data import log_features
        expected = log_features(self.train).mean(axis=0)
        np.testing.assert_allclose(self.detector.scaler_.mean_, expected)
        before = self.detector.bundle()
        changed = self.test.copy(); changed[:, 0] *= 100
        self.detector.scores(changed)
        self.assertEqual(before, self.detector.bundle())
        validation_scores = self.detector.scores(self.val)["autoencoder"]
        self.assertEqual(self.detector.thresholds_["autoencoder"], calibrate(validation_scores))

    def test_portable_json_inference_matches_fitted_model(self):
        scores, flags = score_bundle(self.detector.bundle(), self.test)
        expected = self.detector.scores(self.test)["autoencoder"]
        np.testing.assert_allclose(scores, expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(flags, expected > self.detector.thresholds_["autoencoder"])

    def test_model_rejects_feature_order_and_corrupt_weights(self):
        bad = copy.deepcopy(self.detector.bundle())
        bad["features"] = list(reversed(FEATURES))
        with self.assertRaises(ValueError):
            score_bundle(bad, self.test)
        bad = copy.deepcopy(self.detector.bundle()); bad["weights"][0][0][0] = float("nan")
        with self.assertRaises(ValueError):
            score_bundle(bad, self.test)
        bad = copy.deepcopy(self.detector.bundle()); bad["scale"][0] = 0
        with self.assertRaises(ValueError):
            score_bundle(bad, self.test)


if __name__ == "__main__":
    unittest.main()
