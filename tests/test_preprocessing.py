"""Tests for the BCSC two-stage preprocessing pipeline.

Run from the repository root:

    python -m unittest discover -s tests -v

The Test split is never loaded here. Validation data is used only for
transform-time behaviour (never for fitting).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_splits import FEATURES, TARGET, WEIGHT, load_split
from src.preprocessing.pipeline import (
    FAMILIES,
    NOMINAL_FEATURES,
    ORDINAL_CODES,
    ORDINAL_FEATURES,
    build_pipeline,
    encode_user_inputs,
    feature_names,
    fit_pipeline,
    load_pipeline,
    save_pipeline,
    transform_features,
)
from src.preprocessing.raw_input_mapping import (
    CHOICES,
    UNKNOWN_BCSC_CODE,
    UNKNOWN_INPUT,
    InputMappingError,
    RawInputMapper,
    human_to_interface_record,
    human_to_interface_frame,
    encode_human_inputs,
)

EXPECTED_ONEHOT_DIMS = 43
EXPECTED_TREE_DIMS = 26


def _inverse_choices(feature: str) -> dict[int, int]:
    return {code: ui for ui, code in CHOICES[feature].items()}


def _interface_frame(codes: pd.DataFrame) -> pd.DataFrame:
    """Convert BCSC codes to interface choices (inverse of the documented map)."""
    data = {}
    for feature in FEATURES:
        inverse = _inverse_choices(feature)
        mapped = codes[feature].map(inverse)
        if mapped.isna().any():
            raise AssertionError(f"test fixture contains unmappable {feature} codes")
        data[feature] = mapped.astype("int64")
    return pd.DataFrame(data, columns=FEATURES)


class TestRawInputMapping(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = RawInputMapper()

    def test_choices_are_documented_bcsc_categories(self) -> None:
        self.assertEqual(sorted(CHOICES), sorted(FEATURES))
        self.assertEqual(CHOICES["agegrp"], {i: i for i in range(1, 11)})
        self.assertEqual(CHOICES["bmi"], {1: 1, 2: 2, 3: 3, 4: 4, 0: 9})
        # agefirst: BCSC 0 = <30, 1 = >=30, 2 = nulliparous, 9 = unknown
        self.assertEqual(CHOICES["agefirst"], {1: 0, 2: 1, 3: 2, 0: 9})
        self.assertEqual(CHOICES["nrelbc"], {1: 0, 2: 1, 3: 2, 0: 9})
        self.assertEqual(CHOICES["density"], {1: 1, 2: 2, 3: 3, 4: 4, 0: 9})
        for feature in FEATURES:
            if feature != "agegrp":
                self.assertEqual(CHOICES[feature][UNKNOWN_INPUT], UNKNOWN_BCSC_CODE)
            # Interface codes are unique (the inverse used at inference is well defined).
            self.assertEqual(len(set(CHOICES[feature].values())), len(CHOICES[feature]))
        # agegrp has no Unknown: code 9 is a valid 75-79 band.
        self.assertNotIn(UNKNOWN_INPUT, CHOICES["agegrp"])

    def test_transform_full_record(self) -> None:
        record = {
            "agegrp": 5,
            "bmi": 2,
            "agefirst": 3,
            "menopaus": 2,
            "surgmeno": 1,
            "hrt": 2,
            "nrelbc": 3,
            "density": 4,
            "brstproc": 1,
            "lastmamm": 2,
        }
        self.assertEqual(
            self.mapper.transform(record),
            {
                "agegrp": 5,
                "bmi": 2,
                "agefirst": 2,
                "menopaus": 1,
                "surgmeno": 0,
                "hrt": 1,
                "nrelbc": 2,
                "density": 4,
                "brstproc": 0,
                "lastmamm": 1,
            },
        )
        self.assertEqual(list(self.mapper.transform(record)), FEATURES)

    def test_unknown_selection_maps_to_code_9(self) -> None:
        for feature in FEATURES:
            if feature == "agegrp":
                continue
            with self.subTest(feature=feature):
                record = {f: sorted(CHOICES[f])[0] for f in FEATURES}
                record[feature] = UNKNOWN_INPUT
                self.assertEqual(self.mapper.transform(record)[feature], 9)

    def test_invalid_inputs_are_rejected_not_coerced(self) -> None:
        base = {f: sorted(CHOICES[f])[0] for f in FEATURES}
        bad_values = [
            ("agegrp", 0),  # no Unknown for agegrp
            ("agegrp", 11),
            ("bmi", 5),
            ("agefirst", 9),  # interface 9 is not a valid choice
            ("menopaus", 3),
            ("surgmeno", 3),
            ("hrt", 3),
            ("nrelbc", 4),
            ("density", 10),
            ("brstproc", 3),
            ("lastmamm", 3),
            ("bmi", -1),
            ("bmi", 1.5),
            ("bmi", None),
            ("bmi", True),
            ("bmi", "not-a-number"),
        ]
        for feature, value in bad_values:
            with self.subTest(feature=feature, value=value):
                record = dict(base)
                record[feature] = value
                with self.assertRaises(InputMappingError):
                    self.mapper.transform(record)

    def test_record_requires_exact_fields(self) -> None:
        record = {f: sorted(CHOICES[f])[0] for f in FEATURES}
        with self.assertRaises(InputMappingError):
            self.mapper.transform({k: v for k, v in record.items() if k != "bmi"})
        extra = dict(record)
        extra["count"] = 1
        with self.assertRaises(InputMappingError):
            self.mapper.transform(extra)

    def test_frame_mapping_and_validation(self) -> None:
        codes = pd.DataFrame(
            [
                [1, 9, 0, 9, 0, 9, 0, 1, 0, 9],
                [10, 4, 2, 1, 1, 1, 2, 4, 1, 1],
            ],
            columns=FEATURES,
        )
        interface = _interface_frame(codes)
        out = self.mapper.transform_frame(interface)
        self.assertEqual(list(out.columns), FEATURES)
        pd.testing.assert_frame_equal(out, codes.astype("int64"))

        with self.assertRaises(InputMappingError):
            self.mapper.transform_frame(interface.drop(columns=["density"]))
        extra = interface.copy()
        extra[WEIGHT] = 1
        with self.assertRaises(InputMappingError):
            self.mapper.transform_frame(extra)

        invalid = interface.copy()
        invalid.loc[0, "agefirst"] = 9
        with self.assertRaises(InputMappingError):
            self.mapper.transform_frame(invalid)

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw_input_mapping.json"
            self.mapper.save(path)
            loaded = RawInputMapper.load(path)
            self.assertEqual(loaded.to_dict(), self.mapper.to_dict())
            record = {f: sorted(CHOICES[f])[0] for f in FEATURES}
            self.assertEqual(loaded.transform(record), self.mapper.transform(record))
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["features"], FEATURES)
            self.assertEqual(data["unknown_input"], UNKNOWN_INPUT)


class TestModelPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.X_train_full, _, _ = load_split("train")
        cls.X_val_full, _, _ = load_split("val")
        cls.X_train = cls.X_train_full.sample(20_000, random_state=0).reset_index(drop=True)
        cls.X_val = cls.X_val_full.sample(10_000, random_state=0).reset_index(drop=True)
        cls.pipelines = {
            family: fit_pipeline(cls.X_train, family=family) for family in FAMILIES
        }

    def test_feature_contract(self) -> None:
        self.assertEqual(len(FEATURES), 10)
        self.assertNotIn(TARGET, FEATURES)
        self.assertNotIn(WEIGHT, FEATURES)
        self.assertNotIn("training", FEATURES)
        for feature in ("invasive", "race", "Hispanic"):
            self.assertNotIn(feature, FEATURES)

        for bad in (
            self.X_train[FEATURES[::-1]],
            self.X_train.drop(columns=["bmi"]),
            self.X_train.assign(count=1),
            self.X_train.assign(cancer=0),
        ):
            with self.assertRaises(ValueError):
                fit_pipeline(bad, family="onehot")
            with self.assertRaises(ValueError):
                transform_features(self.pipelines["onehot"], bad)

    def test_feature_count_and_order(self) -> None:
        for family, expected in (("onehot", EXPECTED_ONEHOT_DIMS), ("ordinal_tree", EXPECTED_TREE_DIMS)):
            with self.subTest(family=family):
                pipe = self.pipelines[family]
                names = feature_names(pipe)
                Z = transform_features(pipe, self.X_train)
                self.assertEqual(Z.ndim, 2)
                self.assertEqual(Z.shape[1], expected)
                self.assertEqual(len(names), expected)
                self.assertEqual(len(set(names)), expected)
                self.assertTrue(np.isfinite(Z).all())
                # Every encoded column is traceable to exactly one approved feature.
                self.assertTrue(all(name.split("_")[0] in FEATURES for name in names))

        self.assertEqual(ORDINAL_FEATURES, ["agegrp", "bmi", "nrelbc", "density"])
        self.assertEqual(
            sorted(ORDINAL_FEATURES + NOMINAL_FEATURES), sorted(FEATURES)
        )
        self.assertEqual(ORDINAL_CODES["agegrp"], list(range(1, 11)))

    def test_stable_dimensions_across_splits(self) -> None:
        for family in FAMILIES:
            with self.subTest(family=family):
                pipe = self.pipelines[family]
                names = feature_names(pipe)
                Z_val = transform_features(pipe, self.X_val)
                self.assertEqual(Z_val.shape, (len(self.X_val), len(names)))
                self.assertEqual(feature_names(pipe), names)

    def test_ordinal_ranks_and_unknown_flags(self) -> None:
        pipe = self.pipelines["ordinal_tree"]
        names = feature_names(pipe)
        row = self.X_train.iloc[[0]].copy()
        row["agegrp"] = 10
        row["bmi"] = 9
        row["density"] = 1
        row["nrelbc"] = 2
        out = transform_features(pipe, row)[0]
        values = dict(zip(names, out.tolist()))
        self.assertEqual(values["agegrp_ordinal"], 10.0)
        self.assertNotIn("agegrp_unknown", values)  # agegrp 9 is valid, not Unknown
        self.assertEqual(values["bmi_ordinal"], 0.0)
        self.assertEqual(values["bmi_unknown"], 1.0)
        self.assertEqual(values["density_ordinal"], 1.0)
        self.assertEqual(values["density_unknown"], 0.0)
        self.assertEqual(values["nrelbc_ordinal"], 3.0)
        self.assertEqual(values["nrelbc_unknown"], 0.0)
        # agegrp code 9 is a valid 75-79 category, encoded as rank 9 (not Unknown).
        row["agegrp"] = 9
        values = dict(zip(names, transform_features(pipe, row)[0].tolist()))
        self.assertEqual(values["agegrp_ordinal"], 9.0)
        self.assertNotIn("agegrp_unknown", values)

    def test_fit_uses_training_categories_only_and_transform_never_refits(self) -> None:
        # Remove a training category (lastmamm=9) so validation holds a value
        # the encoder has never seen. Fitting must not learn it.
        train_subset = self.X_train[self.X_train["lastmamm"] != 9].reset_index(drop=True)
        pipe = fit_pipeline(train_subset, family="onehot")
        names = feature_names(pipe)
        self.assertNotIn("lastmamm_9", names)
        encoder = pipe.named_transformers_["categorical"]
        categories_before = [list(c) for c in encoder.categories_]

        val_unseen = self.X_val[self.X_val["lastmamm"] == 9].head(25)
        self.assertGreater(len(val_unseen), 0)
        Z = transform_features(pipe, val_unseen)
        lastmamm_cols = [i for i, name in enumerate(names) if name.startswith("lastmamm_")]
        self.assertTrue((Z[:, lastmamm_cols] == 0).all())

        # Transform must not add categories, even from unseen validation rows.
        categories_after = [list(c) for c in encoder.categories_]
        self.assertEqual(categories_before, categories_after)
        self.assertEqual(feature_names(pipe), names)

    def test_unknown_and_unseen_code_handling(self) -> None:
        pipe = self.pipelines["onehot"]
        names = feature_names(pipe)
        row = self.X_train.iloc[[0]].copy()
        for feature in FEATURES:
            if feature == "agegrp":
                row[feature] = 10
            else:
                row[feature] = 9
        values = dict(zip(names, transform_features(pipe, row)[0].tolist()))
        self.assertEqual(values["agegrp_10"], 1.0)
        for feature in NOMINAL_FEATURES:
            self.assertEqual(values[f"{feature}_9"], 1.0)

        # Documented Unknown is a real category; an undocumented nominal code is
        # ignored safely (all-zero block) rather than crashing at inference.
        unseen_nominal = self.X_train.iloc[[0]].copy()
        unseen_nominal["agefirst"] = 7
        Z = transform_features(pipe, unseen_nominal)
        agefirst_cols = [i for i, name in enumerate(names) if name.startswith("agefirst_")]
        self.assertTrue((Z[:, agefirst_cols] == 0).all())

        # Ordinal features have a fixed documented domain: undocumented codes are
        # data errors and must not be silently coerced to Unknown.
        ordinal_pipe = self.pipelines["ordinal_tree"]
        bad_ordinal = self.X_train.iloc[[0]].copy()
        bad_ordinal["agegrp"] = 11
        with self.assertRaises(ValueError):
            transform_features(ordinal_pipe, bad_ordinal)

    def test_save_reload_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for family in FAMILIES:
                with self.subTest(family=family):
                    pipe = self.pipelines[family]
                    path = Path(tmp) / f"{family}.joblib"
                    save_pipeline(pipe, path)
                    loaded = load_pipeline(path)
                    self.assertEqual(feature_names(loaded), feature_names(pipe))
                    np.testing.assert_allclose(
                        transform_features(loaded, self.X_val),
                        transform_features(pipe, self.X_val),
                        rtol=0,
                        atol=0,
                    )

    def test_interface_to_encoded_path_matches_source_path(self) -> None:
        codes = self.X_train.head(500).reset_index(drop=True)
        interface = _interface_frame(codes)
        mapper = RawInputMapper()
        for family in FAMILIES:
            with self.subTest(family=family):
                pipe = self.pipelines[family]
                np.testing.assert_allclose(
                    encode_user_inputs(mapper, pipe, interface),
                    transform_features(pipe, mapper.transform_frame(interface)),
                    rtol=0,
                    atol=0,
                )
                np.testing.assert_allclose(
                    encode_user_inputs(mapper, pipe, interface),
                    transform_features(pipe, codes),
                    rtol=0,
                    atol=0,
                )

    def test_build_pipeline_is_unfitted_and_rejects_unknown_family(self) -> None:
        fresh = build_pipeline("onehot")
        self.assertFalse(hasattr(fresh, "transformers_"))
        with self.assertRaises(ValueError):
            build_pipeline("magic")


class TestHumanInputMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.preprocessing.pipeline import load_pipeline

        cls.mapper = RawInputMapper()
        cls.onehot = load_pipeline("models/preprocessing/onehot_pipeline.joblib")
        cls.ordinal = load_pipeline("models/preprocessing/ordinal_tree_pipeline.joblib")

    def test_example_1_premenopausal(self) -> None:
        patient = {
            "age": 42,
            "bmi": 27.5,
            "has_given_birth": True,
            "age_at_first_birth": 26,
            "is_postmenopausal": False,
            "menopause_type": None,
            "currently_on_hrt": None,
            "first_degree_relatives_with_breast_cancer": 1,
            "breast_density": "heterogeneously dense",
            "prior_breast_procedure": False,
            "last_mammogram_result": "negative",
        }
        self.assertEqual(
            human_to_interface_record(patient),
            {
                "agegrp": 2,
                "bmi": 2,
                "agefirst": 1,
                "menopaus": 1,
                "surgmeno": 0,
                "hrt": 0,
                "nrelbc": 2,
                "density": 3,
                "brstproc": 1,
                "lastmamm": 1,
            },
        )

    def test_example_2_postmenopausal(self) -> None:
        patient = {
            "age": 61,
            "bmi": 32.1,
            "has_given_birth": True,
            "age_at_first_birth": 31,
            "is_postmenopausal": True,
            "menopause_type": "natural",
            "currently_on_hrt": False,
            "first_degree_relatives_with_breast_cancer": "2+",
            "breast_density": "extremely dense",
            "prior_breast_procedure": True,
            "last_mammogram_result": "false positive",
        }
        self.assertEqual(
            human_to_interface_record(patient),
            {
                "agegrp": 6,
                "bmi": 3,
                "agefirst": 2,
                "menopaus": 2,
                "surgmeno": 1,
                "hrt": 1,
                "nrelbc": 3,
                "density": 4,
                "brstproc": 2,
                "lastmamm": 2,
            },
        )

    def test_example_3_uncertain(self) -> None:
        patient = {
            "age": 53,
            "bmi": 23,
            "has_given_birth": False,
            "age_at_first_birth": None,
            "is_postmenopausal": None,
            "menopause_type": None,
            "currently_on_hrt": None,
            "first_degree_relatives_with_breast_cancer": 0,
            "breast_density": None,
            "prior_breast_procedure": False,
            "last_mammogram_result": "negative",
        }
        self.assertEqual(
            human_to_interface_record(patient),
            {
                "agegrp": 4,
                "bmi": 1,
                "agefirst": 3,
                "menopaus": 0,
                "surgmeno": 0,
                "hrt": 0,
                "nrelbc": 1,
                "density": 0,
                "brstproc": 1,
                "lastmamm": 1,
            },
        )

    def test_age_boundaries(self) -> None:
        base = {
            "age": 42,
            "bmi": 23.0,
            "has_given_birth": False,
            "age_at_first_birth": None,
            "is_postmenopausal": None,
            "menopause_type": None,
            "currently_on_hrt": None,
            "first_degree_relatives_with_breast_cancer": 0,
            "breast_density": None,
            "prior_breast_procedure": False,
            "last_mammogram_result": "negative",
        }
        for age, expected in ((35, 1), (39, 1), (40, 2), (84, 10)):
            patient = dict(base)
            patient["age"] = age
            self.assertEqual(human_to_interface_record(patient)["agegrp"], expected)
        for bad_age in (34, 85, True, "42"):
            patient = dict(base)
            patient["age"] = bad_age
            with self.assertRaises(InputMappingError):
                human_to_interface_record(patient)

    def test_bmi_boundaries_and_unknown(self) -> None:
        base = {
            "age": 42,
            "bmi": 23.0,
            "has_given_birth": False,
            "age_at_first_birth": None,
            "is_postmenopausal": None,
            "menopause_type": None,
            "currently_on_hrt": None,
            "first_degree_relatives_with_breast_cancer": 0,
            "breast_density": None,
            "prior_breast_procedure": False,
            "last_mammogram_result": "negative",
        }
        cases = ((10, 1), (24.99, 1), (25, 2), (29.99, 2), (30, 3), (34.99, 3), (35, 4), (40, 4))
        for bmi, expected in cases:
            patient = dict(base)
            patient["bmi"] = bmi
            self.assertEqual(human_to_interface_record(patient)["bmi"], expected)
        patient = dict(base)
        patient["bmi"] = None
        self.assertEqual(human_to_interface_record(patient)["bmi"], 0)
        for bad_bmi in (9.99, True, "23", float("nan")):
            patient = dict(base)
            patient["bmi"] = bad_bmi
            with self.assertRaises(InputMappingError):
                human_to_interface_record(patient)

    def test_nulliparous_and_conflict(self) -> None:
        patient = {
            "age": 42,
            "bmi": 23.0,
            "has_given_birth": False,
            "age_at_first_birth": None,
            "is_postmenopausal": None,
            "menopause_type": None,
            "currently_on_hrt": None,
            "first_degree_relatives_with_breast_cancer": 0,
            "breast_density": None,
            "prior_breast_procedure": False,
            "last_mammogram_result": "negative",
        }
        self.assertEqual(human_to_interface_record(patient)["agefirst"], 3)
        conflict = dict(patient)
        conflict["age_at_first_birth"] = 25
        with self.assertRaises(InputMappingError):
            human_to_interface_record(conflict)

    def test_unknown_fields(self) -> None:
        patient = {
            "age": 42,
            "bmi": None,
            "has_given_birth": True,
            "age_at_first_birth": None,
            "is_postmenopausal": None,
            "menopause_type": None,
            "currently_on_hrt": None,
            "first_degree_relatives_with_breast_cancer": None,
            "breast_density": None,
            "prior_breast_procedure": None,
            "last_mammogram_result": None,
        }
        record = human_to_interface_record(patient)
        self.assertEqual(record["bmi"], 0)
        self.assertEqual(record["density"], 0)
        self.assertEqual(record["menopaus"], 0)
        self.assertEqual(record["nrelbc"], 0)
        self.assertEqual(record["brstproc"], 0)
        self.assertEqual(record["lastmamm"], 0)

    def test_invalid_values(self) -> None:
        base = {
            "age": 42,
            "bmi": 23.0,
            "has_given_birth": True,
            "age_at_first_birth": 26,
            "is_postmenopausal": True,
            "menopause_type": "natural",
            "currently_on_hrt": False,
            "first_degree_relatives_with_breast_cancer": 1,
            "breast_density": "heterogeneously dense",
            "prior_breast_procedure": False,
            "last_mammogram_result": "negative",
        }
        invalid_cases = [
            ("breast_density", "unknown density"),
            ("last_mammogram_result", "positive"),
            ("first_degree_relatives_with_breast_cancer", "2"),
            ("first_degree_relatives_with_breast_cancer", -1),
            ("menopause_type", "induced"),
            ("has_given_birth", 1),
            ("is_postmenopausal", "False"),
            ("prior_breast_procedure", 0),
        ]
        for field, value in invalid_cases:
            patient = dict(base)
            patient[field] = value
            with self.assertRaises(InputMappingError):
                human_to_interface_record(patient)

        # Conditional validation.
        pre_with_type = dict(base)
        pre_with_type["is_postmenopausal"] = False
        with self.assertRaises(InputMappingError):
            human_to_interface_record(pre_with_type)

        # Exact fields.
        missing = dict(base)
        del missing["age"]
        with self.assertRaises(InputMappingError):
            human_to_interface_record(missing)
        extra = dict(base)
        extra["count"] = 1
        with self.assertRaises(InputMappingError):
            human_to_interface_record(extra)

    def test_frame_multiple_patients(self) -> None:
        patients = [
            {
                "age": 42,
                "bmi": 27.5,
                "has_given_birth": True,
                "age_at_first_birth": 26,
                "is_postmenopausal": False,
                "menopause_type": None,
                "currently_on_hrt": None,
                "first_degree_relatives_with_breast_cancer": 1,
                "breast_density": "heterogeneously dense",
                "prior_breast_procedure": False,
                "last_mammogram_result": "negative",
            },
            {
                "age": 61,
                "bmi": 32.1,
                "has_given_birth": True,
                "age_at_first_birth": 31,
                "is_postmenopausal": True,
                "menopause_type": "natural",
                "currently_on_hrt": False,
                "first_degree_relatives_with_breast_cancer": "2+",
                "breast_density": "extremely dense",
                "prior_breast_procedure": True,
                "last_mammogram_result": "false positive",
            },
        ]
        frame = human_to_interface_frame(patients)
        self.assertEqual(list(frame.columns), FEATURES)
        self.assertEqual(len(frame), 2)
        single = human_to_interface_frame(patients[0])
        self.assertEqual(len(single), 1)

    def test_encode_human_inputs_onehot(self) -> None:
        patients = [
            {
                "age": 42,
                "bmi": 27.5,
                "has_given_birth": True,
                "age_at_first_birth": 26,
                "is_postmenopausal": False,
                "menopause_type": None,
                "currently_on_hrt": None,
                "first_degree_relatives_with_breast_cancer": 1,
                "breast_density": "heterogeneously dense",
                "prior_breast_procedure": False,
                "last_mammogram_result": "negative",
            }
        ]
        encoded = encode_human_inputs(patients, self.mapper, self.onehot)
        self.assertEqual(encoded.shape, (1, 43))
        self.assertTrue(__import__("numpy").isfinite(encoded).all())

    def test_encode_human_inputs_ordinal(self) -> None:
        patients = [
            {
                "age": 53,
                "bmi": 23,
                "has_given_birth": False,
                "age_at_first_birth": None,
                "is_postmenopausal": None,
                "menopause_type": None,
                "currently_on_hrt": None,
                "first_degree_relatives_with_breast_cancer": 0,
                "breast_density": None,
                "prior_breast_procedure": False,
                "last_mammogram_result": "negative",
            }
        ]
        encoded = encode_human_inputs(patients, self.mapper, self.ordinal)
        self.assertEqual(encoded.shape, (1, 26))
        self.assertTrue(__import__("numpy").isfinite(encoded).all())

    def test_encode_human_inputs_matches_existing_path(self) -> None:
        patient = {
            "age": 42,
            "bmi": 27.5,
            "has_given_birth": True,
            "age_at_first_birth": 26,
            "is_postmenopausal": False,
            "menopause_type": None,
            "currently_on_hrt": None,
            "first_degree_relatives_with_breast_cancer": 1,
            "breast_density": "heterogeneously dense",
            "prior_breast_procedure": False,
            "last_mammogram_result": "negative",
        }
        interface = human_to_interface_frame(patient)
        from src.preprocessing.pipeline import encode_user_inputs

        expected = encode_user_inputs(self.mapper, self.onehot, interface)
        actual = encode_human_inputs(patient, self.mapper, self.onehot)
        __import__("numpy").testing.assert_allclose(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
