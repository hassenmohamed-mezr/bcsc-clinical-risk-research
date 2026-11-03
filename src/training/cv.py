"""Training-only, ``count``-weighted cross-validation utilities.

The dataset is aggregated: each row represents ``count`` individual cases. This
module never expands rows. It constructs deterministic weighted class-stratified
folds, reports the represented weight and weighted cancer prevalence of every
fold, and computes the project's weighted metrics.

Nothing here fits a model on import. ``cross_validate_weighted`` only fits when
a caller supplies an encoded Training matrix and a model factory; the
configuration checks and unit tests use synthetic data or a non-training spy.

The external Validation split is not accepted by these helpers, and the Test
split is never referenced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.config import DEFAULT_THRESHOLD, RANDOM_STATE
from src.training.config import CV_POLICY, POSITIVE_CLASS

PRIMARY_SCORER: str = "pr_auc"
SUPPORTING_SCORERS: tuple[str, ...] = ("roc_auc", "neg_brier")


# ---- weighted fold construction ------------------------------------------
@dataclass(frozen=True)
class WeightedFold:
    """One deterministic fold over aggregated rows (no row expansion)."""

    fold: int
    train_index: np.ndarray
    validation_index: np.ndarray


def _validate_weighted_target(
    y: Sequence[int] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y)
    w_arr = np.asarray(sample_weight, dtype=float)
    if y_arr.ndim != 1 or w_arr.ndim != 1:
        raise ValueError("y and sample_weight must be one-dimensional")
    if len(y_arr) != len(w_arr):
        raise ValueError(
            f"y length ({len(y_arr)}) does not match sample_weight length ({len(w_arr)})"
        )
    if len(y_arr) == 0:
        raise ValueError("cannot build folds from an empty target")
    if not np.all(np.isin(y_arr, (0, 1))):
        raise ValueError("target must be binary 0/1")
    if not np.isfinite(w_arr).all() or not (w_arr > 0).all():
        raise ValueError("sample_weight must be finite and strictly positive")
    return y_arr.astype(int), w_arr


def weighted_folds(
    y: Sequence[int] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray,
    *,
    n_splits: int | None = None,
    random_state: int | None = None,
) -> list[WeightedFold]:
    """Build deterministic weighted, cancer-stratified folds without expansion.

    Within each class, rows are ordered by descending ``count`` (ties follow a
    seeded permutation) and assigned one at a time to the fold with the smallest
    accumulated represented weight *for that class*. This greedy rule balances
    both class weight and total represented weight. It is an approximation, not
    an exact optimal stratification; the fold diagnostics report the achieved
    balance.

    Every row appears in exactly one validation fold. The test portion of a fold
    is never used to fit the scaler or the model.
    """
    y_arr, w_arr = _validate_weighted_target(y, sample_weight)
    n_splits = CV_POLICY["n_splits"] if n_splits is None else int(n_splits)
    random_state = CV_POLICY["seed"] if random_state is None else int(random_state)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(y_arr):
        raise ValueError("n_splits cannot exceed the number of rows")

    rng = np.random.default_rng(random_state)
    class_weight = np.zeros((n_splits, 2), dtype=float)
    assignment = np.full(len(y_arr), -1, dtype=int)

    for class_label in (0, 1):
        class_rows = np.flatnonzero(y_arr == class_label)
        if len(class_rows) < n_splits:
            raise ValueError(
                f"class {class_label} has {len(class_rows)} rows, fewer than "
                f"n_splits={n_splits}"
            )
        shuffled = rng.permutation(class_rows)
        order = shuffled[np.argsort(-w_arr[shuffled], kind="stable")]
        for row in order:
            # Tie-break on total fold weight, then fold id, for determinism.
            best_fold = min(
                range(n_splits),
                key=lambda fold: (
                    class_weight[fold, class_label],
                    class_weight[fold].sum(),
                    fold,
                ),
            )
            assignment[row] = best_fold
            class_weight[best_fold, class_label] += w_arr[row]

    all_rows = np.arange(len(y_arr))
    return [
        WeightedFold(
            fold=fold,
            train_index=all_rows[assignment != fold],
            validation_index=all_rows[assignment == fold],
        )
        for fold in range(n_splits)
    ]


class WeightedStratifiedKFold:
    """scikit-learn-compatible splitter backed by :func:`weighted_folds`.

    The weight vector is supplied at construction because scikit-learn's
    ``split(X, y)`` protocol has no weight argument. The object is accepted by
    ``CalibratedClassifierCV(cv=...)`` and by code that consumes an iterable of
    ``(train_index, validation_index)`` pairs.
    """

    def __init__(
        self,
        n_splits: int | None = None,
        *,
        sample_weight: Sequence[float] | np.ndarray,
        random_state: int | None = None,
    ):
        self.n_splits = CV_POLICY["n_splits"] if n_splits is None else int(n_splits)
        self.sample_weight = sample_weight
        self.random_state = CV_POLICY["seed"] if random_state is None else int(random_state)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y, groups=None):
        folds = weighted_folds(
            y,
            self.sample_weight,
            n_splits=self.n_splits,
            random_state=self.random_state,
        )
        for fold in folds:
            yield fold.train_index, fold.validation_index


# ---- fold diagnostics -----------------------------------------------------
@dataclass(frozen=True)
class FoldDiagnostics:
    fold: int
    rows: int
    positive_rows: int
    weight_sum: float
    positive_weight: float
    negative_weight: float
    weighted_prevalence: float
    weight_share: float
    weight_relative_deviation: float
    prevalence_relative_deviation: float


def _relative_deviation(value: float, reference: float) -> float:
    if reference == 0:
        return 0.0 if value == 0 else float("inf")
    return abs(value - reference) / abs(reference)


def fold_diagnostics(
    y: Sequence[int] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray,
    folds: Iterable[WeightedFold],
) -> list[FoldDiagnostics]:
    """Report rows, represented weight, and weighted cancer prevalence per fold."""
    y_arr, w_arr = _validate_weighted_target(y, sample_weight)
    total_weight = float(w_arr.sum())
    overall_prevalence = float(w_arr[y_arr == 1].sum()) / total_weight
    fold_list = folds if isinstance(folds, list) else list(folds)
    if not fold_list:
        raise ValueError("at least one fold is required")

    diagnostics: list[FoldDiagnostics] = []
    for fold in fold_list:
        idx = np.asarray(fold.validation_index)
        fold_weight = float(w_arr[idx].sum())
        positive_weight = float(w_arr[idx][y_arr[idx] == 1].sum())
        negative_weight = fold_weight - positive_weight
        prevalence = positive_weight / fold_weight if fold_weight > 0 else float("nan")
        diagnostics.append(
            FoldDiagnostics(
                fold=fold.fold,
                rows=int(len(idx)),
                positive_rows=int((y_arr[idx] == 1).sum()),
                weight_sum=fold_weight,
                positive_weight=positive_weight,
                negative_weight=negative_weight,
                weighted_prevalence=prevalence,
                weight_share=fold_weight / total_weight if total_weight else float("nan"),
                weight_relative_deviation=_relative_deviation(
                    fold_weight, total_weight / len(fold_list)
                ),
                prevalence_relative_deviation=_relative_deviation(
                    prevalence, overall_prevalence
                ),
            )
        )
    return diagnostics


def assess_fold_balance(
    diagnostics: Sequence[FoldDiagnostics],
    *,
    weight_tolerance: float | None = None,
    prevalence_tolerance: float | None = None,
) -> dict[str, Any]:
    """Check whether folds differ materially in represented weight or prevalence.

    The tolerances are pre-declared diagnostic thresholds, not a statistical
    test. A violation means the fold construction should be reviewed before the
    model comparison; it does not by itself invalidate the dataset.
    """
    tolerance = CV_POLICY["balance_tolerance"]
    weight_tolerance = (
        tolerance["weight_relative_deviation"]
        if weight_tolerance is None
        else float(weight_tolerance)
    )
    prevalence_tolerance = (
        tolerance["prevalence_relative_deviation"]
        if prevalence_tolerance is None
        else float(prevalence_tolerance)
    )

    violations: list[str] = []
    for diag in diagnostics:
        if diag.weight_relative_deviation > weight_tolerance:
            violations.append(
                f"fold {diag.fold}: represented weight deviates "
                f"{diag.weight_relative_deviation:.4%} > {weight_tolerance:.2%}"
            )
        if diag.prevalence_relative_deviation > prevalence_tolerance:
            violations.append(
                f"fold {diag.fold}: weighted prevalence deviates "
                f"{diag.prevalence_relative_deviation:.4%} > {prevalence_tolerance:.2%}"
            )
    return {
        "within_tolerance": not violations,
        "material_imbalance": bool(violations),
        "weight_relative_deviation_tolerance": weight_tolerance,
        "prevalence_relative_deviation_tolerance": prevalence_tolerance,
        "violations": violations,
    }


# ---- weighted metrics and scorers -----------------------------------------
def weighted_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """All project metrics, with ``count`` applied as ``sample_weight``."""
    y_arr, w_arr = _validate_weighted_target(y_true, sample_weight)
    score = np.asarray(y_score, dtype=float)
    if score.ndim != 1 or len(score) != len(y_arr):
        raise ValueError("y_score must be one-dimensional and aligned with y_true")
    if not np.isfinite(score).all() or not ((score >= 0) & (score <= 1)).all():
        raise ValueError("predicted probabilities must be finite and in [0, 1]")

    total_weight = float(w_arr.sum())
    positive_weight = float(w_arr[y_arr == 1].sum())
    negative_weight = total_weight - positive_weight
    prevalence = positive_weight / total_weight
    predicted_positive = score >= threshold

    tp_w = float(w_arr[(y_arr == 1) & predicted_positive].sum())
    fp_w = float(w_arr[(y_arr == 0) & predicted_positive].sum())
    fn_w = float(w_arr[(y_arr == 1) & ~predicted_positive].sum())
    tn_w = float(w_arr[(y_arr == 0) & ~predicted_positive].sum())

    undefined: list[str] = []
    if positive_weight > 0 and negative_weight > 0:
        roc_auc = float(roc_auc_score(y_arr, score, sample_weight=w_arr))
        pr_auc = float(
            average_precision_score(
                y_arr, score, pos_label=POSITIVE_CLASS, sample_weight=w_arr
            )
        )
    else:
        roc_auc = None
        pr_auc = None
        undefined.append("ROC-AUC and PR-AUC require both classes")

    recall = tp_w / positive_weight if positive_weight > 0 else None
    specificity = tn_w / negative_weight if negative_weight > 0 else None
    if tp_w + fp_w > 0:
        precision = tp_w / (tp_w + fp_w)
    else:
        precision = None
        undefined.append("precision is undefined: no predicted positive weight")
    if positive_weight == 0:
        undefined.append("recall is undefined: no positive weight")
    if negative_weight == 0:
        undefined.append("specificity is undefined: no negative weight")

    brier = float(brier_score_loss(y_arr, score, sample_weight=w_arr))
    return {
        "threshold": float(threshold),
        "rows": int(len(y_arr)),
        "weight_sum": total_weight,
        "positive_weight": positive_weight,
        "negative_weight": negative_weight,
        "weighted_prevalence": prevalence,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "brier": brier,
        "tp_w": tp_w,
        "fp_w": fp_w,
        "tn_w": tn_w,
        "fn_w": fn_w,
        "predicted_positive_weight": tp_w + fp_w,
        "undefined": undefined,
    }


def weighted_scores(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float | None]:
    """Weighted scorer values used for CV selection (PR-AUC primary)."""
    metrics = weighted_metrics(
        y_true, y_score, sample_weight, threshold=threshold
    )
    return {
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "neg_brier": None if metrics["brier"] is None else -metrics["brier"],
    }


def row_equal_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Secondary diagnostic that treats each aggregated row equally.

    This is explicitly *not* the primary evaluation: it ignores ``count`` and
    must never be used for model selection.
    """
    y_arr = np.asarray(y_true)
    return weighted_metrics(
        y_arr,
        y_score,
        np.ones(len(y_arr), dtype=float),
        threshold=threshold,
    )


# ---- fold fitting and scoring ---------------------------------------------
@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_rows: int
    validation_rows: int
    train_weight_sum: float
    validation_weight_sum: float
    metrics: dict[str, Any]


def positive_class_probability(model: Any, X) -> np.ndarray:
    """Return the ``cancer = 1`` probability column from ``predict_proba``."""
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(model, "classes_"))
    matches = np.flatnonzero(classes == POSITIVE_CLASS)
    if len(matches) != 1:
        raise ValueError(
            f"model.classes_ must contain exactly one positive class "
            f"{POSITIVE_CLASS}; got {classes.tolist()}"
        )
    scores = probabilities[:, int(matches[0])]
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("model produced non-finite or out-of-range probabilities")
    return scores


def cross_validate_weighted(
    model_factory: Callable[[], Any],
    X,
    y: Sequence[int] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray,
    *,
    folds: Iterable[WeightedFold] | None = None,
    n_splits: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[FoldResult]:
    """Fit a fresh model per fold and score its validation rows with weights.

    ``X`` must be the encoded Training matrix. The same ``count`` weights are
    passed to ``fit`` (training portion) and to every scorer (validation
    portion). The MLP factory must return the scaler+MLP pipeline, so the scaler
    is fitted on the training portion only.

    This function is implemented for the next phase; it is deliberately not
    executed on project data in the configuration task.
    """
    y_arr, w_arr = _validate_weighted_target(y, sample_weight)
    if len(X) != len(y_arr):
        raise ValueError("X rows do not match y/sample_weight")
    fold_list = (
        weighted_folds(y_arr, w_arr, n_splits=n_splits)
        if folds is None
        else list(folds)
    )

    results: list[FoldResult] = []
    for fold in fold_list:
        train_idx = np.asarray(fold.train_index)
        validation_idx = np.asarray(fold.validation_index)
        model = model_factory()
        model.fit(
            X[train_idx],
            y_arr[train_idx],
            sample_weight=w_arr[train_idx],
        )
        scores = positive_class_probability(model, X[validation_idx])
        metrics = weighted_metrics(
            y_arr[validation_idx],
            scores,
            w_arr[validation_idx],
            threshold=threshold,
        )
        results.append(
            FoldResult(
                fold=fold.fold,
                train_rows=int(len(train_idx)),
                validation_rows=int(len(validation_idx)),
                train_weight_sum=float(w_arr[train_idx].sum()),
                validation_weight_sum=float(w_arr[validation_idx].sum()),
                metrics=metrics,
            )
        )
    return results


if __name__ == "__main__":
    import json
    import sys

    if "--with-train" not in sys.argv[1:]:
        print(
            "usage: python -m src.training.cv --with-train\n"
            "Read-only fold diagnostics on the Training split. "
            "No model is fitted and Validation/Test are not loaded."
        )
        raise SystemExit(0)

    from src.data.load_splits import load_split
    from src.training.config import TRAIN_SPLIT

    X_train, y_train, w_train = load_split(TRAIN_SPLIT)
    folds = weighted_folds(y_train, w_train)
    diagnostics = fold_diagnostics(y_train, w_train, folds)
    balance = assess_fold_balance(diagnostics)
    print(
        json.dumps(
            {
                "split": TRAIN_SPLIT,
                "rows": int(len(X_train)),
                "weight_sum": float(w_train.sum()),
                "positive_weight": float(w_train[y_train == 1].sum()),
                "weighted_prevalence": float(w_train[y_train == 1].sum())
                / float(w_train.sum()),
                "n_splits": len(folds),
                "folds": [
                    {
                        **{
                            key: value
                            for key, value in diag.__dict__.items()
                        },
                        "weighted_prevalence_percent": diag.weighted_prevalence * 100,
                        "weight_relative_deviation_percent": diag.weight_relative_deviation
                        * 100,
                        "prevalence_relative_deviation_percent": diag.prevalence_relative_deviation
                        * 100,
                    }
                    for diag in diagnostics
                ],
                "balance": balance,
            },
            indent=2,
        )
    )
