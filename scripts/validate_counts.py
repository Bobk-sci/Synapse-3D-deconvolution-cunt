"""Score the counting pipeline against a known ground truth.

Works on stacks produced by ``make_synapse_test_stack.py``, which writes the
true synapse positions next to the image. Detected synapses are matched to true
ones one-to-one within a tolerance, so a detection cannot be credited twice.

Reports recall (what fraction of real synapses were found), precision (what
fraction of reported synapses are real) and the F1 score, plus the counting
bias, which is what actually propagates into a group comparison.

    python scripts/validate_counts.py --truth data/synth/synth_synapses_01_truth.json \
        --puncta results_count/counts/synth_synapses_01_puncta.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_detected_synapses(puncta_csv: Path, presynaptic: str) -> dict[str, np.ndarray]:
    """Synapse positions, taken as the presynaptic centroid of each pair."""
    found: dict[str, list] = {"excitatory": [], "inhibitory": []}
    with puncta_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            kind = row["synapse_type"]
            if kind in found and row["channel"] == presynaptic:
                found[kind].append(
                    [float(row["z_um"]), float(row["y_um"]), float(row["x_um"])]
                )
    return {k: np.asarray(v).reshape(-1, 3) for k, v in found.items()}


def match(truth: np.ndarray, detected: np.ndarray, tolerance_um: float):
    """One-to-one optimal matching; returns (n_matched, median distance)."""
    if len(truth) == 0 or len(detected) == 0:
        return 0, float("nan")
    from scipy.optimize import linear_sum_assignment

    cost = np.linalg.norm(truth[:, None, :] - detected[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    keep = cost[rows, cols] <= tolerance_um
    distances = cost[rows, cols][keep]
    return int(keep.sum()), (float(np.median(distances)) if keep.any() else float("nan"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--puncta", required=True)
    parser.add_argument("--presynaptic", default="Bassoon")
    parser.add_argument("--tolerance-um", type=float, default=0.5,
                        help="how close a detection must be to count as the same synapse")
    args = parser.parse_args()

    truth = json.loads(Path(args.truth).read_text())
    detected = load_detected_synapses(Path(args.puncta), args.presynaptic)

    # The presynaptic centroid sits half a cleft from the recorded centre.
    print(f"tolérance d'appariement : {args.tolerance_um} um "
          f"(demi-fente vraie : {truth['cleft_um'] / 2:.3f} um)\n")
    header = f"{'type':<14}{'vrai':>7}{'détecté':>10}{'apparié':>10}{'rappel':>9}{'précision':>11}{'F1':>8}{'biais':>9}"
    print(header)
    print("-" * len(header))

    for kind in ("excitatory", "inhibitory"):
        true_points = np.asarray([[p["z_um"], p["y_um"], p["x_um"]] for p in truth[kind]])
        found = detected.get(kind, np.zeros((0, 3)))
        n_true, n_found = len(true_points), len(found)
        n_match, median_d = match(true_points, found, args.tolerance_um)

        recall = n_match / n_true if n_true else float("nan")
        precision = n_match / n_found if n_found else float("nan")
        f1 = (2 * recall * precision / (recall + precision)
              if recall + precision > 0 else float("nan"))
        bias = (n_found - n_true) / n_true if n_true else float("nan")
        print(f"{kind:<14}{n_true:>7}{n_found:>10}{n_match:>10}"
              f"{100 * recall:>8.1f}%{100 * precision:>10.1f}%{f1:>8.3f}{100 * bias:>+8.1f}%")

    print()
    total_true = truth["n_excitatory"] + truth["n_inhibitory"]
    total_found = sum(len(v) for v in detected.values())
    print(f"total : {total_true} vraies, {total_found} détectées "
          f"({100 * (total_found - total_true) / total_true:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
