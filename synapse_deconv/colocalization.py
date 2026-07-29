"""Pairing pre- and post-synaptic puncta into synapses.

A synapse is not an overlap. The pre- and post-synaptic densities sit on either
side of a ~20-30 nm cleft, an order of magnitude below the optical resolution,
so the two labels appear as *apposed* -- adjacent, usually not superimposed.
Requiring voxel overlap would reject most real synapses; the criterion is a
centroid-to-centroid distance below a tolerance.

Distances are computed in micrometres from centroids that were already
converted out of voxel space, so the anisotropy of the grid (a 0.30 um z-step
against a 0.095-0.137 um pixel) is handled by construction rather than by
weighting an axis after the fact.

Matching is **one-to-one and globally optimal**: each pre-synaptic punctum
pairs with at most one post-synaptic punctum and vice versa, and the assignment
minimises the total distance over all pairs (Hungarian algorithm). A greedy
nearest-neighbour rule would let one bright post-synaptic punctum capture
several pre-synaptic ones and inflate the count in dense fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "PairingResult",
    "chance_pairing_rate",
    "estimate_channel_offset",
    "nearest_neighbour_profile",
    "pair_by_contact",
    "pair_puncta",
    "resolve_exclusive_partners",
]


@dataclass
class PairingResult:
    """Which puncta of two channels form synapses."""

    #: (M, 2) indices into the two input centroid arrays.
    pairs: np.ndarray
    #: (M,) centre-to-centre distance in micrometres.
    distances_um: np.ndarray
    #: (M, 3) midpoint of each pair, in micrometres.
    midpoints_um: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.pairs)


def pair_puncta(
    pre_centroids_um: np.ndarray,
    post_centroids_um: np.ndarray,
    *,
    tolerance_um: float = 0.30,
    one_to_one: bool = True,
) -> PairingResult:
    """Pair two sets of puncta by 3D apposition.

    Args:
        pre_centroids_um: (N, 3) presynaptic centroids in micrometres.
        post_centroids_um: (M, 3) postsynaptic centroids in micrometres.
        tolerance_um: maximum centre-to-centre distance for a synapse.
        one_to_one: enforce an exclusive, globally optimal assignment. Set to
            False to allow a punctum to take part in several pairs (useful only
            for exploring how much the exclusivity constraint costs).
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    pre = np.asarray(pre_centroids_um, dtype=float).reshape(-1, 3)
    post = np.asarray(post_centroids_um, dtype=float).reshape(-1, 3)
    empty = PairingResult(
        pairs=np.zeros((0, 2), dtype=int),
        distances_um=np.zeros((0,), dtype=float),
        midpoints_um=np.zeros((0, 3), dtype=float),
        diagnostics={"n_pre": len(pre), "n_post": len(post),
                     "tolerance_um": tolerance_um, "one_to_one": one_to_one},
    )
    if len(pre) == 0 or len(post) == 0:
        return empty

    # Candidate pairs within tolerance, found with a KD-tree so dense fields do
    # not turn into an N*M distance matrix.
    tree = cKDTree(post)
    candidates = tree.query_ball_point(pre, r=tolerance_um)
    edges = [(i, j) for i, js in enumerate(candidates) for j in js]
    if not edges:
        return empty

    edge_array = np.asarray(edges)
    deltas = pre[edge_array[:, 0]] - post[edge_array[:, 1]]
    edge_distances = np.linalg.norm(deltas, axis=1)

    if not one_to_one:
        pairs = edge_array
        distances = edge_distances
    else:
        # Restrict the assignment problem to puncta that actually have a
        # candidate: the cost matrix is then small even in a dense field.
        pre_used = np.unique(edge_array[:, 0])
        post_used = np.unique(edge_array[:, 1])
        pre_pos = {value: k for k, value in enumerate(pre_used)}
        post_pos = {value: k for k, value in enumerate(post_used)}

        # Unreachable pairs get a cost above tolerance so the optimiser never
        # picks them; they are filtered out afterwards regardless.
        big = tolerance_um * 10.0 + 1.0
        cost = np.full((len(pre_used), len(post_used)), big, dtype=float)
        for (i, j), distance in zip(edge_array, edge_distances):
            cost[pre_pos[i], post_pos[j]] = distance

        rows, cols = linear_sum_assignment(cost)
        selected = [
            (pre_used[r], post_used[c], cost[r, c])
            for r, c in zip(rows, cols)
            if cost[r, c] <= tolerance_um
        ]
        if not selected:
            return empty
        pairs = np.asarray([(i, j) for i, j, _ in selected], dtype=int)
        distances = np.asarray([d for _, _, d in selected], dtype=float)

    order = np.argsort(distances)
    pairs, distances = pairs[order], distances[order]
    midpoints = (pre[pairs[:, 0]] + post[pairs[:, 1]]) / 2.0

    return PairingResult(
        pairs=pairs,
        distances_um=distances,
        midpoints_um=midpoints,
        diagnostics={
            "n_pre": len(pre), "n_post": len(post),
            "n_candidate_edges": len(edges),
            "tolerance_um": tolerance_um, "one_to_one": one_to_one,
            "median_distance_um": float(np.median(distances)),
            "fraction_pre_paired": len(pairs) / len(pre),
            "fraction_post_paired": len(pairs) / len(post),
        },
    )


def pair_by_contact(
    pre_labels: np.ndarray,
    post_labels: np.ndarray,
    pre_centroids_um: np.ndarray,
    post_centroids_um: np.ndarray,
    voxel_size_um: tuple[float, float, float],
    *,
    dilation_um: float = 0.1,
    one_to_one: bool = True,
) -> PairingResult:
    """Alternative criterion: segmented masks that touch after a small dilation.

    Less sensitive than the centroid distance to puncta of very unequal size,
    but it needs the segmentation to be reliable, and it cannot pair two puncta
    whose masks stay apart -- which apposed densities sometimes do.
    """
    from scipy.ndimage import grey_dilation

    dz, dy, dx = voxel_size_um
    size = (max(1, int(round(dilation_um / dz))) * 2 + 1,
            max(1, int(round(dilation_um / dy))) * 2 + 1,
            max(1, int(round(dilation_um / dx))) * 2 + 1)
    grown = grey_dilation(pre_labels, size=size)

    overlap = (grown > 0) & (post_labels > 0)
    if not np.any(overlap):
        return PairingResult(
            pairs=np.zeros((0, 2), dtype=int),
            distances_um=np.zeros((0,), dtype=float),
            midpoints_um=np.zeros((0, 3), dtype=float),
            diagnostics={"criterion": "contact", "dilation_um": dilation_um,
                         "n_pre": len(pre_centroids_um), "n_post": len(post_centroids_um)},
        )

    # Label ids are 1-based and dense, so a pair of ids identifies a contact.
    contacts = np.unique(
        np.stack([grown[overlap], post_labels[overlap]], axis=1), axis=0
    )
    pre = np.asarray(pre_centroids_um, dtype=float).reshape(-1, 3)
    post = np.asarray(post_centroids_um, dtype=float).reshape(-1, 3)
    pairs, distances = [], []
    for pre_label, post_label in contacts:
        i, j = int(pre_label) - 1, int(post_label) - 1
        if 0 <= i < len(pre) and 0 <= j < len(post):
            pairs.append((i, j))
            distances.append(float(np.linalg.norm(pre[i] - post[j])))

    if not pairs:
        return PairingResult(
            pairs=np.zeros((0, 2), dtype=int),
            distances_um=np.zeros((0,), dtype=float),
            midpoints_um=np.zeros((0, 3), dtype=float),
            diagnostics={"criterion": "contact", "dilation_um": dilation_um},
        )

    pairs = np.asarray(pairs, dtype=int)
    distances = np.asarray(distances, dtype=float)
    if one_to_one:
        # Keep the closest contact for each punctum on both sides.
        order = np.argsort(distances)
        seen_pre: set[int] = set()
        seen_post: set[int] = set()
        keep = []
        for k in order:
            i, j = pairs[k]
            if i in seen_pre or j in seen_post:
                continue
            seen_pre.add(int(i))
            seen_post.add(int(j))
            keep.append(k)
        pairs, distances = pairs[keep], distances[keep]

    midpoints = (pre[pairs[:, 0]] + post[pairs[:, 1]]) / 2.0
    return PairingResult(
        pairs=pairs, distances_um=distances, midpoints_um=midpoints,
        diagnostics={"criterion": "contact", "dilation_um": dilation_um,
                     "n_pre": len(pre), "n_post": len(post),
                     "median_distance_um": float(np.median(distances))},
    )


def estimate_channel_offset(
    reference_um: np.ndarray,
    moving_um: np.ndarray,
    *,
    search_radius_um: float = 1.0,
    min_pairs: int = 20,
) -> dict[str, float]:
    """Systematic displacement between two channels, from mutual nearest neighbours.

    Chromatic aberration and detector misalignment shift one channel against
    another by an amount that is small in absolute terms but large compared with
    a 0.3 um apposition tolerance -- between 421 and 618 nm emission on a high-NA
    lens it routinely reaches a few hundred nanometres, and it is worse along z.
    A shift of that size moves every true pair outside the tolerance at once, so
    the pairing collapses while each channel individually looks perfect.

    The estimate is the componentwise median of the displacement between mutual
    nearest neighbours. Real pairs contribute a consistent vector; unrelated
    puncta contribute isotropic noise that the median rejects.
    """
    from scipy.spatial import cKDTree

    reference = np.asarray(reference_um, dtype=float).reshape(-1, 3)
    moving = np.asarray(moving_um, dtype=float).reshape(-1, 3)
    empty = {"n_pairs": 0, "shift_z_um": 0.0, "shift_y_um": 0.0, "shift_x_um": 0.0,
             "shift_norm_um": 0.0, "scatter_um": 0.0, "scatter_ratio": float("inf")}
    if len(reference) < min_pairs or len(moving) < min_pairs:
        return empty

    forward = cKDTree(moving).query(reference, distance_upper_bound=search_radius_um)
    backward = cKDTree(reference).query(moving, distance_upper_bound=search_radius_um)
    displacements = [
        moving[j] - reference[i]
        for i, (distance, j) in enumerate(zip(*forward))
        if np.isfinite(distance) and j < len(moving) and backward[1][j] == i
    ]
    if len(displacements) < min_pairs:
        return empty

    deltas = np.asarray(displacements)
    shift = np.median(deltas, axis=0)
    # Median absolute deviation of the residual: small means a genuine
    # systematic shift, large means the median is averaging noise.
    scatter = float(np.median(np.linalg.norm(deltas - shift, axis=1)))
    norm = float(np.linalg.norm(shift))
    return {
        "n_pairs": len(deltas),
        "shift_z_um": float(shift[0]),
        "shift_y_um": float(shift[1]),
        "shift_x_um": float(shift[2]),
        "shift_norm_um": norm,
        "scatter_um": scatter,
        # Below ~1 the displacement is consistent across pairs, so the shift is
        # real. Well above 1 the median is averaging unrelated puncta and the
        # "shift" is finite-sample noise, not an instrumental offset.
        "scatter_ratio": scatter / norm if norm > 0 else float("inf"),
    }


def nearest_neighbour_profile(
    pre_centroids_um: np.ndarray,
    post_centroids_um: np.ndarray,
    extent_um: np.ndarray,
    *,
    max_distance_um: float = 1.5,
    n_bins: int = 15,
    n_randomisations: int = 10,
    min_expected: float = 5.0,
    seed: int = 0,
) -> dict[str, list]:
    """Fraction of presynaptic puncta with a partner within r, against chance.

    Answers the question a fixed tolerance cannot: **is there real apposition,
    and over what distance?** For every candidate tolerance r the profile gives
    how many presynaptic puncta have a postsynaptic neighbour within r, and how
    many would by chance if one channel were randomly translated.

    The counts are **cumulative**, not binned. A per-bin ratio is unusable at
    short distance, where both counts are tiny: a bin holding 4 observed against
    0.2 expected reports an enrichment of 20 that is pure small-number noise.
    The cumulative form is monotone, stable, and is exactly the quantity a
    tolerance selects.

    Reading it: enrichment well above 1 at short r that decays towards 1 means
    the tolerance can be read off the curve -- take r just before the decay. A
    curve flat at 1 everywhere means the channels are not associated and no
    tolerance will produce a meaningful count. Ratios computed from fewer than
    ``min_expected`` chance counts are returned as None rather than a number
    that only reflects sampling noise.
    """
    from scipy.spatial import cKDTree

    pre = np.asarray(pre_centroids_um, dtype=float).reshape(-1, 3)
    post = np.asarray(post_centroids_um, dtype=float).reshape(-1, 3)
    extent = np.asarray(extent_um, dtype=float)
    radii = np.linspace(max_distance_um / n_bins, max_distance_um, n_bins)
    empty = {"radius_um": radii.tolist(), "observed": [], "chance": [],
             "enrichment": [], "fraction_of_pre": []}
    if len(pre) < 10 or len(post) < 10 or np.any(extent <= 0):
        return empty

    def cumulative(points: np.ndarray) -> np.ndarray:
        distances, _ = cKDTree(points).query(pre)
        return np.array([float((distances <= r).sum()) for r in radii])

    observed = cumulative(post)
    rng = np.random.default_rng(seed)
    chance = np.mean(
        [cumulative((post + rng.uniform(0, extent)) % extent)
         for _ in range(n_randomisations)],
        axis=0,
    )
    enrichment = [
        round(float(o / c), 3) if c >= min_expected else None
        for o, c in zip(observed, chance)
    ]
    return {
        "radius_um": [round(float(r), 3) for r in radii],
        "observed": observed.tolist(),
        "chance": [round(float(c), 2) for c in chance],
        "enrichment": enrichment,
        # The raw coverage, which the enrichment ratio hides: what share of the
        # presynaptic puncta have ANY postsynaptic punctum within r. This is the
        # quantity the literature reports (most Bassoon puncta appose a PSD-95),
        # and it separates the two ways a synapse count can be too low. If
        # coverage stays low at every radius, partners are missing from the
        # detection and no tolerance recovers them; if coverage is high but only
        # at large r, the tolerance or the channel registration is the problem.
        "fraction_of_pre": [round(float(o / len(pre)), 4) for o in observed],
    }


def chance_pairing_rate(
    pre_centroids_um: np.ndarray,
    post_centroids_um: np.ndarray,
    extent_um: np.ndarray,
    *,
    tolerance_um: float = 0.30,
    one_to_one: bool = True,
    n_randomisations: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """How many pairs the same criterion would find by chance alone.

    Two puncta can fall within the tolerance without being a synapse, and the
    denser the labelling the more often that happens. The control applies a
    random rigid translation (with wrap-around) to one channel: each channel
    keeps its own density and clustering, but the true association between them
    is destroyed. What the pairing still finds is the coincidence rate.

    Report the observed count *and* this rate. A count that is not several times
    the chance level is not evidence of colocalisation.
    """
    pre = np.asarray(pre_centroids_um, dtype=float).reshape(-1, 3)
    post = np.asarray(post_centroids_um, dtype=float).reshape(-1, 3)
    extent = np.asarray(extent_um, dtype=float)
    if len(pre) == 0 or len(post) == 0 or np.any(extent <= 0):
        return {"chance_mean": 0.0, "chance_std": 0.0, "n_randomisations": 0}

    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(n_randomisations):
        # A shift of at least a tolerance in each axis, wrapped into the field.
        shift = rng.uniform(tolerance_um * 3, extent - tolerance_um * 3)
        shifted = (post + shift) % extent
        counts.append(
            pair_puncta(pre, shifted, tolerance_um=tolerance_um,
                        one_to_one=one_to_one).count
        )
    return {
        "chance_mean": float(np.mean(counts)),
        "chance_std": float(np.std(counts)),
        "n_randomisations": int(n_randomisations),
    }


def resolve_exclusive_partners(
    excitatory: PairingResult,
    inhibitory: PairingResult,
) -> tuple[PairingResult, PairingResult, int]:
    """Stop one presynaptic punctum being counted as two different synapses.

    Both results must come from pairing the SAME presynaptic detection, since
    the conflict is detected on the presynaptic index: ``excitatory.pairs[:, 0]``
    and ``inhibitory.pairs[:, 0]`` have to refer to one array of centroids.

    A Bassoon punctum close enough to both a PSD-95 and a Gephyrin punctum would
    otherwise be counted once in each class, inflating the total. A terminal is
    either excitatory or inhibitory, so the conflict is resolved in favour of
    the closer partner and the other pairing is dropped.

    Returns the two filtered results and the number of conflicts resolved.
    """
    if excitatory.count == 0 or inhibitory.count == 0:
        return excitatory, inhibitory, 0

    exc_pre = {int(i): k for k, i in enumerate(excitatory.pairs[:, 0])}
    inh_pre = {int(i): k for k, i in enumerate(inhibitory.pairs[:, 0])}
    conflicts = set(exc_pre) & set(inh_pre)
    if not conflicts:
        return excitatory, inhibitory, 0

    drop_exc, drop_inh = set(), set()
    for pre_index in conflicts:
        k_exc, k_inh = exc_pre[pre_index], inh_pre[pre_index]
        if excitatory.distances_um[k_exc] <= inhibitory.distances_um[k_inh]:
            drop_inh.add(k_inh)
        else:
            drop_exc.add(k_exc)

    def filtered(result: PairingResult, drop: set[int]) -> PairingResult:
        if not drop:
            return result
        keep = np.array([k for k in range(result.count) if k not in drop], dtype=int)
        diagnostics = dict(result.diagnostics)
        diagnostics["n_dropped_ambiguous"] = len(drop)
        return PairingResult(
            pairs=result.pairs[keep],
            distances_um=result.distances_um[keep],
            midpoints_um=result.midpoints_um[keep],
            diagnostics=diagnostics,
        )

    logger.info(
        "%d presynaptic punctum(a) matched both PSD-95 and Gephyrin; "
        "assigned to the closer partner", len(conflicts),
    )
    return filtered(excitatory, drop_exc), filtered(inhibitory, drop_inh), len(conflicts)
