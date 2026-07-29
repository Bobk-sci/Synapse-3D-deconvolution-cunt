"""Pairing puncta into synapses: apposition, exclusivity, chance level."""

from __future__ import annotations

import numpy as np
import pytest

from synapse_deconv.colocalization import (
    chance_pairing_rate,
    estimate_channel_offset,
    pair_by_contact,
    pair_puncta,
    resolve_exclusive_partners,
)


def test_apposed_pair_is_found_without_overlapping():
    """The cleft means the two centroids never coincide."""
    pre = np.array([[1.0, 2.0, 3.0]])
    post = np.array([[1.0, 2.0, 3.15]])          # 0.15 um apart
    result = pair_puncta(pre, post, tolerance_um=0.30)
    assert result.count == 1
    assert result.distances_um[0] == pytest.approx(0.15)
    np.testing.assert_allclose(result.midpoints_um[0], [1.0, 2.0, 3.075])


def test_pair_beyond_tolerance_is_rejected():
    pre = np.array([[0.0, 0.0, 0.0]])
    post = np.array([[0.0, 0.0, 0.5]])
    assert pair_puncta(pre, post, tolerance_um=0.30).count == 0
    assert pair_puncta(pre, post, tolerance_um=0.60).count == 1


def test_distance_is_physical_not_in_voxels():
    """0.30 um along z is one z-step but three pixels laterally."""
    pre = np.array([[0.0, 0.0, 0.0]])
    axial = np.array([[0.29, 0.0, 0.0]])
    lateral = np.array([[0.0, 0.0, 0.29]])
    # Both are the same physical distance, so both must pair or neither does.
    assert pair_puncta(pre, axial, tolerance_um=0.30).count == 1
    assert pair_puncta(pre, lateral, tolerance_um=0.30).count == 1
    assert pair_puncta(pre, np.array([[0.31, 0.0, 0.0]]), tolerance_um=0.30).count == 0
    assert pair_puncta(pre, np.array([[0.0, 0.0, 0.31]]), tolerance_um=0.30).count == 0


def test_one_to_one_stops_a_punctum_being_counted_twice():
    """Two presynaptic puncta near one postsynaptic punctum are not two synapses."""
    pre = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]])
    post = np.array([[0.0, 0.0, 0.1]])
    exclusive = pair_puncta(pre, post, tolerance_um=0.30, one_to_one=True)
    permissive = pair_puncta(pre, post, tolerance_um=0.30, one_to_one=False)
    assert exclusive.count == 1
    assert permissive.count == 2


def test_assignment_is_globally_optimal_not_greedy():
    """A greedy pass would take the first short edge and strand the rest."""
    pre = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.25]])
    post = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, 0.30]])
    result = pair_puncta(pre, post, tolerance_um=0.30, one_to_one=True)
    assert result.count == 2
    assert float(result.distances_um.sum()) == pytest.approx(0.10, abs=1e-9)


def test_empty_inputs_are_handled():
    empty = np.zeros((0, 3))
    assert pair_puncta(empty, np.array([[0.0, 0.0, 0.0]]), tolerance_um=0.3).count == 0
    assert pair_puncta(np.array([[0.0, 0.0, 0.0]]), empty, tolerance_um=0.3).count == 0
    assert pair_puncta(empty, empty, tolerance_um=0.3).count == 0


def test_ambiguous_presynaptic_punctum_is_assigned_to_the_closer_partner():
    """A terminal is excitatory or inhibitory, never counted as both."""
    pre = np.array([[0.0, 0.0, 0.0]])
    psd = np.array([[0.0, 0.0, 0.10]])
    geph = np.array([[0.0, 0.0, 0.25]])

    excitatory = pair_puncta(pre, psd, tolerance_um=0.30)
    inhibitory = pair_puncta(pre, geph, tolerance_um=0.30)
    assert excitatory.count == 1 and inhibitory.count == 1     # both, before resolution

    exc, inh, n_conflicts = resolve_exclusive_partners(excitatory, inhibitory)
    assert n_conflicts == 1
    assert exc.count == 1                    # PSD-95 is closer
    assert inh.count == 0


def test_resolution_is_a_no_op_without_conflicts():
    """Both pairings index the SAME presynaptic array, as in the pipeline."""
    pre = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    exc = pair_puncta(pre, np.array([[0.0, 0.0, 0.1]]), tolerance_um=0.3)
    inh = pair_puncta(pre, np.array([[5.0, 0.0, 0.1]]), tolerance_um=0.3)
    # Different presynaptic puncta, so nothing to arbitrate.
    assert exc.pairs[0, 0] != inh.pairs[0, 0]
    a, b, n = resolve_exclusive_partners(exc, inh)
    assert (a.count, b.count, n) == (1, 1, 0)


def test_chance_level_is_low_when_puncta_are_sparse():
    rng = np.random.default_rng(1)
    extent = np.array([9.0, 25.0, 25.0])
    pre = rng.uniform(0, extent, size=(150, 3))
    # Genuine partners, 0.15 um away.
    post = pre + rng.normal(0, 0.05, size=pre.shape)

    observed = pair_puncta(pre, post, tolerance_um=0.30).count
    chance = chance_pairing_rate(pre, post, extent, tolerance_um=0.30, n_randomisations=5)
    assert observed > 100
    assert chance["chance_mean"] < 0.15 * observed


def test_chance_level_rises_with_density():
    """The same tolerance buys more accidental pairs in a crowded field."""
    rng = np.random.default_rng(2)
    extent = np.array([5.0, 10.0, 10.0])
    sparse_pre = rng.uniform(0, extent, size=(50, 3))
    sparse_post = rng.uniform(0, extent, size=(50, 3))
    dense_pre = rng.uniform(0, extent, size=(1500, 3))
    dense_post = rng.uniform(0, extent, size=(1500, 3))

    sparse = chance_pairing_rate(sparse_pre, sparse_post, extent,
                                 tolerance_um=0.30, n_randomisations=5)
    dense = chance_pairing_rate(dense_pre, dense_post, extent,
                                tolerance_um=0.30, n_randomisations=5)
    assert dense["chance_mean"] / 1500 > sparse["chance_mean"] / 50


def test_contact_criterion_pairs_touching_masks():
    pre_labels = np.zeros((10, 20, 20), dtype=np.int32)
    post_labels = np.zeros((10, 20, 20), dtype=np.int32)
    pre_labels[5, 10, 9:11] = 1
    post_labels[5, 10, 11:13] = 1
    voxel = (0.30, 0.095, 0.095)
    pre_c = np.array([[5 * 0.30, 10 * 0.095, 9.5 * 0.095]])
    post_c = np.array([[5 * 0.30, 10 * 0.095, 11.5 * 0.095]])

    result = pair_by_contact(pre_labels, post_labels, pre_c, post_c, voxel,
                             dilation_um=0.15)
    assert result.count == 1
    assert result.diagnostics["criterion"] == "contact"


def test_contact_criterion_rejects_distant_masks():
    pre_labels = np.zeros((10, 20, 20), dtype=np.int32)
    post_labels = np.zeros((10, 20, 20), dtype=np.int32)
    pre_labels[5, 10, 2] = 1
    post_labels[5, 10, 17] = 1
    voxel = (0.30, 0.095, 0.095)
    result = pair_by_contact(
        pre_labels, post_labels,
        np.array([[1.5, 0.95, 0.19]]), np.array([[1.5, 0.95, 1.615]]),
        voxel, dilation_um=0.10,
    )
    assert result.count == 0


def test_pairing_is_deterministic():
    rng = np.random.default_rng(3)
    pre = rng.uniform(0, 10, size=(80, 3))
    post = pre + rng.normal(0, 0.06, size=pre.shape)
    a = pair_puncta(pre, post, tolerance_um=0.3)
    b = pair_puncta(pre, post, tolerance_um=0.3)
    np.testing.assert_array_equal(a.pairs, b.pairs)


def test_channel_offset_is_measured():
    """A systematic shift must be recovered, not averaged away."""
    rng = np.random.default_rng(10)
    reference = rng.uniform(0, 20, size=(300, 3))
    true_shift = np.array([0.35, 0.12, -0.08])
    moving = reference + true_shift + rng.normal(0, 0.03, size=reference.shape)

    offset = estimate_channel_offset(reference, moving, search_radius_um=1.0)
    assert offset["n_pairs"] > 200
    assert offset["shift_z_um"] == pytest.approx(true_shift[0], abs=0.03)
    assert offset["shift_y_um"] == pytest.approx(true_shift[1], abs=0.03)
    assert offset["shift_x_um"] == pytest.approx(true_shift[2], abs=0.03)
    assert offset["scatter_um"] < 0.1
    assert offset["scatter_ratio"] < 1.0        # consistent -> a real shift


def test_no_offset_is_reported_as_none():
    rng = np.random.default_rng(11)
    reference = rng.uniform(0, 20, size=(300, 3))
    moving = reference + rng.normal(0, 0.03, size=reference.shape)
    offset = estimate_channel_offset(reference, moving, search_radius_um=1.0)
    assert offset["shift_norm_um"] < 0.03


def test_unrelated_channels_give_a_large_scatter():
    """Without real correspondence the median is averaging noise; say so."""
    rng = np.random.default_rng(12)
    reference = rng.uniform(0, 20, size=(300, 3))
    moving = rng.uniform(0, 20, size=(300, 3))
    offset = estimate_channel_offset(reference, moving, search_radius_um=1.0)
    # scatter_ratio separates the two regimes cleanly: ~3.7 here against ~0.12
    # for a genuine shift, so 1.0 is a safe cut.
    assert offset["scatter_ratio"] > 1.0


def test_an_offset_collapses_the_pairing_and_correction_restores_it():
    """This is the failure mode the measurement exists to catch."""
    rng = np.random.default_rng(13)
    pre = rng.uniform(0, 20, size=(400, 3))
    true_post = pre + rng.normal(0, 0.05, size=pre.shape)      # real partners
    shift = np.array([0.0, 0.30, 0.25])                        # 0.39 um, > tolerance
    observed_post = true_post + shift

    without = pair_puncta(pre, observed_post, tolerance_um=0.30)
    offset = estimate_channel_offset(pre, observed_post, search_radius_um=1.0)
    corrected = observed_post - np.array(
        [offset["shift_z_um"], offset["shift_y_um"], offset["shift_x_um"]]
    )
    after = pair_puncta(pre, corrected, tolerance_um=0.30)

    assert without.count < 0.5 * len(pre)        # pairing has collapsed
    assert after.count > 0.95 * len(pre)         # and is restored
