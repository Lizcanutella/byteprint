from __future__ import annotations

import numpy as np
import pytest
import torch

from byteprint.recon import Autoencoder, ReconExpert, aeroblade_score, load_recon_expert


class FakeDistance:
    """Mean absolute difference, so expectations are hand-checkable."""

    backend = "fake"

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).abs().mean(dim=(1, 2, 3))


def identity_ae(name: str = "perfect") -> Autoencoder:
    return Autoencoder(name=name, reconstruct=lambda x: x)


def darkening_ae(name: str = "lossy", amount: float = 0.5) -> Autoencoder:
    return Autoencoder(name=name, reconstruct=lambda x: x - amount)


def crops(n: int, size: int = 32) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8) for _ in range(n)]


def test_the_expert_width_is_the_number_of_autoencoders() -> None:
    expert = ReconExpert([identity_ae(), darkening_ae()], distance=FakeDistance())

    assert expert.dim == 2


def test_embedding_returns_one_distance_per_crop_per_autoencoder() -> None:
    expert = ReconExpert([identity_ae(), darkening_ae()], distance=FakeDistance())

    assert expert.embed(crops(4)).shape == (4, 2)


def test_a_perfect_autoencoder_reports_zero_distance() -> None:
    expert = ReconExpert([identity_ae()], distance=FakeDistance())

    assert expert.embed(crops(3))[:, 0] == pytest.approx(0.0, abs=1e-6)


def test_a_lossy_autoencoder_reports_the_distortion_it_introduced() -> None:
    expert = ReconExpert([darkening_ae(amount=0.5)], distance=FakeDistance())

    assert expert.embed(crops(2))[:, 0] == pytest.approx(0.5, abs=1e-6)


def test_distance_columns_follow_the_autoencoder_order() -> None:
    expert = ReconExpert(
        [darkening_ae("lossy", 0.5), identity_ae("perfect")], distance=FakeDistance()
    )

    row = expert.embed(crops(1))[0]

    assert row[0] > row[1]
    assert expert.ae_names == ["lossy", "perfect"]


def test_crops_are_processed_in_batches() -> None:
    seen: list[int] = []

    def counting(x: torch.Tensor) -> torch.Tensor:
        seen.append(x.shape[0])
        return x

    expert = ReconExpert(
        [Autoencoder("counting", counting)], distance=FakeDistance(), batch_size=2
    )
    expert.embed(crops(5))

    assert seen == [2, 2, 1]


def test_embedding_no_crops_yields_an_empty_matrix_of_the_right_width() -> None:
    expert = ReconExpert([identity_ae(), darkening_ae()], distance=FakeDistance())

    assert expert.embed([]).shape == (0, 2)


def test_pixels_reach_the_autoencoder_in_the_minus_one_to_one_range() -> None:
    seen: list[torch.Tensor] = []

    def recorder(x: torch.Tensor) -> torch.Tensor:
        seen.append(x.clone())
        return x

    expert = ReconExpert([Autoencoder("rec", recorder)], distance=FakeDistance())
    expert.embed([np.zeros((32, 32, 3), dtype=np.uint8), np.full((32, 32, 3), 255, dtype=np.uint8)])

    assert seen[0].min() == pytest.approx(-1.0)
    assert seen[0].max() == pytest.approx(1.0)


def test_the_expert_name_records_which_autoencoders_were_used() -> None:
    expert = ReconExpert([identity_ae("sd15"), darkening_ae("sdxl")], distance=FakeDistance())

    assert "sd15" in expert.name and "sdxl" in expert.name


def test_the_score_negates_the_smallest_reconstruction_error() -> None:
    # AEROBLADE: generated images reconstruct better, so a small error is evidence of synthesis.
    distances = np.array([[0.4, 0.1], [0.9, 0.8]])

    assert aeroblade_score(distances) == pytest.approx([-0.1, -0.8])


def test_the_score_ranks_a_well_reconstructed_image_above_a_poorly_reconstructed_one() -> None:
    distances = np.array([[0.05, 0.9], [0.7, 0.8]])

    scores = aeroblade_score(distances)

    assert scores[0] > scores[1]


def test_the_score_works_for_a_single_autoencoder() -> None:
    assert aeroblade_score(np.array([[0.3], [0.1]])) == pytest.approx([-0.3, -0.1])


def test_scoring_an_empty_matrix_yields_an_empty_vector() -> None:
    assert aeroblade_score(np.zeros((0, 3))).shape == (0,)


def test_an_unknown_autoencoder_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="dall-e"):
        load_recon_expert(["dall-e"], device="cpu")


def test_requesting_no_autoencoders_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ReconExpert([], distance=FakeDistance())
