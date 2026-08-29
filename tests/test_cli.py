from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from byteprint.cli import main

DIM = 16


class StubBackbone:
    """A backbone whose features expose the fixture's planted grid artifact."""

    name = "stub"
    dim = DIM

    def embed(self, crops):
        rows = []
        for crop in crops:
            plane = np.asarray(crop, dtype=np.float64).mean(axis=2)
            spectrum = np.abs(np.fft.rfft2(plane))
            rows.append(np.resize(np.log1p(spectrum).ravel()[:DIM], DIM))
        return np.stack(rows).astype(np.float32)


@pytest.fixture
def stub_factory():
    return lambda name, device, batch_size: StubBackbone()


def run(argv, factory=None) -> int:
    return main(argv, backbone_factory=factory) if factory else main(argv)


def test_fixture_command_builds_both_splits(tmp_path: Path) -> None:
    run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "6"])

    assert (tmp_path / "data" / "train" / "real").is_dir()
    assert (tmp_path / "data" / "test" / "fake").is_dir()


def test_fixture_command_creates_the_requested_number_of_images(tmp_path: Path) -> None:
    run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "6"])

    reals = list((tmp_path / "data" / "train" / "real").glob("*.png"))

    assert len(reals) == 6


def test_fixture_generators_land_in_their_own_folders(tmp_path: Path) -> None:
    run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "6"])

    names = {p.name for p in (tmp_path / "data" / "train" / "fake").iterdir()}

    assert len(names) >= 2


def test_extract_populates_a_cache(tmp_path: Path, stub_factory) -> None:
    run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "6"])

    code = run(
        [
            "extract",
            "--data", str(tmp_path / "data" / "train"),
            "--cache", str(tmp_path / "cache"),
            "--crop-size", "28", "--crops", "2",
        ],
        stub_factory,
    )

    assert code == 0
    assert (tmp_path / "cache" / "features.npy").exists()


def test_train_writes_a_probe(tmp_path: Path, stub_factory, capsys) -> None:
    _build_cache(tmp_path, stub_factory)

    code = run(["train", "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "probe.joblib")])

    assert code == 0
    assert (tmp_path / "probe.joblib").exists()
    assert "AUC" in capsys.readouterr().out


def test_eval_reports_per_generator_scores(tmp_path: Path, stub_factory, capsys) -> None:
    _build_cache(tmp_path, stub_factory)
    run(["train", "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "probe.joblib")])
    capsys.readouterr()

    run(["eval", "--cache", str(tmp_path / "cache"), "--probe", str(tmp_path / "probe.joblib")])

    assert "gridnet" in capsys.readouterr().out


def test_eval_can_break_results_down_by_laundering_spec(
    tmp_path: Path, stub_factory, capsys
) -> None:
    _build_cache(tmp_path, stub_factory)
    run(["train", "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "probe.joblib")])
    capsys.readouterr()

    run(
        ["eval", "--cache", str(tmp_path / "cache"),
         "--probe", str(tmp_path / "probe.joblib"), "--by-spec"]
    )

    assert "none" in capsys.readouterr().out


def test_leave_one_generator_out_reports_every_generator(
    tmp_path: Path, stub_factory, capsys
) -> None:
    _build_cache(tmp_path, stub_factory)

    code = run(["logo", "--cache", str(tmp_path / "cache")])

    out = capsys.readouterr().out
    assert code == 0
    assert "gridnet" in out and "ringnet" in out


def test_training_on_a_single_class_fails_with_a_clear_message(
    tmp_path: Path, stub_factory, capsys
) -> None:
    run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "6"])
    (tmp_path / "onesided" / "real").mkdir(parents=True)
    for image in (tmp_path / "data" / "train" / "real").glob("*.png"):
        (tmp_path / "onesided" / "real" / image.name).write_bytes(image.read_bytes())
    run(
        ["extract", "--data", str(tmp_path / "onesided"), "--cache", str(tmp_path / "c2"),
         "--crop-size", "28", "--crops", "1"],
        stub_factory,
    )
    capsys.readouterr()

    code = run(["train", "--cache", str(tmp_path / "c2"), "--out", str(tmp_path / "p2.joblib")])

    assert code != 0
    assert "both classes" in capsys.readouterr().err


def test_an_unknown_command_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        run(["frobnicate"])


def _build_cache(tmp_path: Path, factory) -> None:
    run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "24"])
    run(
        [
            "extract",
            "--data", str(tmp_path / "data" / "train"),
            "--cache", str(tmp_path / "cache"),
            "--crop-size", "28", "--crops", "2",
            "--specs", "none,jpeg:60",
        ],
        factory,
    )


# --- reconstruction expert + fusion -------------------------------------


def stub_recon_expert():
    """A ReconExpert over two toy autoencoders: one lossless, one low-pass."""
    import torch
    from byteprint.recon import Autoencoder, ReconExpert

    def lowpass(x: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.avg_pool2d(x, 4)
        return torch.nn.functional.interpolate(pooled, size=x.shape[-2:], mode="nearest")

    class L1:
        backend = "l1"

        def __call__(self, a, b):
            return (a - b).abs().mean(dim=(1, 2, 3))

    return ReconExpert(
        [Autoencoder("identity", lambda x: x), Autoencoder("lowpass", lowpass)],
        distance=L1(),
    )


@pytest.fixture
def recon_factory():
    return lambda ae_ids, device, batch_size: stub_recon_expert()


def _run(argv, *, backbone=None, recon=None) -> int:
    kwargs = {}
    if backbone:
        kwargs["backbone_factory"] = backbone
    if recon:
        kwargs["recon_factory"] = recon
    return main(argv, **kwargs)


def _build_both_caches(tmp_path: Path, stub_factory, recon_factory) -> None:
    _run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "24"])
    common = [
        "--data", str(tmp_path / "data" / "train"),
        "--crop-size", "28", "--crops", "2", "--specs", "none,jpeg:60",
    ]
    _run(["extract", *common, "--cache", str(tmp_path / "dino")], backbone=stub_factory)
    _run(
        ["extract", *common, "--cache", str(tmp_path / "recon"), "--expert", "recon"],
        recon=recon_factory,
    )


def test_extract_can_build_a_reconstruction_cache(tmp_path: Path, recon_factory) -> None:
    _run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "6"])

    code = _run(
        ["extract", "--data", str(tmp_path / "data" / "train"),
         "--cache", str(tmp_path / "recon"), "--expert", "recon",
         "--crop-size", "28", "--crops", "1"],
        recon=recon_factory,
    )

    assert code == 0
    assert np.load(tmp_path / "recon" / "features.npy").shape[1] == 2  # one column per AE


def test_the_reconstruction_cache_records_which_autoencoders_were_used(
    tmp_path: Path, recon_factory
) -> None:
    import json

    _run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "6"])
    _run(
        ["extract", "--data", str(tmp_path / "data" / "train"),
         "--cache", str(tmp_path / "recon"), "--expert", "recon",
         "--crop-size", "28", "--crops", "1"],
        recon=recon_factory,
    )

    config = json.loads((tmp_path / "recon" / "config.json").read_text())

    assert "identity" in config["backbone"] and "lowpass" in config["backbone"]


def test_fuse_writes_a_detector_and_reports_the_ablation(
    tmp_path: Path, stub_factory, recon_factory, capsys
) -> None:
    _build_both_caches(tmp_path, stub_factory, recon_factory)
    _run(["train", "--cache", str(tmp_path / "dino"), "--out", str(tmp_path / "probe.joblib")])
    capsys.readouterr()

    code = _run(
        ["fuse", "--dino-cache", str(tmp_path / "dino"), "--recon-cache", str(tmp_path / "recon"),
         "--probe", str(tmp_path / "probe.joblib"), "--out", str(tmp_path / "fused.joblib")]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert (tmp_path / "fused.joblib").exists()
    assert "probe only" in out and "recon only" in out and "fused" in out


def test_eval_scores_a_fused_detector(
    tmp_path: Path, stub_factory, recon_factory, capsys
) -> None:
    _build_both_caches(tmp_path, stub_factory, recon_factory)
    _run(["train", "--cache", str(tmp_path / "dino"), "--out", str(tmp_path / "probe.joblib")])
    _run(
        ["fuse", "--dino-cache", str(tmp_path / "dino"), "--recon-cache", str(tmp_path / "recon"),
         "--probe", str(tmp_path / "probe.joblib"), "--out", str(tmp_path / "fused.joblib")]
    )
    capsys.readouterr()

    code = _run(
        ["eval", "--cache", str(tmp_path / "dino"), "--recon-cache", str(tmp_path / "recon"),
         "--fused", str(tmp_path / "fused.joblib")]
    )

    assert code == 0
    assert "recon only" in capsys.readouterr().out


def test_fused_eval_can_break_down_by_laundering_spec(
    tmp_path: Path, stub_factory, recon_factory, capsys
) -> None:
    _build_both_caches(tmp_path, stub_factory, recon_factory)
    _run(["train", "--cache", str(tmp_path / "dino"), "--out", str(tmp_path / "probe.joblib")])
    _run(
        ["fuse", "--dino-cache", str(tmp_path / "dino"), "--recon-cache", str(tmp_path / "recon"),
         "--probe", str(tmp_path / "probe.joblib"), "--out", str(tmp_path / "fused.joblib")]
    )
    capsys.readouterr()

    _run(
        ["eval", "--cache", str(tmp_path / "dino"), "--recon-cache", str(tmp_path / "recon"),
         "--fused", str(tmp_path / "fused.joblib"), "--by-spec"]
    )

    assert "jpeg:60" in capsys.readouterr().out


def test_fusing_caches_with_no_shared_images_fails_clearly(
    tmp_path: Path, stub_factory, recon_factory, capsys
) -> None:
    _build_both_caches(tmp_path, stub_factory, recon_factory)
    _run(["train", "--cache", str(tmp_path / "dino"), "--out", str(tmp_path / "probe.joblib")])
    _run(["fixture", "--out", str(tmp_path / "other"), "--per-class", "6", "--seed", "9"])
    _run(
        ["extract", "--data", str(tmp_path / "other" / "train"),
         "--cache", str(tmp_path / "recon2"), "--expert", "recon",
         "--crop-size", "28", "--crops", "1"],
        recon=recon_factory,
    )
    capsys.readouterr()

    code = _run(
        ["fuse", "--dino-cache", str(tmp_path / "dino"), "--recon-cache", str(tmp_path / "recon2"),
         "--probe", str(tmp_path / "probe.joblib"), "--out", str(tmp_path / "f2.joblib")]
    )

    assert code != 0
    assert "no shared" in capsys.readouterr().err


def test_eval_without_a_fused_detector_still_requires_a_probe(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _run(["eval", "--cache", str(tmp_path / "dino")])


def test_an_unknown_autoencoder_id_is_reported(tmp_path: Path) -> None:
    _run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "6"])

    code = _run(
        ["extract", "--data", str(tmp_path / "data" / "train"),
         "--cache", str(tmp_path / "r"), "--expert", "recon", "--aes", "dall-e",
         "--crop-size", "28", "--crops", "1"]
    )

    assert code != 0


# --- swappable parts: registries and plugins -----------------------------


def test_list_reports_every_extension_point(capsys) -> None:
    assert run(["list"]) == 0

    out = capsys.readouterr().out
    for section in ("backbones", "heads", "crop modes", "autoencoders", "ladders"):
        assert section in out
    assert "dinov2_vits14" in out and "linear-svm" in out and "texture" in out


def test_list_marks_the_defaults(capsys) -> None:
    run(["list"])

    out = capsys.readouterr().out

    assert "dinov2_vits14            dim 384  <- default" in out


def test_train_accepts_an_alternative_head(tmp_path: Path, stub_factory, capsys) -> None:
    _build_cache(tmp_path, stub_factory)

    code = run(
        ["train", "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "svm.joblib"),
         "--head", "linear-svm"]
    )

    assert code == 0
    assert "linear-svm head" in capsys.readouterr().out
    assert (tmp_path / "svm.joblib").exists()


def test_the_trained_head_is_recorded_in_the_saved_probe(tmp_path: Path, stub_factory) -> None:
    from byteprint.probe import LinearProbe

    _build_cache(tmp_path, stub_factory)
    run(
        ["train", "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "mlp.joblib"),
         "--head", "mlp"]
    )

    assert LinearProbe.load(tmp_path / "mlp.joblib").config.head == "mlp"


def test_an_unknown_head_is_reported_with_the_alternatives(
    tmp_path: Path, stub_factory, capsys
) -> None:
    _build_cache(tmp_path, stub_factory)

    code = run(
        ["train", "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "x.joblib"),
         "--head", "transformer"]
    )

    assert code != 0
    assert "logreg" in capsys.readouterr().err


def test_an_unknown_backbone_is_reported_at_use_time(tmp_path: Path, capsys) -> None:
    """Not an argparse choice, so a --plugin backbone is as valid as a built-in one."""
    run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "4"])

    code = run(
        ["extract", "--data", str(tmp_path / "data" / "train"),
         "--cache", str(tmp_path / "c"), "--backbone", "resnet50",
         "--crop-size", "28", "--crops", "1"]
    )

    assert code != 0
    assert "dinov2_vits14" in capsys.readouterr().err


def test_an_unknown_crop_mode_is_reported(tmp_path: Path, stub_factory, capsys) -> None:
    run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "4"])

    code = run(
        ["extract", "--data", str(tmp_path / "data" / "train"),
         "--cache", str(tmp_path / "c"), "--crop-mode", "quadtree",
         "--crop-size", "28", "--crops", "1"],
        stub_factory,
    )

    assert code != 0
    assert "quadtree" in capsys.readouterr().err


def test_an_unimportable_plugin_is_reported(tmp_path: Path, capsys) -> None:
    code = run(["list", "--plugin", "byteprint_not_a_module"])

    assert code != 0
    assert "byteprint_not_a_module" in capsys.readouterr().err


def test_a_plugin_module_is_imported_before_the_command_runs(tmp_path: Path, capsys) -> None:
    """A registration in a --plugin module shows up in `byteprint list`."""
    plugin = tmp_path / "byteprint_test_plugin.py"
    plugin.write_text(
        "from byteprint.heads import register_head\n"
        "from sklearn.naive_bayes import GaussianNB\n"
        "@register_head('plugin-nb')\n"
        "def _build(config):\n"
        "    return GaussianNB()\n"
    )
    import sys as _sys

    _sys.path.insert(0, str(tmp_path))
    try:
        assert run(["list", "--plugin", "byteprint_test_plugin"]) == 0
        assert "plugin-nb" in capsys.readouterr().out
    finally:
        _sys.path.remove(str(tmp_path))
        _sys.modules.pop("byteprint_test_plugin", None)
        from byteprint.heads import HEADS

        HEADS._entries.pop("plugin-nb", None)


# --- the laundering ladder on the command line ---------------------------


def test_extract_can_build_every_rung_of_the_official_ladder(
    tmp_path: Path, stub_factory
) -> None:
    from byteprint.cache import EmbeddingStore, ExtractConfig
    from byteprint.launder import OFFICIAL_LADDER

    run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "4"])
    code = run(
        ["extract", "--data", str(tmp_path / "data" / "train"),
         "--cache", str(tmp_path / "ladder"), "--ladder", "official",
         "--crop-size", "28", "--crops", "1"],
        stub_factory,
    )

    assert code == 0
    config = ExtractConfig(
        backbone="stub", dim=DIM, crop_size=28, crops_per_image=1,
        crop_mode="texture", seed=0,
    )
    store = EmbeddingStore.open(tmp_path / "ladder", config)
    assert set(store.specs()) == set(OFFICIAL_LADDER)


def test_specs_ladder_stays_an_alias_for_the_official_list(
    tmp_path: Path, stub_factory
) -> None:
    from byteprint.cache import EmbeddingStore, ExtractConfig
    from byteprint.launder import OFFICIAL_LADDER

    run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "4"])
    run(
        ["extract", "--data", str(tmp_path / "data" / "train"),
         "--cache", str(tmp_path / "ladder"), "--specs", "ladder",
         "--crop-size", "28", "--crops", "1"],
        stub_factory,
    )

    config = ExtractConfig(
        backbone="stub", dim=DIM, crop_size=28, crops_per_image=1,
        crop_mode="texture", seed=0,
    )
    store = EmbeddingStore.open(tmp_path / "ladder", config)
    assert set(store.specs()) == set(OFFICIAL_LADDER)


def test_an_unknown_ladder_is_reported(tmp_path: Path, capsys) -> None:
    run(["fixture", "--out", str(tmp_path / "data"), "--per-class", "4"])

    code = run(
        ["extract", "--data", str(tmp_path / "data" / "train"),
         "--cache", str(tmp_path / "c"), "--ladder", "kitchen-sink",
         "--crop-size", "28", "--crops", "1"]
    )

    assert code != 0
    assert "kitchen-sink" in capsys.readouterr().err


# --- the deliverable: directory -> JSON ----------------------------------


def _images_dir(tmp_path: Path) -> Path:
    """A flat directory of images, the way a judge would hand one over."""
    import shutil

    run(["fixture", "--out", str(tmp_path / "fx"), "--per-class", "6"])
    out = tmp_path / "inbox"
    out.mkdir()
    for index, path in enumerate(sorted((tmp_path / "fx" / "test").rglob("*.png"))):
        shutil.copy(path, out / f"img_{index:03d}.png")
    return out


def _trained_probe(tmp_path: Path, stub_factory) -> Path:
    _build_cache(tmp_path, stub_factory)
    probe = tmp_path / "probe.joblib"
    run(["train", "--cache", str(tmp_path / "cache"), "--out", str(probe)])
    return probe


def test_score_writes_the_required_json_shape(tmp_path: Path, stub_factory) -> None:
    import json

    probe = _trained_probe(tmp_path, stub_factory)
    images = _images_dir(tmp_path)
    out = tmp_path / "predictions.json"

    code = run(["score", str(images), "--probe", str(probe), "--out", str(out)], stub_factory)

    assert code == 0
    payload = json.loads(out.read_text())
    assert len(payload) == len(list(images.glob("*.png")))
    for record in payload:
        assert set(record) == {"image_path", "pred"}
        assert 0.0 <= record["pred"] <= 1.0


def test_score_needs_no_cache_because_the_probe_carries_its_settings(
    tmp_path: Path, stub_factory
) -> None:
    """A judge gets a probe file and a directory. Nothing else should be required."""
    from byteprint.probe import LinearProbe

    probe_path = _trained_probe(tmp_path, stub_factory)

    assert LinearProbe.load(probe_path).extract_config is not None


def test_score_reports_relative_paths_on_request(tmp_path: Path, stub_factory) -> None:
    import json

    probe = _trained_probe(tmp_path, stub_factory)
    images = _images_dir(tmp_path)
    out = tmp_path / "p.json"

    run(["score", str(images), "--probe", str(probe), "--out", str(out), "--relative"],
        stub_factory)

    assert all(not r["image_path"].startswith("/") for r in json.loads(out.read_text()))


def test_score_on_an_empty_directory_is_an_error(tmp_path: Path, stub_factory, capsys) -> None:
    probe = _trained_probe(tmp_path, stub_factory)
    empty = tmp_path / "empty"
    empty.mkdir()

    code = run(["score", str(empty), "--probe", str(probe), "--out", str(tmp_path / "p.json")],
               stub_factory)

    assert code != 0
    assert "no images found" in capsys.readouterr().err


def test_score_on_a_missing_directory_is_reported(tmp_path: Path, stub_factory, capsys) -> None:
    probe = _trained_probe(tmp_path, stub_factory)

    code = run(["score", str(tmp_path / "nope"), "--probe", str(probe),
                "--out", str(tmp_path / "p.json")], stub_factory)

    assert code != 0
    assert "does not exist" in capsys.readouterr().err


def test_score_survives_a_corrupt_file_and_flags_it(tmp_path: Path, stub_factory) -> None:
    import json

    probe = _trained_probe(tmp_path, stub_factory)
    images = _images_dir(tmp_path)
    (images / "truncated.png").write_bytes(b"\x89PNG\r\n\x1a\n truncated")
    out = tmp_path / "p.json"

    code = run(["score", str(images), "--probe", str(probe), "--out", str(out)], stub_factory)

    assert code == 0
    payload = json.loads(out.read_text())
    broken = [r for r in payload if r["image_path"].endswith("truncated.png")]
    assert len(broken) == 1 and "error" in broken[0]


def test_predict_no_longer_requires_a_cache(tmp_path: Path, stub_factory, capsys) -> None:
    probe = _trained_probe(tmp_path, stub_factory)
    images = _images_dir(tmp_path)

    code = run(["predict", "--probe", str(probe), str(next(images.glob("*.png")))],
               stub_factory)

    assert code == 0
    assert "img_" in capsys.readouterr().out
