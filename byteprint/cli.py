"""Command line entry point.

    byteprint fixture --out data                      # synthetic smoke-test data
    byteprint extract --data data/train --cache cache/train --augment 3
    byteprint train   --cache cache/train --out runs/probe.joblib
    byteprint eval    --cache cache/test  --probe runs/probe.joblib --by-spec
    byteprint logo    --cache cache/train             # leave-one-generator-out
    byteprint predict --probe runs/probe.joblib IMAGE...
    byteprint list                                    # what is registered

The required deliverable interface -- an image directory in, a JSON file of
{image_path, pred} out:

    byteprint score IMAGE_DIR --probe runs/probe.joblib --out predictions.json

Swappable parts -- backbone, head (and so the training loss), crop strategy --
are named entries in a registry. `--plugin` imports a module so whatever it
registers becomes available, which is how a branch adds one without editing
this file:

    byteprint train --plugin myteam.heads --head my-head ...

Two experts, fused at the score level:

    byteprint extract --data data/train --cache cache/recon --expert recon
    byteprint fuse    --dino-cache cache/train --recon-cache cache/recon \
                   --probe runs/probe.joblib --out runs/fused.joblib
    byteprint eval    --cache cache/test --recon-cache cache/recon-test \
                   --fused runs/fused.joblib --by-spec
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from byteprint import fixture
from byteprint.backbone import BACKBONES, DEFAULT_BACKBONE, load_backbone, resolve_device
from byteprint.cache import EmbeddingStore, ExtractConfig, StaleCacheError
from byteprint.crops import CROP_MODES, DEFAULT_CROP_MODE
from byteprint.data import scan_split
from byteprint.heads import DEFAULT_HEAD, HEADS
from byteprint.launder import LADDERS, NO_OP, ladder
from byteprint.metrics import evaluate
from byteprint.pipeline import extract, load_image
from byteprint.fusion import FusedDetector, join_caches
from byteprint.probe import LinearProbe, ProbeConfig
from byteprint.recon import AUTOENCODERS, DEFAULT_AUTOENCODERS, aeroblade_score, load_recon_expert
from byteprint.score import UNSCORABLE, ProbeScorer, score_directory, write_predictions
from byteprint.registry import load_plugins

DEFAULT_FPR_TARGETS = (0.01, 0.001)


# -- helpers ---------------------------------------------------------------


def _split_indices(n: int, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    cut = max(1, int(round(n * holdout)))
    return order[cut:], order[:cut]


def _load_store(cache: Path, *, rebuild: bool = False) -> EmbeddingStore:
    import json

    config_path = Path(cache) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"no cache at {cache}; run `byteprint extract` first")
    config = ExtractConfig(**json.loads(config_path.read_text()))
    return EmbeddingStore.open(cache, config, rebuild=rebuild)


def _print_ablation(labels, parts, *, generators=None, specs=None, title: str) -> None:
    """Print probe-only / recon-only / fused side by side."""
    print(title)
    width = max(len(name) for name in _ABLATION_LABELS.values())
    for key, name in _ABLATION_LABELS.items():
        report = evaluate(labels, parts[key], fpr_targets=(0.01,))
        print(
            f"  {name.ljust(width)}   AUC {report.auc:.4f}   "
            f"TPR@1%FPR {report.tpr_at_fpr[0.01]:.4f}"
        )

    if generators is not None:
        fused = evaluate(labels, parts["fused"], generators=generators, fpr_targets=(0.01,))
        if fused.per_generator:
            print("\n  fused, per generator")
            for name in sorted(fused.per_generator, key=lambda k: fused.per_generator[k].auc):
                score = fused.per_generator[name]
                print(f"    {name:<16} AUC {score.auc:.4f}   n {score.n_fake}")

    if specs is not None:
        print("\n  fused, laundering ladder")
        specs_array = np.asarray(specs)
        for spec in sorted(set(specs_array.tolist()), key=lambda x: (x != NO_OP, x)):
            mask = specs_array == spec
            if len(np.unique(np.asarray(labels)[mask])) < 2:
                print(f"    {spec:<24} (needs both classes)")
                continue
            rung = evaluate(np.asarray(labels)[mask], parts["fused"][mask], fpr_targets=(0.01,))
            print(
                f"    {spec:<24} AUC {rung.auc:.4f}   "
                f"TPR@1%FPR {rung.tpr_at_fpr[0.01]:.4f}"
            )


_ABLATION_LABELS = {"probe": "probe only", "recon": "recon only", "fused": "fused"}


# -- commands --------------------------------------------------------------


def cmd_fixture(args: argparse.Namespace) -> int:
    root = fixture.build(args.out, per_class=args.per_class, size=args.size, seed=args.seed)
    print(f"wrote synthetic dataset to {root}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Everything that can be named on the command line, and where it came from."""
    print("backbones      (--backbone)")
    for name in BACKBONES.names():
        marker = "  <- default" if name == DEFAULT_BACKBONE else ""
        print(f"  {name:<24} dim {BACKBONES[name].dim}{marker}")

    print("\nheads          (--head)")
    for name in HEADS.names():
        marker = "  <- default" if name == DEFAULT_HEAD else ""
        print(f"  {name}{marker}")

    print("\ncrop modes     (--crop-mode)")
    for name in CROP_MODES.names():
        marker = "  <- default" if name == DEFAULT_CROP_MODE else ""
        print(f"  {name}{marker}")

    print("\nautoencoders   (--aes)")
    for name in sorted(AUTOENCODERS):
        marker = "  <- default" if name in DEFAULT_AUTOENCODERS else ""
        print(f"  {name}{marker}")

    print("\nladders        (--ladder)")
    for name in sorted(LADDERS):
        print(f"  {name:<24} {len(LADDERS[name])} rungs: {' '.join(LADDERS[name])}")
    return 0


def cmd_extract(args: argparse.Namespace, backbone_factory, recon_factory) -> int:
    samples = scan_split(args.data)
    if not samples:
        print(f"no images found under {args.data}", file=sys.stderr)
        return 1

    # Resolve the crop mode before loading any weights: an unknown one would
    # otherwise fail once per image inside extract() and look like a bad dataset.
    CROP_MODES.resolve(args.crop_mode)

    if args.expert == "recon":
        ae_ids = [a.strip() for a in args.aes.split(",") if a.strip()]
        backbone = recon_factory(ae_ids, args.device, args.batch_size)
    else:
        backbone = backbone_factory(args.backbone, args.device, args.batch_size)

    config = ExtractConfig(
        backbone=backbone.name,
        dim=backbone.dim,
        crop_size=args.crop_size,
        crops_per_image=args.crops,
        crop_mode=args.crop_mode,
        seed=args.seed,
    )

    try:
        store = EmbeddingStore.open(args.cache, config, rebuild=args.rebuild)
    except StaleCacheError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    specs = tuple(s.strip() for s in args.specs.split(",")) if args.specs else (NO_OP,)
    stats = extract(
        samples,
        backbone=backbone,
        store=store,
        specs=specs,
        augment=args.augment,
        seed=args.seed,
        workers=args.workers,
    )
    store.flush()
    print(f"{len(samples)} images -> {stats.render()} ({len(store)} rows in {args.cache})")
    if stats.added == 0 and stats.failed:
        print(
            f"every image failed to extract -- run with -v to see why "
            f"({stats.failed} failures)",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    store = _load_store(args.cache)
    features, labels = store.matrix(), store.labels()
    if len(np.unique(labels)) < 2:
        print("training needs both classes present; got only one", file=sys.stderr)
        return 1

    fit_idx, calib_idx = _split_indices(len(labels), args.calib_fraction, args.seed)
    probe = LinearProbe(ProbeConfig(head=args.head, C=args.C, seed=args.seed))
    probe.fit(features[fit_idx], labels[fit_idx])
    probe.calibrate(features[calib_idx], labels[calib_idx], target_fpr=args.target_fpr)
    # Travel with the settings the features were built under, so `score` and
    # `predict` need nothing but the probe file itself.
    probe.extract_config = store.config
    probe.save(args.out)

    generators = np.asarray(store.generators())
    report = evaluate(
        labels[calib_idx],
        probe.score(features[calib_idx]),
        generators=generators[calib_idx].tolist(),
        fpr_targets=DEFAULT_FPR_TARGETS,
        label="calibration split",
    )
    print(report.render())
    print(
        f"\n{args.head} head, threshold {probe.threshold:.6f} "
        f"at {args.target_fpr:.1%} FPR -> {args.out}"
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    store = _load_store(args.cache)
    probe = LinearProbe.load(args.probe)
    features, labels = store.matrix(), store.labels()
    generators = store.generators()
    scores = probe.score(features)

    print(
        evaluate(
            labels, scores, generators=generators,
            fpr_targets=DEFAULT_FPR_TARGETS, label=str(args.cache),
        ).render()
    )

    if args.by_spec:
        print("\nlaundering ladder")
        specs = np.asarray(store.specs())
        for spec in sorted(set(specs.tolist()), key=lambda s: (s != NO_OP, s)):
            mask = specs == spec
            if len(np.unique(labels[mask])) < 2:
                print(f"  {spec:<24} (needs both classes)")
                continue
            rung = evaluate(labels[mask], scores[mask], fpr_targets=(0.01,))
            print(f"  {spec:<24} AUC {rung.auc:.4f}   TPR@1%FPR {rung.tpr_at_fpr[0.01]:.4f}")
    return 0


def cmd_logo(args: argparse.Namespace) -> int:
    """Leave-one-generator-out: the only protocol that measures what matters."""
    store = _load_store(args.cache)
    features, labels = store.matrix(), store.labels()
    generators = np.asarray(store.generators())
    held_out = sorted(set(generators[labels == 1].tolist()))

    print(f"leave-one-generator-out over {len(held_out)} generators\n")
    aucs = []
    for name in held_out:
        train_mask = generators != name
        test_mask = (generators == name) | (labels == 0)
        if len(np.unique(labels[train_mask])) < 2:
            print(f"  {name:<16} skipped (no fakes left to train on)")
            continue

        probe = LinearProbe(ProbeConfig(head=args.head, C=args.C, seed=args.seed))
        probe.fit(features[train_mask], labels[train_mask])
        report = evaluate(
            labels[test_mask], probe.score(features[test_mask]), fpr_targets=(0.01,)
        )
        aucs.append(report.auc)
        print(
            f"  held out {name:<16} AUC {report.auc:.4f}   "
            f"TPR@1%FPR {report.tpr_at_fpr[0.01]:.4f}"
        )

    if aucs:
        print(f"\n  mean unseen-generator AUC {float(np.mean(aucs)):.4f}")
    return 0


def cmd_fuse(args: argparse.Namespace) -> int:
    joined = join_caches(_load_store(args.dino_cache), _load_store(args.recon_cache))
    if joined.n == 0:
        print(
            f"no shared images between {args.dino_cache} and {args.recon_cache}; "
            "both caches must be extracted from the same split",
            file=sys.stderr,
        )
        return 1
    if len(np.unique(joined.labels)) < 2:
        print("fusion needs both classes present; got only one", file=sys.stderr)
        return 1

    fit_idx, calib_idx = _split_indices(joined.n, args.calib_fraction, args.seed)
    detector = FusedDetector(LinearProbe.load(args.probe), seed=args.seed)
    detector.fit(joined.dino[fit_idx], joined.recon[fit_idx], joined.labels[fit_idx])
    detector.calibrate(
        joined.dino[calib_idx],
        joined.recon[calib_idx],
        joined.labels[calib_idx],
        target_fpr=args.target_fpr,
    )
    detector.save(args.out)

    parts = detector.component_scores(joined.dino[calib_idx], joined.recon[calib_idx])
    _print_ablation(
        joined.labels[calib_idx],
        parts,
        generators=[joined.generators[i] for i in calib_idx],
        title=f"calibration split ({len(calib_idx)} of {joined.n} joined rows)",
    )
    print(f"\nthreshold {detector.threshold:.6f} at {args.target_fpr:.1%} FPR -> {args.out}")
    return 0


def cmd_eval_fused(args: argparse.Namespace) -> int:
    joined = join_caches(_load_store(args.cache), _load_store(args.recon_cache))
    if joined.n == 0:
        print(
            f"no shared images between {args.cache} and {args.recon_cache}",
            file=sys.stderr,
        )
        return 1

    detector = FusedDetector.load(args.fused)
    parts = detector.component_scores(joined.dino, joined.recon)
    _print_ablation(
        joined.labels,
        parts,
        generators=joined.generators,
        specs=joined.specs if args.by_spec else None,
        title=f"{args.cache} + {args.recon_cache} ({joined.n} joined rows)",
    )
    return 0


def _scorer_for(probe: LinearProbe, args: argparse.Namespace, backbone_factory) -> ProbeScorer:
    """Rebuild the exact extraction setup the probe was trained under."""
    config = probe.extract_config
    if config is None:
        raise ValueError(
            "this probe predates self-contained extraction settings; retrain it with "
            "`byteprint train`, or pass --cache to borrow a cache's settings"
        )
    backbone = backbone_factory(config.backbone, args.device, args.batch_size)
    return ProbeScorer(backbone=backbone, probe=probe, config=config)


def cmd_score(args: argparse.Namespace, backbone_factory) -> int:
    """The deliverable: a directory of images -> a JSON file of {image_path, pred}."""
    probe = LinearProbe.load(args.probe)

    if args.cache:  # explicit override, e.g. scoring with a different crop size
        probe.extract_config = _load_store(args.cache).config

    scorer = _scorer_for(probe, args, backbone_factory)
    predictions = score_directory(
        args.images,
        scorer=scorer,
        relative=args.relative,
        chunk_size=args.chunk_size,
        strict=args.strict,
    )
    if not predictions:
        print(f"no images found under {args.images}", file=sys.stderr)
        return 1

    write_predictions(predictions, args.out)

    failed = [p for p in predictions if p.error is not None]
    config = probe.extract_config
    print(
        f"{len(predictions)} images -> {args.out}\n"
        f"  {config.backbone}, {config.crops_per_image}x{config.crop_size}px "
        f"{config.crop_mode} crops, threshold {probe.threshold:.4f}"
        + (f" at {probe.target_fpr:.1%} FPR" if probe.target_fpr else "")
    )
    if failed:
        print(f"  {len(failed)} unreadable, scored {UNSCORABLE} -- see the error field",
              file=sys.stderr)
    return 0


def cmd_predict(args: argparse.Namespace, backbone_factory) -> int:
    from byteprint.crops import select_crops

    probe = LinearProbe.load(args.probe)
    if args.cache:
        probe.extract_config = _load_store(args.cache).config
    store_config = probe.extract_config
    if store_config is None:
        print("this probe carries no extraction settings; retrain it or pass --cache",
              file=sys.stderr)
        return 1

    backbone = backbone_factory(store_config.backbone, args.device, args.batch_size)

    for path in args.images:
        crops = select_crops(
            load_image(path),
            crop_size=store_config.crop_size,
            top_k=store_config.crops_per_image,
            mode=store_config.crop_mode,
            seed=store_config.seed,
        )
        pooled = backbone.embed(crops).mean(axis=0, keepdims=True)
        score = float(probe.score(pooled)[0])
        verdict = "synthetic" if score >= probe.threshold else "real"
        print(f"{score:.4f}  {verdict:<10} {path}")
    return 0


# -- argument parsing ------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="byteprint",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log extraction progress")

    # Every subcommand takes --plugin, so a module that registers a backbone,
    # head or crop mode can be pulled in wherever it is needed.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--plugin", action="append", metavar="MODULE",
        help="import a module so its registrations apply; repeatable. "
             "Also read from the BYTEPRINT_PLUGINS environment variable.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", parents=[common], help="show registered backbones, heads and modes")

    fix = sub.add_parser("fixture", parents=[common], help="write a synthetic smoke-test dataset")
    fix.add_argument("--out", default="data", type=Path)
    fix.add_argument("--per-class", type=int, default=64)
    fix.add_argument("--size", type=int, default=96)
    fix.add_argument("--seed", type=int, default=0)

    ext = sub.add_parser("extract", parents=[common], help="embed a split into a cache")
    ext.add_argument("--data", required=True, type=Path, help="split directory (real/ + fake/)")
    ext.add_argument("--cache", required=True, type=Path)
    ext.add_argument(
        "--expert", default="dinov2", choices=["dinov2", "recon"],
        help="dinov2 = frozen-feature probe; recon = AEROBLADE reconstruction error",
    )
    ext.add_argument(
        "--backbone", default=DEFAULT_BACKBONE,
        help=f"registered backbone; built in: {', '.join(BACKBONES.names())} (`byteprint list`)",
    )
    ext.add_argument(
        "--aes", default=",".join(DEFAULT_AUTOENCODERS),
        help=f"--expert recon: comma-separated autoencoders from {sorted(AUTOENCODERS)}",
    )
    ext.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ext.add_argument("--batch-size", type=int, default=16)
    ext.add_argument("--crop-size", type=int, default=224, help="must be a multiple of 14")
    ext.add_argument("--crops", type=int, default=4, help="crops kept per image")
    ext.add_argument(
        "--crop-mode", default=DEFAULT_CROP_MODE,
        help=f"crop strategy; built in: {', '.join(CROP_MODES.names())}",
    )
    ext.add_argument("--augment", type=int, default=0, help="random laundering views per image")
    ext.add_argument(
        "--ladder", default="", metavar="NAME",
        help=f"extract every rung of a named ladder: {', '.join(sorted(LADDERS))}",
    )
    ext.add_argument("--specs", default="", help="comma-separated laundering specs")
    ext.add_argument(
        "--workers",
        type=int,
        default=1,
        help="threads decoding, laundering and cropping ahead of the backbone. "
        "The cache is byte-identical whatever this is set to; only the wall "
        "clock changes. About 4 is the sweet spot on full-size photographs "
        "(~2.2x); more contends on the GIL, and on small images the handover "
        "costs more than the work (default: %(default)s)",
    )
    ext.add_argument("--rebuild", action="store_true", help="discard a stale cache")
    ext.add_argument("--seed", type=int, default=0)

    tr = sub.add_parser("train", parents=[common], help="fit and calibrate the linear probe")
    tr.add_argument("--cache", required=True, type=Path)
    tr.add_argument("--out", required=True, type=Path)
    tr.add_argument(
        "--head", default=DEFAULT_HEAD,
        help=f"training objective; built in: {', '.join(HEADS.names())}",
    )
    tr.add_argument("--C", type=float, default=1.0, help="inverse L2 regularisation strength")
    tr.add_argument("--calib-fraction", type=float, default=0.2)
    tr.add_argument("--target-fpr", type=float, default=0.01)
    tr.add_argument("--seed", type=int, default=0)

    ev = sub.add_parser("eval", parents=[common], help="score a cache with a trained probe")
    ev.add_argument("--cache", required=True, type=Path)
    ev.add_argument("--probe", type=Path, help="required unless --fused is given")
    ev.add_argument("--recon-cache", type=Path, help="reconstruction cache for the same split")
    ev.add_argument("--fused", type=Path, help="a detector written by `byteprint fuse`")
    ev.add_argument("--by-spec", action="store_true", help="break results down by laundering rung")

    lo = sub.add_parser("logo", parents=[common], help="leave-one-generator-out over a cache")
    lo.add_argument("--cache", required=True, type=Path)
    lo.add_argument("--head", default=DEFAULT_HEAD)
    lo.add_argument("--C", type=float, default=1.0)
    lo.add_argument("--seed", type=int, default=0)

    fu = sub.add_parser("fuse", parents=[common], help="fit a score-level fusion of the two experts")
    fu.add_argument("--dino-cache", required=True, type=Path)
    fu.add_argument("--recon-cache", required=True, type=Path)
    fu.add_argument("--probe", required=True, type=Path, help="a probe from `byteprint train`")
    fu.add_argument("--out", required=True, type=Path)
    fu.add_argument("--calib-fraction", type=float, default=0.2)
    fu.add_argument("--target-fpr", type=float, default=0.01)
    fu.add_argument("--seed", type=int, default=0)

    sc = sub.add_parser(
        "score", parents=[common],
        help="THE DELIVERABLE: an image directory -> a JSON file of {image_path, pred}",
    )
    sc.add_argument("images", type=Path, metavar="IMAGE_DIR", help="directory to score")
    sc.add_argument("--probe", required=True, type=Path, help="a probe from `byteprint train`")
    sc.add_argument("--out", default=Path("predictions.json"), type=Path)
    sc.add_argument(
        "--relative", action="store_true",
        help="report paths relative to IMAGE_DIR rather than absolute",
    )
    sc.add_argument("--chunk-size", type=int, default=8, help="images embedded per batch")
    sc.add_argument(
        "--strict", action="store_true",
        help="fail on the first unreadable image instead of scoring it 0.5",
    )
    sc.add_argument(
        "--cache", type=Path,
        help="override the probe's own extraction settings with a cache's",
    )
    sc.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    sc.add_argument("--batch-size", type=int, default=16)

    pr = sub.add_parser("predict", parents=[common], help="score individual image files")
    pr.add_argument("--probe", required=True, type=Path)
    pr.add_argument("--cache", type=Path, help="override the probe's extraction settings")
    pr.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    pr.add_argument("--batch-size", type=int, default=16)
    pr.add_argument("images", nargs="+", type=Path)

    return parser


def _default_factory(name: str, device: str, batch_size: int):
    return load_backbone(name, device=device, batch_size=batch_size)


def _default_recon_factory(ae_ids, device: str, batch_size: int):
    return load_recon_expert(ae_ids, device=device, batch_size=batch_size)


def main(
    argv: list[str] | None = None,
    *,
    backbone_factory=_default_factory,
    recon_factory=_default_recon_factory,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval":
        if args.fused and not args.recon_cache:
            parser.error("--fused also needs --recon-cache")
        if not args.fused and not args.probe:
            parser.error("eval needs --probe, or --fused with --recon-cache")
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    try:
        load_plugins()  # BYTEPRINT_PLUGINS, so a cluster job sets it once
        load_plugins(getattr(args, "plugin", None) or [])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.command == "extract":
        # `--specs ladder` stays as an alias for the official list.
        if args.specs == "ladder":
            args.ladder, args.specs = "official", ""
        if args.ladder:
            try:
                args.specs = ",".join(ladder(args.ladder))
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1

    try:
        if args.command == "list":
            return cmd_list(args)
        if args.command == "fixture":
            return cmd_fixture(args)
        if args.command == "extract":
            return cmd_extract(args, backbone_factory, recon_factory)
        if args.command == "train":
            return cmd_train(args)
        if args.command == "eval":
            return cmd_eval_fused(args) if args.fused else cmd_eval(args)
        if args.command == "fuse":
            return cmd_fuse(args)
        if args.command == "logo":
            return cmd_logo(args)
        if args.command == "score":
            return cmd_score(args, backbone_factory)
        if args.command == "predict":
            return cmd_predict(args, backbone_factory)
    except (FileNotFoundError, StaleCacheError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
