"""The repo is a public deliverable. Nothing here may identify our environment.

This is enforcement, not a convention. Cluster hostnames, node inventories,
logins and absolute home paths are useful in local notes and are a liability in
a public repository, so a pattern that leaks back into a committed file fails
the build the same way a broken function would.

Local notes live in `CLUSTER.local.md`, which `.gitignore` excludes.

Scope: every file git would include in a commit -- tracked files plus untracked
files that are not ignored -- so a leak is caught *before* it is committed
rather than after. This file is excluded from its own scan, since it necessarily
contains the patterns it looks for.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Each entry is (regex, what it would reveal). Case-insensitive unless noted.
FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"cinaps", "the cluster's name"),
    (r"workdir2?/", "the cluster's filesystem layout"),
    (r"johan|boscher", "a teammate's login"),
    (r"\bnode\d{1,2}\b", "a specific compute node"),
    (r"GV100|A6000|RTX\s*PRO\s*6000", "the node inventory"),
    (r"paris-?saclay", "the institution"),
    (r"LMO-CPU", "the SLURM partition"),
    (r"ProxyJump|ssh_config|ssh -F", "SSH access details"),
    (r"/home/[a-z0-9_.-]+/", "an absolute path under someone's home directory"),
]

# Generic scheduler vocabulary is deliberately allowed: saying "a SLURM cluster,
# 2x 48 GB GPUs" states the compute budget, which the Feasibility criterion asks
# for, without naming the machine.
ALLOWED_ANYWHERE = {"slurm", "sbatch", "srun"}

SELF = Path(__file__).name


def _git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.skip(f"not a git repository, or git unavailable: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line]


def publishable_files() -> list[Path]:
    """Everything git would put in a commit: tracked, plus untracked-not-ignored."""
    names = _git("ls-files", "-co", "--exclude-standard")
    return [REPO_ROOT / name for name in names if Path(name).name != SELF]


def test_git_reports_at_least_the_package_and_tests() -> None:
    """Guard the guard: an empty file list would make every scan below vacuous."""
    names = {p.name for p in publishable_files()}

    assert "pyproject.toml" in names
    assert any(name.startswith("test_") for name in names)


@pytest.mark.parametrize("pattern,reveals", FORBIDDEN_PATTERNS)
def test_no_publishable_file_reveals_our_environment(pattern: str, reveals: str) -> None:
    compiled = re.compile(pattern, re.IGNORECASE)
    offenders = []

    for path in publishable_files():
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore")
        for number, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()[:90]}")

    assert not offenders, (
        f"these files would be published and reveal {reveals}:\n  "
        + "\n  ".join(offenders)
        + f"\n\nMove it to CLUSTER.local.md (gitignored). Pattern: {pattern!r}"
    )


def test_local_only_notes_are_not_publishable() -> None:
    """CLUSTER.local.md and friends must never appear in a commit."""
    published = {p.name for p in publishable_files()}

    assert not [name for name in published if name.endswith(".local.md")]


def test_claude_directory_is_not_publishable() -> None:
    """.claude/ carries environment-specific notes and stays local.

    Judged relative to the repo root: a git worktree is conventionally created
    *under* `.claude/worktrees/`, so an absolute-path check would fail every
    file in it and turn a real guard into noise everyone learns to ignore.
    """
    published = [
        p for p in publishable_files() if ".claude" in p.relative_to(REPO_ROOT).parts
    ]

    assert not published


def test_the_local_notes_file_is_ignored_by_git() -> None:
    """A gitignore rule that stops matching is a silent hole."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", "CLUSTER.local.md"],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    if result.returncode == 128:
        pytest.skip("not a git repository")

    assert result.returncode == 0, "CLUSTER.local.md is no longer gitignored"


def test_generic_scheduler_words_stay_allowed() -> None:
    """Documenting the compute budget must not be collateral damage."""
    for word in ALLOWED_ANYWHERE:
        assert not any(re.search(pattern, word, re.IGNORECASE)
                       for pattern, _ in FORBIDDEN_PATTERNS), (
            f"{word!r} is generic vocabulary but a forbidden pattern matches it"
        )
