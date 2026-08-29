"""Named extension points.

Everything a teammate is likely to want to swap -- the backbone, the
classification head and its loss, the crop-sampling strategy -- lives in a
registry rather than in an ``if`` ladder inside the code that uses it. Adding
one means writing a function in your own module and decorating it; it means
never editing a file someone else is also editing, which is the whole point
when six people are on six branches.

    from byteprint.backbone import register_backbone

    @register_backbone("siglip2_so400m", dim=1152, patch_size=14)
    def _build():
        import timm
        return timm.create_model("vit_so400m_patch14_siglip_384", pretrained=True, num_classes=0)

Then point the CLI at the module that defines it:

    byteprint extract --plugin myteam.backbones --backbone siglip2_so400m ...

Registries validate at *use* time, not at argument-parse time, so a plugin
loaded from the command line is as first-class as one shipped in this package.
"""

from __future__ import annotations

import importlib
import os
from typing import Iterator, Mapping, TypeVar

T = TypeVar("T")

PLUGIN_ENV_VAR = "BYTEPRINT_PLUGINS"


class Registry(Mapping[str, T]):
    """A name -> entry table, readable as a plain mapping."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str, entry: T, *, replace: bool = False) -> T:
        """Add an entry. Refuses to shadow an existing name unless asked to.

        Silent shadowing is the failure mode this guards against: two branches
        register ``"probe"``, import order decides which one runs, and the
        results are irreproducible in a way that takes a day to find.
        """
        if not name:
            raise ValueError(f"a {self.kind} needs a non-empty name")
        if name in self._entries and not replace:
            raise ValueError(
                f"{self.kind} {name!r} is already registered; "
                f"pass replace=True if shadowing it is deliberate"
            )
        self._entries[name] = entry
        return entry

    def resolve(self, name: str) -> T:
        """Look up an entry, reporting the alternatives when it is missing."""
        if name not in self._entries:
            raise ValueError(
                f"unknown {self.kind} {name!r}; registered: {', '.join(self.names()) or '(none)'}"
                f" -- a custom one needs --plugin to import the module that registers it"
            )
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __getitem__(self, name: str) -> T:
        return self._entries[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()})"


def load_plugins(modules: str | list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Import modules so their registrations run. Returns what was imported.

    Takes an explicit list, a comma-separated string, or -- when given nothing
    -- whatever ``BYTEPRINT_PLUGINS`` names, so a cluster job can set the variable
    once instead of threading a flag through every command.
    """
    if modules is None:
        modules = os.environ.get(PLUGIN_ENV_VAR, "")
    if isinstance(modules, str):
        modules = [m.strip() for m in modules.split(",") if m.strip()]

    imported = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise ValueError(f"could not import plugin module {module!r}: {exc}") from exc
        imported.append(module)
    return imported
