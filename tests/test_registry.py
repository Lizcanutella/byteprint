from __future__ import annotations

import pytest

from byteprint.registry import PLUGIN_ENV_VAR, Registry, load_plugins


def test_a_registered_entry_can_be_resolved_by_name() -> None:
    registry: Registry[int] = Registry("widget")
    registry.register("small", 1)

    assert registry.resolve("small") == 1


def test_registering_reports_the_entry_back_so_it_can_decorate() -> None:
    registry: Registry[object] = Registry("widget")

    def build() -> str:
        return "built"

    assert registry.register("w", build) is build


def test_a_registry_reads_as_a_mapping() -> None:
    registry: Registry[int] = Registry("widget")
    registry.register("a", 1)
    registry.register("b", 2)

    assert sorted(registry) == ["a", "b"]
    assert registry["a"] == 1
    assert len(registry) == 2
    assert registry.names() == ["a", "b"]


def test_shadowing_a_registered_name_is_refused_by_default() -> None:
    """Two branches registering the same name must not depend on import order."""
    registry: Registry[int] = Registry("widget")
    registry.register("dup", 1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("dup", 2)


def test_shadowing_is_allowed_when_it_is_deliberate() -> None:
    registry: Registry[int] = Registry("widget")
    registry.register("dup", 1)
    registry.register("dup", 2, replace=True)

    assert registry.resolve("dup") == 2


def test_an_empty_name_is_refused() -> None:
    registry: Registry[int] = Registry("widget")

    with pytest.raises(ValueError, match="non-empty"):
        registry.register("", 1)


def test_resolving_an_unknown_name_lists_the_alternatives() -> None:
    registry: Registry[int] = Registry("widget")
    registry.register("small", 1)
    registry.register("large", 2)

    with pytest.raises(ValueError, match="large, small"):
        registry.resolve("medium")


def test_resolving_an_unknown_name_points_at_the_plugin_flag() -> None:
    registry: Registry[int] = Registry("widget")

    with pytest.raises(ValueError, match="--plugin"):
        registry.resolve("nope")


# -- plugin loading --------------------------------------------------------


def test_loading_a_plugin_imports_the_module() -> None:
    assert load_plugins("json") == ["json"]


def test_plugins_can_be_given_as_a_comma_separated_string() -> None:
    assert load_plugins("json,csv") == ["json", "csv"]


def test_an_unimportable_plugin_is_reported_by_name() -> None:
    with pytest.raises(ValueError, match="byteprint_no_such_plugin"):
        load_plugins("byteprint_no_such_plugin")


def test_plugins_default_to_the_environment_variable(monkeypatch) -> None:
    monkeypatch.setenv(PLUGIN_ENV_VAR, "json")

    assert load_plugins() == ["json"]


def test_no_plugins_is_not_an_error(monkeypatch) -> None:
    monkeypatch.delenv(PLUGIN_ENV_VAR, raising=False)

    assert load_plugins() == []
