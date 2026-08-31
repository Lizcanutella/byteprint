"""Ordered, bounded prefetching.

Extraction is CPU-bound on decoding and crop selection while the GPU idles, but
the cache is an ordered append-only file and the laundering draws are
sequential. So the useful shape is: prepare crops on a pool, consume them *in
task order*, and never run so far ahead that a corpus of decoded images piles up
in memory.
"""
from __future__ import annotations

import threading
import time

import pytest

from byteprint.prefetch import ordered_prefetch


def test_results_come_back_in_task_order_even_when_workers_finish_out_of_order() -> None:
    def work(task: int) -> int:
        time.sleep(0.05 if task == 0 else 0.0)  # first task finishes last
        return task * 10

    out = [future.result() for _, future in ordered_prefetch(range(5), work, workers=4)]

    assert out == [0, 10, 20, 30, 40]


def test_the_task_is_handed_back_alongside_its_result() -> None:
    pairs = [(task, future.result()) for task, future in ordered_prefetch("ab", str.upper, workers=2)]

    assert pairs == [("a", "A"), ("b", "B")]


def test_it_does_not_drain_the_task_iterable_up_front() -> None:
    # The point of the bound: a 200k-image corpus must not be turned into 200k
    # in-flight tasks, each holding a decoded image.
    pulled: list[int] = []

    def tasks():
        for index in range(1000):
            pulled.append(index)
            yield index

    stream = ordered_prefetch(tasks(), lambda task: task, workers=2, depth=4)
    next(stream)

    assert len(pulled) <= 5


def test_the_number_in_flight_stays_within_the_depth() -> None:
    running = 0
    peak = 0
    lock = threading.Lock()

    def work(task: int) -> int:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.01)
        with lock:
            running -= 1
        return task

    for _, future in ordered_prefetch(range(50), work, workers=4, depth=4):
        future.result()

    assert peak <= 4


def test_depth_defaults_to_at_least_the_worker_count() -> None:
    # A depth below the worker count would leave workers permanently idle.
    stream = ordered_prefetch(range(10), lambda task: task, workers=3)
    assert [future.result() for _, future in stream] == list(range(10))


def test_a_failing_task_surfaces_through_its_own_future() -> None:
    def work(task: int) -> int:
        if task == 1:
            raise ValueError("boom")
        return task

    results = list(ordered_prefetch(range(3), work, workers=2, depth=2))

    assert results[0][1].result() == 0
    with pytest.raises(ValueError, match="boom"):
        results[1][1].result()
    assert results[2][1].result() == 2


def test_one_failure_does_not_stop_the_tasks_behind_it() -> None:
    def work(task: int) -> int:
        if task == 0:
            raise ValueError("boom")
        return task

    done = [future for _, future in ordered_prefetch(range(4), work, workers=2)]

    assert [f.result() for f in done[1:]] == [1, 2, 3]


def test_an_empty_task_stream_yields_nothing() -> None:
    assert list(ordered_prefetch([], lambda task: task, workers=2)) == []


def test_a_single_worker_starts_no_threads_at_all() -> None:
    # Prefetching only pays when preparing an item costs more than handing it to
    # another thread. On small images it does not, and the whole test suite runs
    # on small images -- so one worker must mean *no pool*, not a pool of one.
    seen: list[int] = []

    def work(task: int) -> int:
        seen.append(threading.get_ident())
        return task

    out = [future.result() for _, future in ordered_prefetch(range(20), work, workers=1)]

    assert out == list(range(20))
    assert set(seen) == {threading.get_ident()}


def test_a_single_worker_still_surfaces_failures_through_the_future() -> None:
    def work(task: int) -> int:
        if task == 1:
            raise ValueError("boom")
        return task

    results = list(ordered_prefetch(range(3), work, workers=1))

    assert results[0][1].result() == 0
    with pytest.raises(ValueError, match="boom"):
        results[1][1].result()
    assert results[2][1].result() == 2


def test_a_single_worker_stays_lazy_over_its_task_stream() -> None:
    pulled: list[int] = []

    def tasks():
        for index in range(1000):
            pulled.append(index)
            yield index

    stream = ordered_prefetch(tasks(), lambda task: task, workers=1)
    next(stream)

    assert len(pulled) == 1


def test_a_worker_count_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="workers"):
        list(ordered_prefetch(range(3), lambda task: task, workers=0))


def test_a_depth_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="depth"):
        list(ordered_prefetch(range(3), lambda task: task, workers=1, depth=0))


def test_abandoning_the_stream_early_shuts_the_pool_down() -> None:
    threads_before = threading.active_count()

    stream = ordered_prefetch(range(100), lambda task: task, workers=4, depth=8)
    next(stream)
    stream.close()

    # Give the pool a moment to wind down, then confirm it did.
    for _ in range(100):
        if threading.active_count() <= threads_before:
            break
        time.sleep(0.01)
    assert threading.active_count() <= threads_before
