"""Run work ahead on a thread pool without losing the order it was asked in.

Extraction spends most of its time decoding a JPEG, laundering it and scoring
candidate crops -- all of which release the GIL -- and comparatively little time
in the backbone. Overlapping the two keeps the accelerator fed.

Two properties make this safe to drop under an append-only cache:

*Order is preserved.* Results are yielded in the order their tasks were
submitted, whatever order the workers actually finish in, so the cache's row
order does not depend on thread scheduling.

*The run-ahead is bounded.* At most ``depth`` tasks are ever in flight, and the
task iterable is pulled lazily, so a 200k-image corpus does not become 200k
queued tasks each holding a decoded image.

Failures are not the helper's business: each task's exception stays inside its
own future, so the caller decides whether one bad image ends the run.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def ordered_prefetch(
    tasks: Iterable[T],
    work: Callable[[T], R],
    *,
    workers: int,
    depth: int | None = None,
) -> Iterator[tuple[T, "Future[R]"]]:
    """Yield ``(task, future)`` in task order, at most ``depth`` in flight.

    ``depth`` defaults to twice ``workers``: enough slack that a worker always
    has something queued behind the task it just finished, without holding many
    more decoded images than there are threads to have produced them.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    depth = workers * 2 if depth is None else depth
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")

    stream = iter(tasks)

    if workers == 1:
        # Prefetching only pays when preparing an item costs more than handing
        # it to another thread, and on small images it does not. One worker
        # therefore means no pool and no queue at all -- just the plain loop,
        # wearing the same (task, future) interface so the caller needs no
        # second code path.
        for task in stream:
            settled: Future[R] = Future()
            try:
                settled.set_result(work(task))
            except BaseException as exc:  # noqa: BLE001 -- the caller's to handle
                settled.set_exception(exc)
            yield task, settled
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque[tuple[T, Future[R]]] = deque()

        def top_up() -> None:
            while len(pending) < depth:
                try:
                    task = next(stream)
                except StopIteration:
                    return
                pending.append((task, pool.submit(work, task)))

        top_up()
        while pending:
            item = pending.popleft()
            # Refill *before* yielding: the consumer is about to go and do its
            # own slow thing, and the pool should be busy while it does.
            top_up()
            yield item
