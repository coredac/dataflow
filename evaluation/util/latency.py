"""
Latency model for Neura PLDI 2026 artifact evaluation.

Computes per-segment and per-benchmark cycle counts based on compiled II,
pipeline depth (steps), and architecture execution model.
"""

from __future__ import annotations
import functools

from .config import (
    CPU_TRANSITION_CYCLES, MARIONETTE_TRANSITION_CYCLES,
    SegConfig,
)


def _marionette_cost(trips: list, steps: int, t: int) -> float:
    """
    Recursive latency for controller-based (Marionette) execution.
    CPU drives every loop; CGRA invocation costs (steps + t) each.

        cost([N, ...rest]) = N * (cost(rest) + t)
        cost([])           = steps
    """
    if not trips:
        return steps
    return trips[0] * (_marionette_cost(trips[1:], steps, t) + t)


def segment_latency(
    seg: SegConfig,
    ii: int,
    steps: int,
    cpu_transition: int = CPU_TRANSITION_CYCLES,
    include_steps: bool = False,
) -> float:
    """
    Compute cycles for one SegConfig.

    body_only=True  (Marionette):
        cycles = _marionette_cost(cpu_trips, steps, t)

    body_only=False, fast_switch=True:
        Default:       prod(outer) * [(inner-1)*II + transition]
                       (single-outer special case: prod(outer) * (inner-1)*II)
        include_steps: prod(outer) * [(inner-1)*II + steps + transition]

    body_only=False, fast_switch=False:
        cycles = prod(outer) * inner * (steps + transition)
    """
    if seg.body_only:
        t = MARIONETTE_TRANSITION_CYCLES if seg.fast_switch else cpu_transition
        return _marionette_cost(seg.cpu_trips, steps, t)

    outer = functools.reduce(lambda a, b: a * b, seg.cpu_trips, 1) if seg.cpu_trips else 1

    if seg.fast_switch:
        if include_steps:
            return outer * ((seg.cgra_trips - 1) * ii + steps + cpu_transition)
        else:
            if outer == 1:
                return outer * ((seg.cgra_trips - 1) * ii)
            else:
                return outer * ((seg.cgra_trips - 1) * ii + cpu_transition)
    else:
        return outer * seg.cgra_trips * (steps + cpu_transition)


def bench_latency(
    segs_with_results: list[tuple[SegConfig, int, int]],
    cpu_transition: int = CPU_TRANSITION_CYCLES,
    include_steps: bool = False,
) -> float:
    """
    Compute total latency for one benchmark.

    Independent segments (group=-1): summed via segment_latency().
    Grouped segments (group>=0): share outer CPU loop; per-outer-iteration
    costs are summed first, then multiplied by shared outer trip counts.
    """
    t = MARIONETTE_TRANSITION_CYCLES
    total = 0.0
    groups: dict[int, list] = {}

    for item in segs_with_results:
        seg = item[0]
        if seg.group >= 0:
            groups.setdefault(seg.group, []).append(item)
        else:
            seg2, ii2, steps2 = item
            total += segment_latency(seg2, ii2, steps2, cpu_transition, include_steps)

    for group_items in groups.values():
        outer_trips = group_items[0][0].group_outer_trips
        prod_outer = functools.reduce(lambda a, b: a * b, outer_trips, 1)
        per_outer = sum(
            _marionette_cost(seg.cpu_trips, steps, t)
            for seg, ii, steps in group_items
        )
        total += prod_outer * per_outer

    return total
