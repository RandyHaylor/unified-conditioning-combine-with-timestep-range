"""Shared helpers for the range-aware conditioning merge nodes."""

FLOAT_EPSILON_FOR_RANGE_BREAKPOINT_DEDUPE = 1e-6


def compute_sorted_unique_breakpoints(*range_endpoint_values):
    raw_breakpoints_sorted = sorted(range_endpoint_values)
    deduped_breakpoints = []
    for breakpoint_value in raw_breakpoints_sorted:
        if not deduped_breakpoints or (breakpoint_value - deduped_breakpoints[-1]) > FLOAT_EPSILON_FOR_RANGE_BREAKPOINT_DEDUPE:
            deduped_breakpoints.append(breakpoint_value)
    return deduped_breakpoints


def interval_is_inside_range(sub_interval_start, sub_interval_end, range_start, range_end):
    if range_end <= range_start:
        return False
    return (range_start - FLOAT_EPSILON_FOR_RANGE_BREAKPOINT_DEDUPE) <= sub_interval_start and sub_interval_end <= (range_end + FLOAT_EPSILON_FOR_RANGE_BREAKPOINT_DEDUPE)


def compute_effective_range_from_widget_and_upstream(widget_start, widget_end, upstream_metadata_dict):
    upstream_start_percent = float(upstream_metadata_dict.get("start_percent", 0.0))
    upstream_end_percent = float(upstream_metadata_dict.get("end_percent", 1.0))
    effective_start = max(widget_start, upstream_start_percent)
    effective_end = min(widget_end, upstream_end_percent)
    return effective_start, effective_end
