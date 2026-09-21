import re
from typing import Dict, Optional, Tuple

import requests

# Matches a Prometheus text-exposition line: metric_name{labels} value [timestamp]
# Labels are optional; value/timestamp may be int/float/exponent/inf/nan.
_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)(?:\s+\S+)?\s*$"
)

ACCEPTED_METRIC = "vllm:spec_decode_num_accepted_tokens_total"
DRAFT_METRIC = "vllm:spec_decode_num_draft_tokens_total"
PREFIX_HITS_METRIC = "vllm:prefix_cache_hits_total"
PREFIX_QUERIES_METRIC = "vllm:prefix_cache_queries_total"


def parse_prometheus_text(text: str) -> Dict[str, float]:
    """
    Parses Prometheus text-exposition format, summing values across all label
    sets for each metric name. Ignores comments, malformed lines and
    non-numeric values.
    """
    totals: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue
        name, _labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def fetch_metrics(metrics_url: str, timeout: float = 5.0) -> Dict[str, float]:
    """
    Fetches and parses a Prometheus metrics endpoint. Returns an empty dict
    (never raises) on any network/parsing failure so a missing/unreachable
    --metrics-url never fails the benchmark.
    """
    try:
        response = requests.get(metrics_url, timeout=timeout)
        response.raise_for_status()
        return parse_prometheus_text(response.text)
    except Exception:
        return {}


def compute_metric_deltas(
    before: Dict[str, float], after: Dict[str, float]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Given two metric snapshots, returns (accept_per_draft, prefix_hit_rate).
    Either value is None when the underlying counters are absent or the
    denominator delta is not positive.
    """

    def _delta(name: str) -> Optional[float]:
        if name not in before or name not in after:
            return None
        return after[name] - before[name]

    accept_per_draft = None
    accepted_delta = _delta(ACCEPTED_METRIC)
    draft_delta = _delta(DRAFT_METRIC)
    if accepted_delta is not None and draft_delta is not None and draft_delta > 0:
        accept_per_draft = accepted_delta / draft_delta

    prefix_hit_rate = None
    hits_delta = _delta(PREFIX_HITS_METRIC)
    queries_delta = _delta(PREFIX_QUERIES_METRIC)
    if hits_delta is not None and queries_delta is not None and queries_delta > 0:
        prefix_hit_rate = hits_delta / queries_delta

    return accept_per_draft, prefix_hit_rate
