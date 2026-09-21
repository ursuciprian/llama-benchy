import pytest

from llama_benchy.metrics import (
    ACCEPTED_METRIC,
    DRAFT_METRIC,
    PREFIX_HITS_METRIC,
    PREFIX_QUERIES_METRIC,
    compute_metric_deltas,
    parse_prometheus_text,
)

SAMPLE_TEXT = """
# HELP vllm:spec_decode_num_accepted_tokens_total total accepted tokens
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{model_name="qwen"} 120.0
vllm:spec_decode_num_draft_tokens_total{model_name="qwen"} 300.0
vllm:prefix_cache_hits_total{model_name="qwen"} 50
vllm:prefix_cache_queries_total{model_name="qwen"} 100
vllm:some_other_metric_total 5
"""


def test_parse_prometheus_text_sums_across_label_sets():
    text = SAMPLE_TEXT + 'vllm:spec_decode_num_accepted_tokens_total{model_name="other"} 30.0\n'

    parsed = parse_prometheus_text(text)

    assert parsed[ACCEPTED_METRIC] == 150.0
    assert parsed[DRAFT_METRIC] == 300.0
    assert parsed[PREFIX_HITS_METRIC] == 50.0
    assert parsed[PREFIX_QUERIES_METRIC] == 100.0


def test_parse_prometheus_text_ignores_comments_and_blank_lines():
    parsed = parse_prometheus_text("# just a comment\n\n   \n")
    assert parsed == {}


def test_parse_prometheus_text_ignores_malformed_lines():
    parsed = parse_prometheus_text("not_a_valid_line_at_all\nfoo_total notanumber\n")
    assert parsed == {}


def test_compute_metric_deltas_happy_path():
    before = parse_prometheus_text(SAMPLE_TEXT)
    after_text = SAMPLE_TEXT.replace("120.0", "180.0").replace("300.0", "450.0")
    after_text = after_text.replace('"qwen"} 50', '"qwen"} 80').replace('"qwen"} 100', '"qwen"} 150')
    after = parse_prometheus_text(after_text)

    accept_per_draft, prefix_hit_rate = compute_metric_deltas(before, after)

    assert accept_per_draft == pytest.approx((180 - 120) / (450 - 300))
    assert prefix_hit_rate == pytest.approx((80 - 50) / (150 - 100))


def test_compute_metric_deltas_blank_when_metrics_absent():
    accept_per_draft, prefix_hit_rate = compute_metric_deltas({}, {})
    assert accept_per_draft is None
    assert prefix_hit_rate is None


def test_compute_metric_deltas_blank_when_denominator_delta_not_positive():
    before = {ACCEPTED_METRIC: 10, DRAFT_METRIC: 20}
    after = {ACCEPTED_METRIC: 10, DRAFT_METRIC: 20}  # no change

    accept_per_draft, _ = compute_metric_deltas(before, after)
    assert accept_per_draft is None
