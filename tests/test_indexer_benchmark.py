from benchmarks.indexer_benchmark import (
    BenchmarkConfig,
    build_parser,
    resolve_output_path,
    run_benchmark,
    validate_config,
)


def test_parser_accepts_shape_and_runtime_options():
    args = build_parser().parse_args(
        [
            "--query-rows",
            "4",
            "--history",
            "32",
            "--top-k",
            "8",
            "--chunk-size",
            "7",
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--output",
            "results/test.json",
        ]
    )
    assert args.query_rows == 4
    assert args.history == 32
    assert args.top_k == 8
    assert args.chunk_size == 7
    assert args.repeat == 2


def test_relative_output_path_is_rooted_at_repository():
    result = resolve_output_path("results/test.json")
    assert result.is_absolute()
    assert result.name == "test.json"


def test_validate_config_rejects_top_k_larger_than_history():
    try:
        validate_config(BenchmarkConfig(history=8, top_k=9))
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:  # pragma: no cover - assertion gives a clearer failure message
        raise AssertionError("invalid configuration was accepted")


def test_validate_config_rejects_non_positive_threads():
    try:
        validate_config(BenchmarkConfig(threads=0))
    except ValueError as exc:
        assert "threads" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid thread count was accepted")


def test_small_cpu_run_checks_dense_and_chunked_exactness():
    result = run_benchmark(
        BenchmarkConfig(
            query_rows=2,
            history=32,
            head_dim=8,
            top_k=4,
            chunk_size=7,
            warmup=0,
            repeat=1,
        )
    )
    assert result["correctness"]["topk_positions_match"] is True
    assert result["correctness"]["max_score_abs_error"] <= 1e-5
    assert result["correctness"]["chunked"]["passed"] is True
    assert result["paths"]["dense"]["median_ms"] >= 0
    assert result["paths"]["chunked"]["median_ms"] >= 0
