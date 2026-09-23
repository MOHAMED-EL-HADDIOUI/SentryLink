"""Simulate / bench / report smoke tests (structure + honesty, not timing)."""

import json

from sentrylink import bench, report
from sentrylink.simulate import run_simulation


def test_simulate_all_verticals():
    for vertical in ("retail", "manufacturing", "healthcare"):
        rep = run_simulation(vertical)
        assert rep["cohort_size"] == 4
        assert rep["mechanism"] == "laplace" and rep["sensitivity"] == 2.0
        assert rep["audit_valid"] is True and rep["audit_entries"] > 0
        assert rep["remaining_budget"] >= 0
        assert rep["accounting_regime"] in ("rdp", "basic")
        assert rep["quantization_error"] >= 0
        assert rep["aggregation_error"] < 1e-6
        assert -1.0 <= rep["released_correlation"] <= 1.0
        json.dumps(rep)  # JSON-serializable


def test_simulate_cli_json(capsys):
    from sentrylink.simulate import main

    assert main(["--vertical", "retail", "--json"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["vertical"] == "retail"


def test_bench_structure():
    results = bench.run_benchmarks()
    assert {
        "registration", "histogram", "variance", "correlation", "fl_round",
        "secagg_mask_finalize", "mpc_reconstruct", "sqlite_transaction",
        "audit_append",
    } <= set(results)
    assert all(v >= 0 for v in results.values())
    json.dumps(results)


def test_report_structure_quick():
    rep = report.run_report(include_tests=False, include_demo=False)
    assert rep["title"] == "SentryLink Release Report"
    assert rep["protocol_version"] == "1"
    assert rep["schema_version"] >= 1
    for section in ("security", "privacy", "persistence"):
        assert all(v == "PASS" or "/" in v for v in rep[section].values()), rep[section]
    assert rep["demo"] == {}
