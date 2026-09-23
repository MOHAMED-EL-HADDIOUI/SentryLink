"""Red-team suite must catch everything it probes."""

from sentrylink import redteam


def test_redteam_all_checks_pass():
    results = redteam.run_all()
    assert len(results) == len(redteam.CHECKS) >= 15
    failures = [r for r in results if not r["passed"]]
    assert failures == [], failures


def test_redteam_main_exit_code(capsys):
    assert redteam.main() == 0
    out = capsys.readouterr().out
    assert "[PASS]" in out and "[FAIL]" not in out
