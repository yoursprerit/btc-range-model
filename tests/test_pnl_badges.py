"""Strategy P&L badges: two-significant-digit return and sign-first deltas."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import overall_core as oc  # noqa: E402


@pytest.mark.parametrize("ret, txt", [
    (-0.00187, "-0.19%"), (0.0051, "+0.51%"), (0.005, "+0.5%"),
    (-0.0013, "-0.13%"), (0.053, "+5.3%"), (0.123, "+12%"), (-0.0999, "-10%"),
    (1.234, "+123%"), (12.5, "+1,250%"), (0.0, "0%"), (1e-9, "0%"),
])
def test_pct_sig2_keeps_two_significant_digits(ret, txt):
    assert oc.pct_sig2(ret) == txt


def test_pct_sig2_handles_nan():
    assert oc.pct_sig2(float("nan")) == "—"


def test_signed_delta_leads_with_the_sign_streamlit_reads():
    loss = oc.signed_delta("$88 on $47,119 contributed", True)
    gain = oc.signed_delta("$88 on $47,119 contributed", False)
    assert loss.startswith("-") and not gain.startswith("-")
    assert loss == "-$88 on $47,119 contributed" and gain.startswith("+$88")
