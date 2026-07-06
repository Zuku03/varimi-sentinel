"""Tests for the advisory service and USSD flow."""

import pytest

from varimi import config
from varimi.serving import advisory
from varimi.serving.ussd import UssdSession


def test_advise_contract_all_languages():
    for lang in advisory.LANGUAGES:
        a = advisory.advise("Mutare", "Maize", language=lang)
        assert a["risk_level"] in config.RISK_CLASSES
        assert a["price_direction"] in config.PRICE_DIR_CLASSES
        assert a["language"] == lang
        assert a["crop"] in a["message"]
        assert len(a["drivers"]) == 3


def test_advise_rejects_unknown_language():
    with pytest.raises(ValueError, match="language"):
        advisory.advise("Mutare", "Maize", language="xx")


def test_ussd_full_path_reaches_advisory():
    s = UssdSession()
    assert "language" in s.prompt().lower()
    s.send("1")            # English
    s.send("1")            # first province
    s.send("1")            # first district
    final = s.send("1")    # first crop -> advisory
    assert "risk" in final.lower()
    assert s.step == "done"