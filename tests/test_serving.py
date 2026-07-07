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

def test_local_language_messages_fully_localized():
    # Guard against half-localized output: no English action fragments may
    # appear in sn/nd advisories (leads, terms and actions are all catalogued).
    english_fragments = ("advice", "monitor", "consider", "conditions", "storage", "market")
    for lang in ("sn", "nd"):
        msg = advisory.advise("Mutare", "Maize", language=lang)["message"].lower()
        for frag in english_fragments:
            assert frag not in msg, f"{lang} message contains English fragment {frag!r}"


def test_action_catalog_covers_all_languages():
    from varimi.serving.advisory import _ACTIONS, LANGUAGES

    for key, variants in _ACTIONS.items():
        assert set(variants) == set(LANGUAGES), f"catalog entry {key} incomplete"