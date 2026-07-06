"""USSD/SMS advisory flow (officer-mediated, feature-phone friendly).

A minimal menu state machine that walks a user language -> province -> district
-> crop and returns the localized advisory. This is a mock of the gateway
interaction (no third-party provider is called); it demonstrates the offline /
low-bandwidth delivery path described in the proposal.
"""

from __future__ import annotations

from varimi.serving import advisory

_LANG_MENU = {"1": "en", "2": "sn", "3": "nd"}
_LANG_TITLE = {"en": "English", "sn": "chiShona", "nd": "isiNdebele"}


def _options(values: list[str]) -> str:
    return "\n".join(f"{i + 1}. {v}" for i, v in enumerate(values))


class UssdSession:
    """Stateful USSD session. Call :meth:`prompt` then :meth:`send` repeatedly."""

    def __init__(self):
        _, self.enriched = advisory._context()
        self.step = "language"
        self.language = "en"
        self.province: str | None = None
        self.district: str | None = None

    def _provinces(self) -> list[str]:
        return sorted(self.enriched["province"].unique())

    def _districts(self) -> list[str]:
        rows = self.enriched[self.enriched["province"] == self.province]
        return sorted(rows["district"].unique())

    def _crops(self) -> list[str]:
        rows = self.enriched[self.enriched["district"] == self.district]
        return sorted(rows["crop"].unique())

    def prompt(self) -> str:
        if self.step == "language":
            return "VaRimi Sentinel\nChoose language:\n" + _options(
                [_LANG_TITLE[v] for v in ("en", "sn", "nd")]
            )
        if self.step == "province":
            return "Province:\n" + _options(self._provinces())
        if self.step == "district":
            return "District:\n" + _options(self._districts())
        if self.step == "crop":
            return "Crop:\n" + _options(self._crops())
        return "Session complete."

    def send(self, choice: str) -> str:
        choice = choice.strip()
        if self.step == "language":
            self.language = _LANG_MENU.get(choice, "en")
            self.step = "province"
            return self.prompt()
        if self.step == "province":
            self.province = self._provinces()[int(choice) - 1]
            self.step = "district"
            return self.prompt()
        if self.step == "district":
            self.district = self._districts()[int(choice) - 1]
            self.step = "crop"
            return self.prompt()
        if self.step == "crop":
            crop = self._crops()[int(choice) - 1]
            self.step = "done"
            return advisory.advise(self.district, crop, language=self.language)["message"]
        return "Session complete."