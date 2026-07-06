"""VaRimi Sentinel - officer demo (Streamlit).

Run: streamlit run src/varimi/serving/app.py
A lightweight decision cockpit for extension officers: overview KPIs, a risk
hotspot map, and the per (district, crop) advisory. Bound to the live model and
dataset-02 (synthetic).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from varimi import config
from varimi.serving import advisory

_RISK_COLOR = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}


@st.cache_resource
def _load():
    model, enriched = advisory._context()
    return model, enriched


def main() -> None:
    st.set_page_config(page_title="VaRimi Sentinel", page_icon="🌾", layout="wide")
    st.title("🌾 VaRimi Sentinel - crop-climate risk & market advisory")
    st.caption(
        "Synthetic dataset-02 (not official statistics). POTRAZ AI4I - Track 3 prototype."
    )
    model, enriched = _load()

    with st.sidebar:
        st.header("Filters")
        language = st.selectbox("Language", advisory.LANGUAGES, index=0)
        province = st.selectbox("Province", sorted(enriched["province"].unique()))
        districts = sorted(enriched[enriched["province"] == province]["district"].unique())
        district = st.selectbox("District", districts)
        crops = sorted(enriched[enriched["district"] == district]["crop"].unique())
        crop = st.selectbox("Crop", crops)
        dc = enriched[(enriched["district"] == district) & (enriched["crop"] == crop)]
        months = sorted(dc["month"].unique())
        month = st.selectbox("Month", months, index=len(months) - 1)

    prov_rows = enriched[enriched["province"] == province].copy()
    preds = model.predict(prov_rows)
    prov_rows["pred_risk"] = preds[config.TARGET_RISK]

    # 1. Overview KPIs
    st.subheader("1. Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts", prov_rows["district"].nunique())
    c2.metric("High-risk rows", int((prov_rows["pred_risk"] == "High").sum()))
    c3.metric("Avg yield (t/ha)", f"{prov_rows[config.TARGET_YIELD].mean():.2f}")
    c4.metric("Crops tracked", prov_rows["crop"].nunique())

    # 2. Explore - risk hotspot map
    st.subheader("2. Explore - risk hotspots")
    latest = sorted(prov_rows["month"].unique())[-1]
    map_df = prov_rows[prov_rows["month"] == latest].copy()
    map_df["color"] = map_df["pred_risk"].map(_RISK_COLOR)
    st.map(map_df, latitude="latitude", longitude="longitude", color="color", size=200)

    # 3. Key insight - the advisory
    st.subheader("3. Key insight - advisory")
    a = advisory.advise(district, crop, month=month, language=language)
    i1, i2, i3 = st.columns(3)
    i1.metric("Risk band", a["risk_level"])
    i2.metric("Yield outlook (t/ha)", a["yield_t_per_ha"])
    i3.metric("Price direction", a["price_direction"])
    st.markdown(f"**Message:** {a['message']}")
    st.markdown("**Why (drivers):**")
    st.dataframe(pd.DataFrame(a["drivers"]), hide_index=True, width="stretch")

    # 4. Recommended action
    st.subheader("4. Recommended action")
    st.success(a["recommended_action"])


if __name__ == "__main__":
    main()