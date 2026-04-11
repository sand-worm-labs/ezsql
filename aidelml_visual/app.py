import streamlit as st
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_registry(path="./working/submission.csv"):
    BASE_DIR = Path(__file__).resolve().parent
    df = pd.read_csv(BASE_DIR / path)

    # 🧼 IMPORTANT FIX: remove hidden spaces / bad formatting
    for col in ["g1", "g2", "g3", "g5"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df


BASE_DIR = Path(__file__).resolve().parent
df = load_registry(BASE_DIR / "working/submission.csv")

st.title("📊 Query Registry Explorer (Taxonomy Only)")

# ─────────────────────────────────────────────
# g1
# ─────────────────────────────────────────────
g1_options = sorted(df["g1"].dropna().unique().tolist())
g1 = st.selectbox("g1 (domain)", g1_options)

g2_df = df[df["g1"] == g1]

if g2_df.empty:
    st.warning("No g2 found for selected g1")
    st.stop()

# ─────────────────────────────────────────────
# g2
# ─────────────────────────────────────────────
g2_options = sorted(g2_df["g2"].dropna().unique().tolist())
g2 = st.selectbox("g2 (module)", g2_options)

g3_df = g2_df[g2_df["g2"] == g2]

if g3_df.empty:
    st.warning("No g3 found for selected g2")
    st.stop()

# ─────────────────────────────────────────────
# g3
# ─────────────────────────────────────────────
g3_options = sorted(g3_df["g3"].dropna().unique().tolist())
g3 = st.selectbox("g3 (dataset)", g3_options)

g5_df = g3_df[g3_df["g3"] == g3]

if g5_df.empty:
    st.warning("No g5 found for selected g3")
    st.stop()

# ─────────────────────────────────────────────
# g5
# ─────────────────────────────────────────────
g5_options = sorted(g5_df["g5"].dropna().unique().tolist())
g5 = st.selectbox("g5 (type)", g5_options)

# ─────────────────────────────────────────────
# FINAL FILTER
# ─────────────────────────────────────────────
filtered = df[
    (df["g1"] == g1) &
    (df["g2"] == g2) &
    (df["g3"] == g3) &
    (df["g5"] == g5)
]

# ─────────────────────────────────────────────
# OUTPUT: QUERY IDS ONLY (YOUR REQUIREMENT)
# ─────────────────────────────────────────────
st.subheader("🧾 Query IDs")

query_ids = filtered["query_id"].tolist()

st.write(query_ids)

st.metric("Total Queries", len(query_ids))

# optional debug table
st.dataframe(filtered[["query_id", "g1", "g2", "g3", "g5"]])