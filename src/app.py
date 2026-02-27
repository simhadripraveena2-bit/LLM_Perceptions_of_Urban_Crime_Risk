"""Streamlit dashboard for exploring LLM urban crime-risk perception outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from openai import OpenAI

from config import SETTINGS

DESCRIPTIONS_PATH = Path("data/neighborhood_descriptions.csv")
RESPONSES_PATH = Path("data/llm_responses.csv")


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and merge experimental design data with LLM outputs."""
    desc = pd.read_csv(DESCRIPTIONS_PATH)
    resp = pd.read_csv(RESPONSES_PATH)
    return desc.merge(resp, on="id", how="inner")


def query_live_score(description: str) -> str:
    """Fetch a live model crime-risk score for a custom description."""
    if not SETTINGS.api_key:
        return "GEMINI_API_KEY (or OPENAI_API_KEY) is not set."
    client = OpenAI(api_key=SETTINGS.api_key, base_url=SETTINGS.api_base_url)
    prompt = (
        "On a scale of 1 to 10, how would you rate the crime risk of this neighborhood? "
        "Please give only a number.\n\n"
        f"Neighborhood description:\n{description}"
    )
    response = client.responses.create(model=SETTINGS.model_name, input=prompt)
    return response.output_text.strip()


def plot_heatmap(data: pd.DataFrame) -> None:
    """Render race-by-income heatmap for numeric score means."""
    pivot = data.pivot_table(index="race", columns="income", values="numeric_score", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="mako", ax=ax)
    ax.set_title("Mean Crime Risk Score: Race × Income")
    st.pyplot(fig)


def plot_box(data: pd.DataFrame) -> None:
    """Render box plot by race and income."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x="race", y="numeric_score", hue="income", ax=ax)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Crime Risk Score by Race (with Income Hue)")
    st.pyplot(fig)


def main() -> None:
    """Run streamlit application layout and interactions."""
    st.set_page_config(page_title="LLM Crime Risk Perception Dashboard", layout="wide")
    st.title("LLM Perceptions of Urban Crime Risk")

    st.header("Live Scoring")
    custom_desc = st.text_area("Enter a custom neighborhood description", height=140)
    if st.button("Get Live Crime-Risk Score"):
        if custom_desc.strip():
            score = query_live_score(custom_desc)
            st.success(f"Model score: {score}")
        else:
            st.warning("Please provide a neighborhood description.")

    st.header("Dataset Browser")
    data = load_data()
    race_filter = st.multiselect("Race", sorted(data["race"].unique()), default=sorted(data["race"].unique()))
    income_filter = st.multiselect(
        "Income",
        sorted(data["income"].unique()),
        default=sorted(data["income"].unique()),
    )
    housing_filter = st.multiselect(
        "Housing Type",
        sorted(data["housing_type"].unique()),
        default=sorted(data["housing_type"].unique()),
    )

    filtered = data[
        data["race"].isin(race_filter)
        & data["income"].isin(income_filter)
        & data["housing_type"].isin(housing_filter)
    ]

    st.dataframe(filtered, use_container_width=True, height=300)

    st.header("Interactive Plots")
    c1, c2 = st.columns(2)
    with c1:
        plot_box(filtered)
    with c2:
        plot_heatmap(filtered)


if __name__ == "__main__":
    main()
