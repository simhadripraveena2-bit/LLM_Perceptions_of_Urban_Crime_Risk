"""Statistical and qualitative analysis for LLM perceptions of neighborhood crime risk."""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression
from transformers import pipeline
from wordcloud import WordCloud

DESCRIPTIONS_PATH = Path("data/neighborhood_descriptions.csv")
RESPONSES_PATH = Path("data/llm_responses.csv")
OUTPUT_DIR = Path("outputs")

THREAT_KEYWORDS = [
    "danger",
    "unsafe",
    "crime",
    "risk",
    "violence",
    "gang",
    "fear",
    "police",
    "theft",
    "robbery",
    "assault",
    "drugs",
]


def load_merged_data() -> pd.DataFrame:
    """Load and merge generated descriptions with LLM responses."""
    desc = pd.read_csv(DESCRIPTIONS_PATH)
    resp = pd.read_csv(RESPONSES_PATH)
    merged = desc.merge(resp, on="id", how="inner")
    merged = merged.dropna(subset=["numeric_score", "qualitative_response"]).copy()
    return merged


def run_anova(merged: pd.DataFrame) -> pd.DataFrame:
    """Run one-way ANOVA for each categorical variable against crime risk score."""
    factors = ["race", "income", "housing_type", "amenity_type"]
    records = []
    for factor in factors:
        groups = [grp["numeric_score"].values for _, grp in merged.groupby(factor)]
        f_stat, p_val = f_oneway(*groups)
        records.append({"factor": factor, "f_stat": f_stat, "p_value": p_val})
    return pd.DataFrame(records)


def run_regression(merged: pd.DataFrame) -> pd.DataFrame:
    """Estimate linear regression coefficients controlling for all design factors."""
    model_df = merged[["numeric_score", "race", "income", "housing_type", "amenity_type"]].copy()
    X = pd.get_dummies(model_df[["race", "income", "housing_type", "amenity_type"]], drop_first=True)
    y = model_df["numeric_score"]
    reg = LinearRegression()
    reg.fit(X, y)
    coeffs = pd.DataFrame({"feature": X.columns, "coefficient": reg.coef_})
    coeffs = coeffs.sort_values("coefficient", key=np.abs, ascending=False)
    return coeffs


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size for two independent samples."""
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def effect_sizes_by_income(merged: pd.DataFrame, baseline_group: str = "white_80") -> pd.DataFrame:
    """Compute racial Cohen's d values within each income bracket using a baseline race group."""
    records: List[Dict[str, float]] = []
    for income, income_df in merged.groupby("income"):
        baseline = income_df.loc[income_df["race"] == baseline_group, "numeric_score"].values
        for race in sorted(income_df["race"].unique()):
            if race == baseline_group:
                continue
            target = income_df.loc[income_df["race"] == race, "numeric_score"].values
            records.append(
                {
                    "income": income,
                    "comparison": f"{race}_vs_{baseline_group}",
                    "cohens_d": cohens_d(target, baseline),
                }
            )
    return pd.DataFrame(records)


def run_sentiment(merged: pd.DataFrame) -> pd.DataFrame:
    """Apply transformer sentiment analysis to qualitative responses."""
    analyzer = pipeline("sentiment-analysis")
    sentiments = analyzer(merged["qualitative_response"].tolist(), truncation=True)
    merged = merged.copy()
    merged["sentiment_label"] = [s["label"] for s in sentiments]
    merged["sentiment_score"] = [s["score"] for s in sentiments]
    return merged


def threat_keyword_counts(merged: pd.DataFrame) -> pd.DataFrame:
    """Count threat-coded keywords per race group in qualitative responses."""
    vectorizer = CountVectorizer(vocabulary=THREAT_KEYWORDS, lowercase=True)
    matrix = vectorizer.fit_transform(merged["qualitative_response"]) 
    counts = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
    counts["race"] = merged["race"].values
    summary = counts.groupby("race").sum().reset_index()
    return summary.melt(id_vars="race", var_name="keyword", value_name="count")


def tfidf_terms_by_group(merged: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Extract top TF-IDF distinguishing terms for each racial group."""
    records = []
    for race, group_df in merged.groupby("race"):
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1000)
        tfidf = vectorizer.fit_transform(group_df["qualitative_response"])
        scores = tfidf.mean(axis=0).A1
        terms = np.array(vectorizer.get_feature_names_out())
        top_idx = np.argsort(scores)[-top_n:][::-1]
        for idx in top_idx:
            records.append({"race": race, "term": terms[idx], "tfidf_score": float(scores[idx])})
    return pd.DataFrame(records)


def generate_plots(merged: pd.DataFrame, keywords_long: pd.DataFrame) -> None:
    """Generate and save all required figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged, x="race", y="numeric_score", hue="income")
    plt.xticks(rotation=25)
    plt.title("Crime Risk Score by Racial Group (Colored by Income)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplot_race_income.png")
    plt.close()

    heatmap_df = merged.pivot_table(index="race", columns="income", values="numeric_score", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Mean Crime Risk Score (Race × Income)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_race_income.png")
    plt.close()

    for race, group_df in merged.groupby("race"):
        text = " ".join(group_df["qualitative_response"].astype(str).tolist())
        if not text.strip():
            continue
        wc = WordCloud(width=1000, height=500, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud: {race}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"wordcloud_{race}.png")
        plt.close()

    plt.figure(figsize=(12, 6))
    top_keywords = keywords_long.sort_values("count", ascending=False).groupby("race").head(8)
    sns.barplot(data=top_keywords, x="keyword", y="count", hue="race")
    plt.title("Top Threat-Coded Keywords by Group")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threat_keywords_by_group.png")
    plt.close()


def main() -> None:
    """Run the full analysis pipeline and save tables/figures."""
    merged = load_merged_data()
    merged = run_sentiment(merged)

    anova_df = run_anova(merged)
    reg_df = run_regression(merged)
    effect_df = effect_sizes_by_income(merged)
    keyword_df = threat_keyword_counts(merged)
    tfidf_df = tfidf_terms_by_group(merged)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_DIR / "merged_with_sentiment.csv", index=False)
    anova_df.to_csv(OUTPUT_DIR / "anova_results.csv", index=False)
    reg_df.to_csv(OUTPUT_DIR / "regression_coefficients.csv", index=False)
    effect_df.to_csv(OUTPUT_DIR / "cohens_d_by_income.csv", index=False)
    keyword_df.to_csv(OUTPUT_DIR / "threat_keyword_counts.csv", index=False)
    tfidf_df.to_csv(OUTPUT_DIR / "tfidf_terms_by_race.csv", index=False)

    generate_plots(merged, keyword_df)
    print("Analysis complete. Outputs written to outputs/")


if __name__ == "__main__":
    main()
