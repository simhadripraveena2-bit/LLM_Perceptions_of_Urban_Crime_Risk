# LinguisticRedline — Real Data Pipeline

This project builds an end-to-end research workflow for studying how LLMs infer neighborhood crime risk from **real census tract demographics and amenity environments**.

## Project Structure

- `src/fetch_census.py`: Fetches ACS5 (2022) tract-level data for the 10 largest US cities and computes derived demographic/housing features.
- `src/fetch_osm.py`: Fetches tract-level OpenStreetMap amenities with caching and derives amenity bucket labels.
- `src/generate_descriptions.py`: Converts tract rows into neutral natural-language neighborhood descriptions.
- `src/pipeline.py`: Master runner for the 3-step real-data pipeline.
- `src/query_llm.py`: Sends `id` and `description` to the configured LLM and writes `data/llm_responses.csv`.
- `src/analysis.py`: Statistical + NLP analysis with city/race percentage/vacancy/amenity controls.
- `src/fairness.py`: Fairness metrics with dominant-race groups, city breakdown, and vacancy-rate stratification.
- `src/app.py`: Streamlit dashboard with city filter, tract browser, and pydeck score map.
- `config.yaml`: Centralized keys/configuration.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set API keys in `config.yaml`:

```yaml
census_api_key: "YOUR_KEY_HERE"
openai_api_key: "YOUR_KEY_HERE"
```

## Run Full Pipeline

```bash
python src/pipeline.py
```

This executes:
1. Census tract fetch (`data/census_tracts.csv`)
2. OSM amenity fetch + merge (`data/tracts_with_amenities.csv`)
3. Description generation (`data/neighborhood_descriptions.csv`)

## Downstream Steps

```bash
python src/query_llm.py
python src/analysis.py
python src/fairness.py
streamlit run src/app.py
```
