# LLM Perceptions of Urban Crime Risk

This project provides an end-to-end research pipeline for studying how LLMs infer urban neighborhood crime risk from controlled linguistic and demographic cues.

## Project Structure

- `src/generate_descriptions.py`: Creates a controlled synthetic dataset of neighborhood descriptions.
- `src/query_llm.py`: Queries a configurable LLM endpoint (default: Google Gemini via OpenAI-compatible API) for numeric risk and qualitative safety perceptions.
- `src/analysis.py`: Runs statistical + NLP analysis and generates plots.
- `src/fairness.py`: Computes fairness metrics and summary reports.
- `src/app.py`: Streamlit dashboard for interactive exploration.
- `src/config.py`: Environment-driven settings (API key/model/rate limit/retries).
- `data/`: Input and generated CSV files.
- `outputs/`: Analysis outputs, fairness tables, and plots.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Google Gemini key:

```bash
export GEMINI_API_KEY="your_key_here"
```

Optional settings:

```bash
export LLM_MODEL="gemini-2.0-flash"
export LLM_API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export REQUESTS_PER_MINUTE="30"
export MAX_RETRIES="5"
```

## Usage

1. Generate controlled descriptions:
```bash
python src/generate_descriptions.py
```

2. Query the LLM:
```bash
python src/query_llm.py
```

3. Run analysis:
```bash
python src/analysis.py
```

4. Compute fairness metrics:
```bash
python src/fairness.py
```

5. Launch dashboard:
```bash
streamlit run src/app.py
```

## Notes

- The generated dataset is a full-factorial design over race, income, housing language, and amenity language.
- The analysis step includes ANOVA, regression, matched-income effect sizes (Cohen's d), sentiment analysis, TF-IDF, keyword frequencies, and plot generation.
- The fairness step computes disparate impact and demographic parity gaps.
