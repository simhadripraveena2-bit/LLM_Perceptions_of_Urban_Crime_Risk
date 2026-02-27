# LLM Perceptions of Urban Crime Risk

This project provides an end-to-end research pipeline for studying how LLMs infer urban neighborhood crime risk from controlled linguistic and demographic cues.

## Project Structure

- `src/generate_descriptions.py`: Creates a controlled synthetic dataset of neighborhood descriptions.
- `src/query_llm.py`: Queries GPT-4o for numeric risk and qualitative safety perceptions.
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

Set your OpenAI key:

```bash
export OPENAI_API_KEY="your_key_here"
```

Optional settings:

```bash
export OPENAI_MODEL="gpt-4o"
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
