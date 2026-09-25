# Zepto Data & AI Platform — Capstone

A single repository containing three connected modules: a web-scraping/data-engineering pipeline, a Titanic analytics/modeling pipeline, and a Zepto policy RAG support assistant.

## Setup and Run

Python 3.10+ is recommended.

### Module 1 — Data Pipeline

```bash
cd data_pipeline
pip install requests beautifulsoup4 pandas
python data_pipeline.py
```

This runs scraping, SQLite database creation, and the required SQL/Pandas comparisons.

### Module 2 — Analytics Pipeline

```bash
cd analytics
pip install pandas seaborn matplotlib scikit-learn imbalanced-learn joblib
python 01_eda.py
python 02_modeling.py
```

`01_eda.py` loads the Titanic dataset once and saves `titanic.csv`; `02_modeling.py` reads that CSV. Charts and written results are generated under `analytics/charts/`, `EDA_INTERPRETATIONS.md`, and `MODEL_RESULTS.md`.

### Module 3 — Support Assistant

```bash
cd support_assistant
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860/` for the user-facing assistant and `http://localhost:7860/docs` for Swagger. The graded default is deterministic `MOCK_LLM=1`; no API key is required.

### Docker

```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

## Module 1 Design

The scraper uses `requests` and BeautifulSoup across three book categories with pagination. Data is cleaned into numeric prices/ratings and boolean stock flags, and GBP is converted using the fixed project rate `1 GBP = 105.50 INR`. SQLite contains normalized `categories` and `books` tables with a foreign key. Six SQL examples cover the required clauses, and the JOIN is reproduced with `pd.merge`.

## Module 2 Design

The EDA documents missingness decisions, generates the required Age/Fare histograms and box plots, IQR outlier counts, survival analyses, exact six-column correlation heatmap, four multivariate charts with written interpretations, and a z-score sanity check. Modeling uses a stratified split, leakage-safe preprocessing, Logistic Regression, Decision Tree and Random Forest, full classification metrics including ROC/AUC, imbalance comparisons, Random Forest GridSearchCV/OOB, and a fare regression side task with residual/heteroscedasticity analysis. The complete best pipeline is saved and reloaded.

## Module 3 Design

Eight local Zepto policy documents are embedded with `all-MiniLM-L6-v2` and stored in persistent ChromaDB using cosine similarity. LangGraph routes keyword-classified policy questions to top-3 retrieval and general questions to the required deterministic mock response. The response is Pydantic validated and served through FastAPI `/ask`. The structured prompt contains ROLE, CONTEXT, TASK, FORMAT, LENGTH, a negative constraint, and a few-shot example and is used by the retrieval-answer execution path.

## Git Workflow

The submission repository should show a feature branch with at least two legitimate commits and a merge commit back into `main`. Verify with:

```bash
git log --graph --all --oneline --decorate
```

## Academic Integrity

The submitter should review and understand the implementation. Standard library/framework documentation may be referenced; project-specific reasoning and implementation should be the submitter's own work.
