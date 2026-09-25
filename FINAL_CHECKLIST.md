# Final Submission Checklist

## Module 1 — Data Pipeline

- [x] Requests + BeautifulSoup scraping implementation
- [x] Three category targets with pagination
- [x] Cleaned price, rating, availability, category fields
- [x] Fixed conversion rate 1 GBP = 105.50 INR
- [x] Normalized SQLite categories/books schema with PK/FK
- [x] Six SQL examples covering required clauses and JOIN
- [x] `pd.read_sql` outputs
- [x] `pd.merge` reproduction of JOIN
- [x] JOIN equivalence check

## Module 2 — Analytics

- [x] Titanic loaded once in EDA with offline fallback to committed CSV
- [x] Profiling and missingness percentages
- [x] Threshold-based missing-value strategy documented
- [x] Age histogram
- [x] Age box plot
- [x] Fare histogram
- [x] Fare box plot
- [x] IQR outlier counts
- [x] Fare mean, median, mode, skewness
- [x] Survival by sex, class, and sex+class
- [x] Exact six-column correlation matrix and heatmap
- [x] Two strongest correlations identified
- [x] Four multivariate charts with written interpretations
- [x] Age/Fare standardization sanity check
- [x] Stratified split before preprocessing
- [x] Leakage-safe ColumnTransformer/Pipeline
- [x] Logistic Regression, Decision Tree, Random Forest
- [x] Decision-tree visualization
- [x] Accuracy, Precision, Recall, F1, ROC/AUC
- [x] ROC curve chart
- [x] Baseline vs class-weight-balanced vs SMOTE
- [x] Random Forest GridSearchCV
- [x] OOB score
- [x] Fare regression with MAE/RMSE/R²/Adjusted R²
- [x] Residual plot
- [x] Heteroscedasticity discussion
- [x] Final classifier recommendation based on actual metrics
- [x] Complete pipeline saved as `best_pipeline.joblib`
- [x] Reloaded pipeline prediction verified

## Module 3 — Support Assistant

- [x] Eight policy documents
- [x] `all-MiniLM-L6-v2` local embeddings
- [x] Persistent ChromaDB with cosine similarity
- [x] Structured prompt: ROLE/CONTEXT/TASK/FORMAT/LENGTH
- [x] Negative constraint
- [x] Few-shot example
- [x] LangGraph StateGraph
- [x] `classify_intent`
- [x] `retrieve_and_answer`
- [x] `direct_answer`
- [x] Conditional routing
- [x] Top-3 retrieval
- [x] Required deterministic mock general response
- [x] Pydantic response with answer/sources/confidence
- [x] FastAPI `/ask`
- [x] Swagger `/docs`
- [x] User-facing UI
- [x] Dockerfile
- [x] No API key required for graded mock mode

## Verification Notes

The analytics scripts were executed in an offline environment using the committed Titanic CSV fallback because the public Seaborn dataset host was unreachable from that environment. The generated charts, reports, and saved pipeline are included in this submission.

Module 1 and Module 3 rely on their normal runtime dependencies/network/model availability when executed on the submitter's machine. The existing generated Module 1 database/CSV and Module 3 ChromaDB artifacts are retained.

## Git

The repository should be pushed with the included Git history. Verify with:

```bash
git log --graph --all --oneline --decorate
```
