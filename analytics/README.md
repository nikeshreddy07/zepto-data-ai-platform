# Analytics Pipeline

The module loads Titanic once with `sns.load_dataset("titanic")`, saves `titanic.csv`, and then continues from that offline copy.

## EDA

The EDA reports profiling and missingness, applies the assignment threshold strategy, calculates IQR outliers, fare mean/median/mode/skewness, survival rates by sex/class, the exact required six-column correlation matrix, and a z-score standardization sanity check. It generates Age/Fare histograms and box plots plus four multivariate story charts. Each required chart has a 2–4 sentence interpretation in `EDA_INTERPRETATIONS.md`.

## Modeling

The modeling script performs a stratified split before preprocessing, uses a leakage-safe `ColumnTransformer`/`Pipeline`, trains Logistic Regression, Decision Tree and Random Forest, creates confusion matrices and ROC curves/AUC, compares baseline/class-weight-balanced/SMOTE training, tunes Random Forest with GridSearchCV and OOB scoring, performs fare regression with MAE/RMSE/R²/Adjusted R², creates a residual plot and heteroscedasticity discussion, and saves the complete fitted Random Forest pipeline to `best_pipeline.joblib`.

## Run

```bash
python 01_eda.py
python 02_modeling.py
```

Generated evidence includes the required Age/Fare plots, correlation heatmap, Decision Tree, ROC curves, residual plot, multivariate charts, `EDA_INTERPRETATIONS.md`, `MODEL_RESULTS.md`, and `best_pipeline.joblib`.

Feature finalization: analytics requirements completed.
