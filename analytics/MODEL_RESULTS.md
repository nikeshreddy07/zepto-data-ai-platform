# Modeling Results

## Classification comparison

              Model  Accuracy  Precision   Recall       F1      AUC
Logistic Regression  0.804469   0.793103 0.666667 0.724409 0.843742
      Decision Tree  0.765363   0.754717 0.579710 0.655738 0.797101
      Random Forest  0.815642   0.800000 0.695652 0.744186 0.830040

## Imbalance comparison

              Variant  Precision   Recall       F1
             baseline   0.793103 0.666667 0.724409
class_weight_balanced   0.729730 0.782609 0.755245
                SMOTE   0.739726 0.782609 0.760563

## Random Forest GridSearchCV

- Best parameters: `{'model__max_depth': 5, 'model__max_features': 'sqrt', 'model__n_estimators': 100}`
- Best CV F1: **0.746**
- OOB score: **0.827**

## Regression — Fare

- MAE: **20.834**
- RMSE: **30.491**
- R²: **0.399**
- Adjusted R²: **0.371**

### Residual / heteroscedasticity discussion

Residual spread differs noticeably across the two predicted-fare halves (spread ratio 3.12), suggesting possible heteroscedasticity. The residual plot is saved as `charts/residual_plot.png`.

## Final classifier recommendation

Based on the held-out test set, Random Forest has the highest F1 score (0.744) among the three classifiers, with accuracy 0.816, precision 0.800, recall 0.696, and AUC 0.830. These metrics provide the basis for selecting the final classifier, while the class-imbalance comparison shows how precision, recall, and F1 change under balanced and SMOTE training.

## Saved pipeline verification

The complete preprocessing + Random Forest pipeline was saved as `best_pipeline.joblib`, reloaded, and used to predict the first five raw test rows successfully.
