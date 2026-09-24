# Module 2 — Titanic Analytics

This module covers the Titanic EDA and modeling workflow for the capstone. It is self-contained inside the `analytics/` folder and uses its own `requirements.txt`, separate from the other project modules.

## What is in this module

- `01_eda.ipynb` loads the Titanic data, checks the data quality, and saves a cleaned version to `titanic.csv`.
- `02_modeling.ipynb` reads that cleaned dataset, compares a few model choices, tunes the strongest classifier, and saves the final pipeline to `titanic_pipeline.joblib`.
- `titanic.csv` is the cleaned dataset used by the modeling notebook.
- `titanic_pipeline.joblib` is the exported training pipeline.

## Setup

```bash
cd analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it end to end

```bash
cd analytics
source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb
```

Run `01_eda.ipynb` first so it creates `titanic.csv`. Then run `02_modeling.ipynb`, which depends on that cleaned output.

## Design decisions

- The workflow is split into two notebooks to keep the exploratory work and the modeling work separate and easier to rerun.
- Missing value handling is explicit: columns with too much missing data are dropped, while the remaining numeric gaps are filled with a meaningful summary value.
- The final model choice is a Logistic Regression pipeline because it gives the strongest balance of performance, interpretability, and simplicity on this dataset.
- The notebook pipeline is saved as a reusable artifact, so the trained model can be loaded later without retraining.

## Short module notes

This module follows the required workflow: the raw Titanic dataset is loaded once in `01_eda.ipynb`, the missing-value work is handled there, and the cleaned data is saved with `df.to_csv("titanic.csv", index=False)`. The modeling notebook then reads that saved file instead of going back to the raw source.

The missing-value rules are tied to the percentage thresholds we used. `deck` is missing in 77.22% of rows, so it gets dropped. `age` is missing in 19.87%, so it gets imputed with the median value within each `(pclass, sex)` group. `embarked` and `embark_town` are each missing in 0.22%, so the two affected rows are removed.

For outliers, the IQR logic gives `age` a small upper tail and `fare` a much bigger one, with 114 high-end fare outliers. The fare distribution is clearly right-skewed: the mean is higher than the median, and the median is higher than the mode. That matches the fact that a small number of expensive first-class tickets pulled the average upward.

The survival story is consistent across the requested breakouts. Women survived at 74.0%, men at 18.9%; first-class survival was 62.6%, second-class 47.3%, and third-class 24.2%. When sex and class are looked at together, first- and second-class women were above 90% survival, while men remained much lower in every class. The correlation matrix was calculated on the required six columns (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`), with `adult_male` and `alone` excluded, and the strongest off-diagonal relationships were `pclass` vs `fare` at `-0.55` and `sibsp` vs `parch` at `+0.42`.

There are multiple multivariate charts with written interpretation in the notebook, including class-by-sex survival, age by survival, fare vs age by class, a pairplot of the main variables, and a family-size view. The standardization check for both `age` and `fare` is also included, and it shows the expected z-score behavior: mean near 0 and standard deviation near 1, without changing the underlying distribution shape.

The modeling stage starts with a stratified train/test split before any preprocessing, and every transform step is fit only on the training split and then applied to the test split. The three classifiers are all trained on the same split, and the decision tree is visualized with `plot_tree` using labeled feature and class names.

The final classification comparison is:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8258 | 0.8136 | 0.7059 | 0.7559 | 0.8600 |
| Decision Tree | 0.7753 | 0.7333 | 0.6471 | 0.6875 | 0.8336 |
| Random Forest (untuned) | 0.7921 | 0.7460 | 0.6912 | 0.7176 | 0.8359 |
| Random Forest (GridSearchCV tuned) | 0.8090 | 0.7656 | 0.7206 | 0.7424 | 0.8365 |

The imbalance comparison includes the baseline model, `class_weight='balanced'`, and SMOTE, with SMOTE applied only to the training fold. The conclusion is that SMOTE gives the best recall improvement without leaking any test data into the resampling step. The Random Forest tuning step also reports the best parameters and the OOB score for `RandomForestClassifier(oob_score=True, ...)`.

The regression side task predicts `fare` and reports MAE, RMSE, R^2, and adjusted R^2 separately from the classification metrics. The residual plot shows clear heteroscedasticity: low predicted fares are fairly tight around zero, while the high-fare end spreads out much more. That makes sense for a skewed price distribution and is consistent with the earlier fare analysis.

My recommendation is to deploy the Logistic Regression pipeline as the production classifier. It has the strongest accuracy, precision, F1, and ROC-AUC on this split, and it is easier to explain than the tree-based alternatives. The saved artifact is the complete fitted pipeline (`preprocessor + estimator` together) in `titanic_pipeline.joblib`, and it reloads cleanly for end-to-end prediction on raw, unprocessed passenger records.
