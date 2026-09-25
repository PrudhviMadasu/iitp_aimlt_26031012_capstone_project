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

For outliers, the IQR rule counts 32 `age` outliers and 114 `fare` outliers. The fare distribution is clearly right-skewed: its mean is 32.10, median is 14.45, and mode is 8.05. That ordering fits the data: a small number of expensive first-class tickets pulls the average upward.

The survival story is consistent across the requested breakouts. Women survived at 74.0%, men at 18.9%; first-class survival was 62.6%, second-class 47.3%, and third-class 24.2%. By sex and class, female survival was 96.7% in first class, 92.1% in second, and 50.0% in third; male survival was 36.9%, 15.7%, and 13.5%, respectively. The correlation matrix was calculated on the required six columns (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`), with `adult_male` and `alone` excluded, and the strongest off-diagonal relationships were `pclass` vs `fare` at `-0.55` and `sibsp` vs `parch` at `+0.42`.

The charts tell the same story from a few angles:

- **Survival by class and sex:** Women in first and second class had the highest survival rates, while third-class women had a much worse outcome. Men were less likely to survive in every class, so sex mattered most, with class adding another clear advantage.
- **Age by survival and sex:** The age distributions overlap substantially, and survivor median age is only a little higher. Age alone does not explain the outcome as well as the gap between men and women.
- **Fare versus age:** Survivors cluster more often at higher fares, while lower-fare passengers include more non-survivors. This supports the class comparison, rather than suggesting age alone drove survival.
- **Pairplot of the main variables:** Survivors are more concentrated in the higher-fare/lower-pclass region, while non-survivors are more common among cheaper tickets. Family size appears related, but is a weaker signal than sex and class.
- **Survival by family size:** Traveling alone has the lowest survival rate, small families do somewhat better, and very large families do worse again. Family size adds context, but remains secondary to sex and class.

The standardization check for both `age` and `fare` is also included, and it shows the expected z-score behavior: mean near 0 and standard deviation near 1, without changing the underlying distribution shape.

The modeling stage starts with a stratified train/test split before any preprocessing, and every transform step is fit only on the training split and then applied to the test split. The three classifiers are all trained on the same split, and the decision tree is visualized with `plot_tree` using labeled feature and class names.

The final classification comparison is:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8258 | 0.8136 | 0.7059 | 0.7559 | 0.8600 |
| Decision Tree | 0.7753 | 0.7333 | 0.6471 | 0.6875 | 0.8336 |
| Random Forest (untuned) | 0.7921 | 0.7460 | 0.6912 | 0.7176 | 0.8359 |
| Random Forest (GridSearchCV tuned) | 0.8090 | 0.7656 | 0.7206 | 0.7424 | 0.8365 |

The separate fare-regression results are:

| Regression model | MAE | RMSE | R² | Adjusted R² |
|---|---:|---:|---:|---:|
| Linear Regression | 21.1238 | 41.6843 | 0.3487 | 0.3097 |

The imbalance comparison uses the same test split for each Random Forest variant, with SMOTE restricted to the training fold:

| Variant | Precision | Recall | F1 |
|---|---:|---:|---:|
| Baseline | 0.7460 | 0.6912 | 0.7176 |
| `class_weight='balanced'` | 0.7541 | 0.6765 | 0.7132 |
| SMOTE (training fold only) | 0.7385 | 0.7059 | 0.7218 |

SMOTE gives the best recall and F1 in this comparison without leaking test data into resampling. The tuned Random Forest uses `max_depth=None`, `max_features='sqrt'`, and `n_estimators=300`; its reported OOB score is 0.8158.

The regression side task predicts `fare` and reports MAE, RMSE, R^2, and adjusted R^2 separately from the classification metrics. The residual plot shows clear heteroscedasticity: low predicted fares are fairly tight around zero, while the high-fare end spreads out much more. That makes sense for a skewed price distribution and is consistent with the earlier fare analysis.

I would deploy the Logistic Regression pipeline for this dataset. On the held-out split it reached 0.8258 accuracy, 0.8136 precision, 0.7559 F1, and 0.8600 ROC-AUC, the strongest values for those metrics among the compared classifiers. The tuned Random Forest had slightly higher recall (0.7206 versus 0.7059), but Logistic Regression had the better overall balance and is easier to explain. The saved artifact is the complete fitted pipeline (`preprocessor + estimator`) in `titanic_pipeline.joblib`, which the notebook reloads and applies to raw passenger records.
