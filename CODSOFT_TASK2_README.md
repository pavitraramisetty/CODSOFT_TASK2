# Credit Card Fraud Detection

**CodSoft Machine Learning Internship --- Task 2**

## Project Overview

This project trains machine-learning classifiers to distinguish
fraudulent credit-card transactions from legitimate transactions. The
dataset is highly imbalanced, so precision, recall, F1-score, ROC-AUC,
and PR-AUC are considered alongside accuracy.

## Dataset

**Kaggle:** [Credit Card Fraud
Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

-   Original records: 284,807
-   Records after duplicate removal: 283,726
-   Features: 30
-   Target: `Class` (`0` = legitimate, `1` = fraud)

## Technologies

-   Python, Pandas, NumPy
-   Matplotlib, Seaborn
-   Scikit-learn
-   Joblib
-   Google Colab

## Workflow

1.  Load and inspect the dataset.
2.  Remove duplicate records.
3.  Examine class imbalance.
4.  Split the data into training and test sets.
5.  Scale `Time` and `Amount` using a scaler fitted on the training
    data.
6.  Train Logistic Regression, Decision Tree, and Random Forest
    classifiers.
7.  Compare models using classification metrics.
8.  Examine the confusion matrix, ROC curve, and feature importance.
9.  Save the selected model bundle.

## Model Performance

  -------------------------------------------------------------------------------
  Model          Accuracy   Precision     Recall   F1-score    ROC-AUC     PR-AUC
  ------------ ---------- ----------- ---------- ---------- ---------- ----------
  Random           0.9995      0.9710     0.7053     0.7978     0.9246     0.8171
  Forest                                                               

  Decision         0.9952      0.2256     0.7789     0.3499     0.8887     0.5375
  Tree                                                                 

  Logistic         0.9752      0.0562     0.8737     0.1057     0.9658     0.6719
  Regression                                                           
  -------------------------------------------------------------------------------

Random Forest achieved the highest F1-score (0.7978) and precision
(0.9710) among the evaluated models. Logistic Regression had the highest
recall (0.8737) and ROC-AUC (0.9658). Model choice depends on the
relative cost of missed fraud and false alerts.

## Random Forest Confusion Matrix

  Actual / Predicted     Legitimate   Fraud
  -------------------- ------------ -------
  Legitimate                 56,649       2
  Fraud                          28      67

## Example Prediction

One test transaction was classified as **Legitimate Transaction**, with
a displayed fraud probability of **0.0%**. This result refers only to
that sample and does not guarantee that other transactions are safe.

## Saved Model

`credit_card_fraud_model.pkl`

The saved bundle contains the trained Random Forest model, scaler, and
feature names.

## Repository Contents

-   `Credit_Card_Fraud_Detection.ipynb` --- notebook
-   `credit_card_fraud_model.pkl` --- saved model bundle
-   `README.md` --- project documentation
-   `results/` --- optional plots and evaluation outputs

## How to Run

1.  Open the notebook in Google Colab or Jupyter.
2.  Download the dataset from Kaggle and upload it as directed in the
    notebook.
3.  Run the notebook cells in order.
4.  To reuse the saved model, load the `.pkl` bundle with Joblib and
    provide features in the same order and preprocessing format used
    during training.

## Conclusion

This project demonstrates fraud detection with supervised
machine-learning classifiers and highlights why evaluation metrics
beyond accuracy are important for imbalanced datasets.

**Internship:** CodSoft Machine Learning Internship --- Task 2.
