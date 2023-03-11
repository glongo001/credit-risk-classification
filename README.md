# Unit 20 Homework: Credit Risk Analysis Report

## Overview of the Analysis

* I used lending data to build a machine-learning model that evaluates borrowers and identifies their creditworthiness.

* I used the `lending_data.csv`, which contains `loan_size`, `interest_rate`, `borrower_income`, `debt_to_income`, `num_of_accounts`, `derogatory_marks`, `total_debt`, and `loan_status`. The `loan_status` column contains either `0` or `1`, where `0` means that the loan is healthy, and `1` means that the loan is at a high risk of defaulting.

* To estimate creditworthiness, first, I stored the labels set from the `loan_status` column in the `y` variable. Then, I stored the features DataFrame (all the columns except `loan_status`) in the `X` variable. And I checked the balance of the labels with `value_counts`. The results showed that, in our dataset, `75036` loans were healthy and `2500` were high-risk.

* I used the `train_test_split` module from sklearn to split the data into training and testing variables, these are: `X_train`, `X_test`, `y_train`, and `y_test`. And I assigned a `random_state` of 1 to the function to ensure that the train/test split is consistent, the same data points are assigned to the training and testing sets across multiple runs of code.

* I created a Logistic Regression Model with the original data. I used `LogisticRegression()`, from sklearn, with a `random_state` of 1. I fit the model with the training data, `X_train` and `y_train`, and predicted on testing data labels with `predict()` using the testing feature data, `X_test`, and the fitted model, `lr_model`.

* I calculated the accuracy score of the model with `balanced_accuracy_score()` from sklearn, I used `y_test` a d `testing_prediction` to obtain the accuracy.

* I generated a confusion matrix for the model with `confusion_matrix()` from sklearn, based on `y_test` and `testing_prediction`.

* I obtained a classification report for the model with `classification_report()` from sklearn, and I used `y_test` and `testing_prediction`.

* I used `RandomOverSampler()` from imbalanced-learn to resample the data. I fit the model with the training data, `X_train` and `y_train`. I generated resampled data, `X_resampled` and `y_resampled`, and used `unique()` to obtain the count of distinct values in the resampled labels data.

* Then, I created a Logistic Regression Model with the resampled data, fit the data, and made predictions. Lastly, I obtained the accuracy score, confusion matrix, and classification report of the resampled model.

## Results

* Machine Learning Model 1:
  * Model 1 Accuracy: `0.9520`.
  * Model 1 Precision: for healthy loans the precision is `1.00`, for high-risk loans the precision is `0.85`.
  * Model 1 Recall: for healthy loans the recall score is `0.99`, for high-risk loans the recall score is `0.91`.



* Machine Learning Model 2:
  * Model 2 Accuracy: `0.9947`.
  * Model 2 Precision: for healthy loans the precision is `0.99`, for high-risk loans the precision is `0.99`.
  * Model 2 Recall: for healthy loans the recall score is `0.99`, for high-risk loans the recall score is `0.99`.

## Summary

* It seems that the logistic regression model with the oversampled data performed better when making predictions.
* Performance depends on the problem we are trying to solve, if we need to predict the healthy loans the logistic regression model with the original data makes better predictions, the classification report generated a precision of `1.00`, a recall of `0.99` and an f1-score of `1.00`, while with the resampled data the precision was `0.99`, the recall was `0.99`, and the f1-score was `0.99`. However, if we need to predict the high-risk loans the linear regression model with the resampled data does much better, the precision was `0.99`, the recall was `0.99`, and the f1-score was `0.99`, while with the original data, the precision was `0.85`, the recall was `0.91`, and the f1-score was `0.88`.
* It is more important to predict high-risk loans, therefore, I would recommend the second model because it does a much better job of predicting high-risk loans than the first model.