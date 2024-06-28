# Heart Disease Prediction Models

This project aims to develop and compare predictive models for identifying individuals at risk of heart disease using machine learning techniques. Three classification methods were implemented and evaluated: Logistic Regression, Naive Bayes, and K-Nearest Neighbors (KNN).

## Models Implemented

### Logistic Regression
- **Accuracy**: [0.97]
- **Evaluation**:
  - Classification Report:
    [precision    recall  f1-score   support

       False       0.97      0.99      0.98       268
        True       0.92      0.74      0.82        31

    accuracy                           0.97       299
   macro avg       0.95      0.87      0.90       299
weighted avg       0.97      0.97      0.96       299]
  - Confusion Matrix:
    [[266  2]
 	 [8   23]]

### Naive Bayes
- **Accuracy**: [0.97]
- **Evaluation**:
  - Classification Report:
    [      precision    recall  f1-score   support

       False       0.90      1.00      0.95       268
        True       0.00      0.00      0.00        31

    accuracy                           0.90       299
   macro avg       0.45      0.50      0.47       299
weighted avg       0.80      0.90      0.85       299]
  - Confusion Matrix:
    [[268  0]
	 [31   0]]

### K-Nearest Neighbors (KNN)
- **Accuracy**: [0.96]
- **Evaluation**:
  - Classification Report:
    [precision    recall  f1-score   support

       False       0.98      0.98      0.98       268
        True       0.81      0.84      0.83        31

    accuracy                           0.96       299
   macro avg       0.90      0.91      0.90       299
weighted avg       0.96      0.96      0.96       299]
  - Confusion Matrix:
    [[268  0]
	 [31   0]]

## Conclusion

Based on the evaluation metrics, Logistic Regression was found to be the most effective for predicting heart disease in this dataset due to [reasons such as accuracy, precision, etc.]. Further refinements and feature engineering could potentially improve the models' performance.

This project demonstrates the application of machine learning in healthcare for early detection and intervention in heart disease, contributing to public health efforts.
