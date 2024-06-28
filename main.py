import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


heart_data = pd.read_csv('logatta.csv')

print("Dataset preview:")
print(heart_data.head())

X = heart_data.drop('accepted for the interview', axis=1)
Y = heart_data['accepted for the interview']
categorical_cols = ['BusinessTravel', 'MaritalStatus', 'OverTime', 'Gender']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

models = {
	'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = {}

for model_name, model in  models.items():
    model.fit(X_train, Y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(Y_test, y_pred)
    
    report = classification_report(Y_test, y_pred)
    matrix = confusion_matrix(Y_test, y_pred)
    
    results[model_name] = {
		'accuracy': accuracy,
		'classification_report': report,
		'confusion_matrix': matrix
	}


for model_name, result in results.items():
    print(f"### {model_name} ###")
    print(f"Accuracy: {result['accuracy']:.2f}")
    print("Classification Report:")
    print(result['classification_report'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("\n")

accuracy_values = [result['accuracy'] for result in results.values()]
model_names = list(results.keys())

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=accuracy_values)
plt.title('Comparison of Model Accuracies')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.show()
