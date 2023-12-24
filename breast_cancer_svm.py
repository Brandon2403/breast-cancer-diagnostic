import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'selected_features' is a list of feature names
selected_features = ['concave points_mean', 'perimeter_worst', 'concavity_worst', 'compactness_mean', 'compactness_worst', 'radius_se', 'diagnosis']

# Load Your Dataset
df = pd.read_csv('data.csv')

# Convert 'diagnosis' column to numeric values (0 for 'B' and 1 for 'M')
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Use only the selected features and the target variable (diagnosis in this case)
X = df[selected_features]
y = df['diagnosis']

# Label Encoding for the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Hyperparameter Tuning
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameter:", best_params) # Get the best hyperparameter

# 2. Regularization
best_svm_model = SVC(**best_params)
best_svm_model.fit(X_train, y_train)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define feature_names
feature_names = selected_features[:-1]

# 4. Train the model with scaled features
best_svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = best_svm_model.predict(X_test_scaled)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Create a single figure
plt.figure(figsize=(14, 5))

# Plot Confusion Matrix
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Visualize the impact of C on model performance
plt.subplot(1, 2, 2)
results = pd.DataFrame(grid_search.cv_results_)
sns.lineplot(x='param_C', y='mean_test_score', data=results)
plt.xscale('log')  # Since C values are typically on a logarithmic scale
plt.xlabel('C')
plt.ylabel('Mean Test Score')
plt.title('Hyperparameter Tuning: Impact of C')
plt.show()

# 6. Interpretability: Analyze support vectors and decision boundary
# Check if the kernel is linear before accessing coef_
if best_params['kernel'] == 'linear':
    # Feature Importance for Linear SVM
    coef = best_svm_model.coef_[0]
    plt.bar(range(len(coef)), coef)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Importance for Linear SVM')
    plt.show()
else:
    print("Feature importance is not available for non-linear kernels.")

# Support vectors and decision boundary
# Extract indices of support vectors
support_vector_indices = best_svm_model.support_

# Plot decision boundary and support vectors
plt.figure(figsize=(10, 8))

# Plot data points
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='autumn', label='Training Data')

# Plot support vectors
plt.scatter(X_train_scaled[support_vector_indices, 0],
            X_train_scaled[support_vector_indices, 1],
            s=100, linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = best_svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Set labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary and Support Vectors')
plt.legend()
plt.show()
