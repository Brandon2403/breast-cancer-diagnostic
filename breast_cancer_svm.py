import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pdpbox import pdp


def load_data(file_path):
    """
    Load data from CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df, selected_features):
    """
    Preprocess data, perform label encoding, and split into features and target variable.
    """
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
    X = df[selected_features]
    y = df['diagnosis']

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y


def train_svm_model(X_train, y_train, param_grid):
    """
    Hyperparameter tuning and model training using SVM.
    """
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model accuracy and plot confusion matrix.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def visualize_hyperparameter_tuning(grid_search):
    """
    Visualize impact of C on model performance.
    """
    results = pd.DataFrame(grid_search.cv_results_)
    sns.lineplot(x='param_C', y='mean_test_score', data=results)
    plt.xscale('log')  # Since C values are typically on a logarithmic scale
    plt.xlabel('C')
    plt.ylabel('Mean Test Score')
    plt.title('Hyperparameter Tuning: Impact of C')
    plt.show()


def visualize_feature_importance(model, feature_names):
    # Visualize feature importance for linear SVM.
    if model.kernel == 'linear':
        coef = model.coef_[0]
        plt.bar(range(len(coef)), coef)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.title('Feature Importance for Linear SVM')
        plt.show()
    else:
        print("Feature importance is not available for non-linear kernels.")


def visualize_partial_dependence(model, X_test, feature_names):
    # Visualize partial dependence plots for selected features.
    for feature in feature_names:
        feature_names = X_test.columns.tolist()
        pdp_feature = pdp.pdp_isolate(model=model, dataset=pd.DataFrame(X_test, columns=feature_names),
                                      model_features=feature_names, feature=feature)
        pdp.pdp_plot(pdp_feature, feature_name=feature, plot_lines=True, n_cluster_centers=20)
        plt.show()

def visualize_pairwise_feature_relationships(df, selected_features):
    # Visualize pairwise feature relationships.
    sns.pairplot(df[selected_features], hue='diagnosis', palette='husl')
    plt.suptitle('Pairwise Feature Relationships', y=1.02)
    plt.show()

def main():
    # Configuration
    data_file_path = 'data.csv'
    selected_features = ['concave points_mean', 'perimeter_worst', 'concavity_worst', 'compactness_mean',
                         'compactness_worst', 'radius_se', 'diagnosis']

    # Load data
    df = load_data(data_file_path)

    # Preprocess data
    X, y = preprocess_data(df, selected_features)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    grid_search = train_svm_model(X_train, y_train, param_grid)

    # Evaluate model
    best_svm_model = grid_search.best_estimator_
    evaluate_model(best_svm_model, X_test, y_test)

    # Visualizations
    visualize_hyperparameter_tuning(grid_search)
    visualize_feature_importance(best_svm_model, selected_features[:-1])
    visualize_partial_dependence(best_svm_model, X_test, selected_features[:-1])
    visualize_pairwise_feature_relationships(df, selected_features)

if __name__ == "__main__":
    main()
