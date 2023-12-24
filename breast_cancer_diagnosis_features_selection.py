import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Your Dataset
df = pd.read_csv('data.csv')

# Convert 'diagnosis' column to numeric values (0 for 'B' and 1 for 'M')
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})


def basic_correlation_analysis():
    # Compute correlation matrix for numeric columns only
    numeric_columns = df.select_dtypes(include=['float64']).columns
    correlation_matrix = df[numeric_columns].corr()

    # Print the correlation matrix
    print("Correlation Matrix:")
    print(correlation_matrix)

    # Correlation Analysis Visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()


def correlation_analysis(selected_features=None, remove_redundant=True):
    if selected_features is None:
        # Features with absolute correlation > 0.5 with 'diagnosis'
        corr_with_target = df.corr()['diagnosis'].abs().sort_values(ascending=False)
        selected_features = corr_with_target[corr_with_target > 0.5].index.tolist()

        # Remove 'diagnosis' from selected features if present
        if 'diagnosis' in selected_features:
            selected_features.remove('diagnosis')

        # Optional removal of redundant features
        if remove_redundant:
            high_corr_matrix = df[selected_features].corr().abs()
            upper_tri = high_corr_matrix.where(np.triu(np.ones(high_corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
            selected_features = [feat for feat in selected_features if feat not in to_drop]

    # Adjust the code to explore data, descriptive statistics, and correlation analysis for selected features
    selected_df = df[selected_features + ['diagnosis']]

    # Correlation Analysis Visualization for selected features
    correlation_matrix_selected = selected_df.corr()
    plt.figure(figsize=(8, 6))  # Adjusted figure size
    sns.heatmap(correlation_matrix_selected, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap (Selected Features)')
    plt.show()

    # Additional Analysis or Modeling using the selected features (selected_df)
    # Your further analysis or modeling code here...

    # Return selected features for further analysis if needed
    return selected_features


def calculate_accuracy(selected_features=None):
    # Split data into features and target variable
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    if selected_features:
        # If selected features are provided, use them
        X_selected = X[selected_features]
        X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        # Train a Logistic Regression model with selected features
        model_sel = LogisticRegression()
        model_sel.fit(X_train_sel, y_train_sel)
        predictions_sel = model_sel.predict(X_test_sel)

        # Calculate accuracy for selected features
        accuracy_sel = accuracy_score(y_test_sel, predictions_sel)
        print(f"Accuracy with selected features: {accuracy_sel * 100:.2f}%")

    # Train a Logistic Regression model with all features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_all = LogisticRegression()
    model_all.fit(X_train, y_train)
    predictions_all = model_all.predict(X_test)

    # Calculate accuracy for all features
    accuracy_all = accuracy_score(y_test, predictions_all)
    print(f"Accuracy with all features: {accuracy_all * 100:.2f}%")


def main():
    while True:
        print("Choose an analysis:")
        print("1. Basic Correlation Analysis")
        print("2. Automatic Feature Selection based on Correlation")
        print("3. Calculate Accuracy rate (unfiltered and filtered features)")
        print("4. Exit")

        choice = input("Enter your choice (1, 2, 3, or 4): ")

        if choice == '1':
            basic_correlation_analysis()  # Perform basic correlation analysis
            print()
        elif choice == '2':
            selected_features = correlation_analysis(remove_redundant=True)  # Perform automatic feature selection based on correlation, pretty neat eh
            print()
        elif choice == '3':
            print()
            if 'selected_features' in locals():
                calculate_accuracy(selected_features)
                print()
            else:
                calculate_accuracy()
                print()
        elif choice == '4':
            print("Exiting the program. Goodboi!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
