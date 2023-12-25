import tkinter as tk
from tkinter import font

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from breast_cancer_diagnosis_features_selection import df


def on_button_click():
    main.config(text="Unknown option")


def exitwindow():
    window.destroy()


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
        X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2,
                                                                            random_state=42)

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


# Create the main window
window = tk.Tk()
window.title("Breast Cancer Diagnostic Testing")
window.geometry('1980x1080')
window.iconbitmap("icon.ico")

# Create a custom font
custom_font = font.Font(family="Helvetica", size=21)

# Left page
main = tk.Label(window, text="Choose an analysis:", font=custom_font)
main.pack(pady=10)
main.place(x=1, y=1)

# Button for option 1
button1 = tk.Button(window, text="1: Basic Correlation Analysis", command=basic_correlation_analysis, font=custom_font)
button1.pack(pady=10)
button1.place(x=1, y=100)

# Button for option 2
button2 = tk.Button(window, text="2: Automatic Feature Selection based on Correlation", command=correlation_analysis,
                    font=custom_font)
button2.pack(pady=10)
button2.place(x=1, y=200)

# Button for option 3
button3 = tk.Button(window, text="3: Calculate Accuracy rate (unfiltered and filtered features)",
                    command=calculate_accuracy, font=custom_font)
button3.pack(pady=10)
button3.place(x=1, y=300)

# Button for option 4
button4 = tk.Button(window, text="4: Exit", command=exitwindow, font=custom_font)
button4.pack(pady=10)
button4.place(x=1, y=400)

# Start the main event loop
window.mainloop()
