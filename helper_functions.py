# Standard library imports
import joblib

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, RidgeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.dummy import DummyClassifier

# Model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import cross_val_score, learning_curve

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from IPython.display import display


def plot_learning_curves(model, X_train, y_train, train_sizes=[0.1, 0.33, 0.55, 0.78, 1.0], model_name='Model'):
    """
    Plot learning curves for a given model and training data.

    Parameters:
    - model: The machine learning model to evaluate.
    - X_train: Features of the training set.
    - y_train: Labels of the training set.
    - train_sizes: List of training sizes to evaluate.
    - model_name: Name of the model for the plot title.

    Returns:
    - None
    """
    # Compute learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        n_jobs=-1,
        train_sizes=train_sizes
    )

    # Calculate mean and standard deviation of training and test scores
    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    test_scores_mean = test_scores.mean(axis=1)
    test_scores_std = test_scores.std(axis=1)

    # Plot learning curves
    plt.figure()
    plt.title(f"Learning Curves for {model_name}")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid()

    # Plot training and validation scores
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation score")

    # Fill areas between the mean score and standard deviation
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # Annotate scores on the plot
    for i, size in enumerate(train_sizes):
        plt.text(size, train_scores_mean[i], f'{train_scores_mean[i]:.2f}', color='r', ha='center', va='bottom')
        plt.text(size, test_scores_mean[i], f'{test_scores_mean[i]:.2f}', color='g', ha='center', va='bottom')

    plt.legend(loc='lower left', bbox_to_anchor=(0, 0), shadow=True)
    plt.show()

def plot_model_metrics_with_size(data, width=800, height=600):
    """
    Plots the comparison of metrics for different models using Plotly.
    Lines representing the same metric for train and test sets have matching colors, and the text labels match the line colors.

    Parameters:
    data (pd.DataFrame): A DataFrame containing the metrics for different models.
    width (int): The width of the figure.
    height (int): The height of the figure.
    """
    # Create the figure
    fig = go.Figure()

    # Define a color map for metrics
    color_map = {
        'Accuracy': 'blue',
        'F1 Score': 'green',
        'Precision': 'orange',
        'Recall': 'red'
    }

    # List of metrics
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

    # Add lines for each metric
    for metric in metrics:
        color = color_map[metric]  # Get the color for the metric
        train_metric = f'{metric} Train Set'
        test_metric = f'{metric} Test Set'

        # Add Train Set line
        fig.add_trace(go.Scatter(
            x=data['Model'],
            y=data[train_metric],
            mode='lines+markers+text',
            line=dict(color=color),
            marker=dict(size=8),
            text=data[train_metric].apply(lambda x: f'{x:.2f}'),
            textposition='top right',  # Adjusted for better visibility
            textfont=dict(size=10, color=color),  # Match text color with the line color
            name=f'{metric} (Train Set)'
        ))

        # Add Test Set line
        fig.add_trace(go.Scatter(
            x=data['Model'],
            y=data[test_metric],
            mode='lines+markers+text',
            line=dict(color=color, dash='dash'),  # Same color, dashed line for distinction
            marker=dict(size=8),
            text=data[test_metric].apply(lambda x: f'{x:.2f}'),
            textposition='bottom right',  # Adjusted for better visibility
            textfont=dict(size=10, color=color),  # Match text color with the line color
            name=f'{metric} (Test Set)'
        ))

    # Update layout
    fig.update_layout(
        title='Comparison of Metrics Across Models',
        xaxis_title='Model',
        yaxis_title='Score',
        legend_title='Metric',
        xaxis=dict(tickangle=-45),  # Rotate x-axis labels for better readability
        width=width,
        height=height
    )

    # Show the figure
    fig.show()

def evaluate_base_model_without_cv_clf(model, X_train, y_train, X_test, y_test, model_name, cm_labels):
    """
    Evaluate a classification model by predicting, plotting confusion matrices,
    calculating classification metrics, and visualizing the metrics with Plotly.

    Parameters:
    - model: Trained classification model.
    - X_train: Features of the training set.
    - y_train: Labels of the training set.
    - X_test: Features of the test set.
    - y_test: Labels of the test set.
    - model_name: Name of the model for labeling.
    - cm_labels: Labels for confusion matrix axes.
    
    Returns:
    - metrics_df: DataFrame containing the classification metrics for the model.
    """
    # Predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Plot confusion matrices
    plot_confusion_matrix(y_train, y_train_pred, cm_labels, f'{model_name} Train Confusion Matrix')
    plot_confusion_matrix(y_test, y_test_pred, cm_labels, f'{model_name} Test Confusion Matrix')
    print('\n')

    # Calculate and display classification metrics
    metrics_df = classification_metrics_df(y_train, y_train_pred, y_test, y_test_pred, model_name)
    print(f"Classification Metrics for {model_name}:\n")
    display(metrics_df)
    print('\n')

    # Plot metrics with Plotly
    plot_metrics_with_plotly(metrics_df)

    return metrics_df

def evaluate_and_plot_base_model_clf(model, X_train, y_train, X_test, y_test, model_name, cm_labels, cv_folds=5, include_cv=False):
    """
    Evaluate a classification model by predicting, plotting confusion matrices,
    calculating classification metrics, and visualizing the metrics with Plotly.
    Optionally includes cross-validation metrics.

    Parameters:
    - model: Trained classification model.
    - X_train: Features of the training set.
    - y_train: Labels of the training set.
    - X_test: Features of the test set.
    - y_test: Labels of the test set.
    - model_name: Name of the model for labeling.
    - cm_labels: Labels for confusion matrix axes.
    - cv_folds: Number of cross-validation folds (default is 5).
    - include_cv: Boolean flag to include cross-validation metrics (default is False).
    
    Returns:
    - metrics_df: DataFrame containing the classification metrics for the model.
    - cv_metrics_df: DataFrame containing the cross-validation metrics for the model (if `include_cv` is True).
    """
    # Predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Plot confusion matrices
    plot_confusion_matrix(y_train, y_train_pred, cm_labels, f'{model_name} Train Confusion Matrix')
    plot_confusion_matrix(y_test, y_test_pred, cm_labels, f'{model_name} Test Confusion Matrix')
    print('\n')

    # Calculate and display classification metrics
    metrics_df = classification_metrics_df(y_train, y_train_pred, y_test, y_test_pred, model_name)
    print(f"Classification Metrics for {model_name}:\n")
    display(metrics_df)
    print('\n')

    # Optionally calculate and display cross-validation metrics
    if include_cv:
        cv_metrics_df = calculate_cv_metrics(model, X_train, y_train, model_name, cv=cv_folds)
        print(f"Cross-Validation Metrics for {model_name}:\n")
        display(cv_metrics_df)
        return metrics_df, cv_metrics_df

    # Plot metrics with Plotly
    plot_metrics_with_plotly(metrics_df)

    return metrics_df


def evaluate_base_model_clf(model, X_train, y_train, X_test, y_test, model_name, cm_labels, cv_folds=5):
    # Predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Plot confusion matrices
    plot_confusion_matrix(y_train, y_train_pred, cm_labels, f'{model_name} Train Confusion Matrix')
    plot_confusion_matrix(y_test, y_test_pred, cm_labels, f'{model_name} Test Confusion Matrix')
    print('\n')
    # Calculate and display classification metrics
    metrics_df = classification_metrics_df(y_train, y_train_pred, y_test, y_test_pred, model_name)
    print(f"Classification Metrics for {model_name}:\n")
    display(metrics_df)
    print('\n')

    # Calculate and display cross-validation metrics
    cv_metrics_df = calculate_cv_metrics(model, X_train, y_train, model_name, cv=cv_folds)
    print(f"Cross-Validation Metrics for {model_name}:\n")
    display(cv_metrics_df)

    # Plot metrics with Plotly
    plot_metrics_with_plotly(metrics_df)

    return metrics_df, cv_metrics_df

# plot functions
def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    """
    Plot a confusion matrix for classification model performance.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - labels: List of class labels for the confusion matrix
    - title: Title for the plot
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def plot_feature_importance(coef, feature_names, model_name):
    """
    Plots the feature importance based on the coefficients of a model.

    Parameters:
    - coef: Array-like, shape (n_features,)
      Coefficients of the features from the model.
    - feature_names: List of strings
      Names of the features.
    - model_name: String
      Name of the model.
    """
    # Create a Series with the feature names as the index
    feature_importance = pd.Series(coef, index=feature_names)

    # Sort the values in descending order
    feature_importance = feature_importance.sort_values(ascending=False)

    # Plot the feature importance using seaborn
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importance.values, y=feature_importance.index, hue=feature_importance.index, palette="Blues_r")
    plt.title(f'{model_name} Feature Importance')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.show()


def plot_metrics_with_plotly(data):
    """
    Plots the comparison of metrics for train and test sets using Plotly.

    Parameters:
    data (dict): A dictionary containing the metrics and their respective values for train and test sets.
    """
    df = pd.DataFrame(data)

    # Melt the DataFrame for easier plotting with Plotly
    df_melted = df.melt(id_vars='Metric', value_vars=['Train Set', 'Test Set'],
                        var_name='Set', value_name='Score')
    df_melted['Score'] = pd.to_numeric(df_melted['Score'], errors='coerce')

    # Create the figure
    fig = go.Figure()

    # Add points for Train Set and Test Set
    for label in ['Train Set', 'Test Set']:
        df_label = df_melted[df_melted['Set'] == label]
        fig.add_trace(go.Scatter(
            x=df_label['Metric'],
            y=df_label['Score'],
            mode='markers+text',  # Only markers
            text=df_label['Score'].apply(lambda x: f'{x:.2f}'),
            textposition='top center',
            name=label
        ))

    # Update layout
    fig.update_layout(
        title='Comparison of Metrics for Train and Test Sets',
        xaxis_title='Metrics',
        yaxis_title='Metric Value',
        legend_title='Set',
        xaxis=dict(tickangle=-45)  # Rotate x-axis labels for better readability
    )

    # Show the figure
    fig.show()

def plot_model_metrics(data):
    """
    Plots the comparison of metrics for different models using Plotly.
    Lines representing the same metric for train and test sets have matching colors, and the text labels match the line colors.

    Parameters:
    data (pd.DataFrame): A DataFrame containing the metrics for different models.
    """
    # Create the figure
    fig = go.Figure()

    # Define a color map for metrics
    color_map = {
        'Accuracy': 'blue',
        'F1 Score': 'green',
        'Precision': 'orange',
        'Recall': 'red'
    }

    # List of metrics
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

    # Add lines for each metric
    for metric in metrics:
        color = color_map[metric]  # Get the color for the metric
        train_metric = f'{metric} Train Set'
        test_metric = f'{metric} Test Set'

        # Add Train Set line
        fig.add_trace(go.Scatter(
            x=data['Model'],
            y=data[train_metric],
            mode='lines+markers+text',
            line=dict(color=color),
            marker=dict(size=8),
            text=data[train_metric].apply(lambda x: f'{x:.2f}'),
            textposition='top right',  # Adjusted for better visibility
            textfont=dict(size=10, color=color),  # Match text color with the line color
            name=f'{metric} (Train Set)'
        ))

        # Add Test Set line
        fig.add_trace(go.Scatter(
            x=data['Model'],
            y=data[test_metric],
            mode='lines+markers+text',
            line=dict(color=color, dash='dash'),  # Same color, dashed line for distinction
            marker=dict(size=8),
            text=data[test_metric].apply(lambda x: f'{x:.2f}'),
            textposition='bottom right',  # Adjusted for better visibility
            textfont=dict(size=10, color=color),  # Match text color with the line color
            name=f'{metric} (Test Set)'
        ))

    # Update layout
    fig.update_layout(
        title='Comparison of Metrics Across Models',
        xaxis_title='Model',
        yaxis_title='Score',
        legend_title='Metric',
        xaxis=dict(tickangle=-45)  # Rotate x-axis labels for better readability
    )

    # Show the figure
    fig.show()

def classification_metrics_df(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name):
    """
    Calculate classification metrics and create a DataFrame with metrics for training and test sets.

    Parameters:
    - y_train_true: True labels for the training set.
    - y_train_pred: Predicted labels for the training set.
    - y_test_true: True labels for the test set.
    - y_test_pred: Predicted labels for the test set.
    - model_name: Name of the model as a string.

    Returns:
    - DataFrame with classification metrics.
    """
    # Calculate metrics for training set
    train_accuracy = round(accuracy_score(y_train_true, y_train_pred), 2)
    train_precision = round(precision_score(y_train_true, y_train_pred), 2)
    train_recall = round(recall_score(y_train_true, y_train_pred), 2)
    train_f1 = round(f1_score(y_train_true, y_train_pred), 2)

    # Calculate metrics for test set
    test_accuracy = round(accuracy_score(y_test_true, y_test_pred), 2)
    test_precision = round(precision_score(y_test_true, y_test_pred), 2)
    test_recall = round(recall_score(y_test_true, y_test_pred), 2)
    test_f1 = round(f1_score(y_test_true, y_test_pred), 2)

    # Create the metrics DataFrame
    metrics_dict = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Train Set': [
            train_accuracy,
            train_precision,
            train_recall,
            train_f1
        ],
        'Test Set': [
            test_accuracy,
            test_precision,
            test_recall,
            test_f1
        ],
        'Model': [model_name] * 4
    }

    df = pd.DataFrame(metrics_dict)
    return df


def combine_and_format_metrics(dfs):
    """
    Combine multiple DataFrames with classification metrics and format them into a single DataFrame.

    Parameters:
    - dfs: List of DataFrames with metrics.

    Returns:
    - DataFrame with combined and formatted metrics.
    """
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # Pivot the DataFrame
    metrics_pivot = combined_df.pivot(index='Model', columns='Metric', values=['Train Set', 'Test Set'])

    # Flatten the multi-level columns
    metrics_pivot.columns = [f'{col[1]} {col[0]}' for col in metrics_pivot.columns]

    # Reset index to get 'Model' as a column
    metrics_pivot.reset_index(inplace=True)

    return metrics_pivot


def calculate_cv_metrics(model, X, y, model_name, cv=5):
    """
    Calculate cross-validation metrics for a given model.

    Parameters:
    - model: The classifier or regressor model.
    - X: Feature matrix.
    - y: Target vector.
    - cv: Number of cross-validation folds. Default is 5.
    - model_name: Name of the model.

    Returns:
    - metrics_df: DataFrame with mean and standard deviation of the cross-validation scores for multiple metrics.
    """
    # Define scoring metrics
    metrics = {
        'Accuracy': make_scorer(accuracy_score),
        'Precision': make_scorer(precision_score),
        'Recall': make_scorer(recall_score),
        'F1 Score': make_scorer(f1_score)
    }

    # Initialize results dictionary
    results = {'Metric': [], 'CV Mean': [], 'CV Std': [], 'Model': []}

    # Calculate scores for each metric
    for metric_name, scorer in metrics.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        results['Metric'].append(metric_name)
        results['CV Mean'].append(f'{mean_score:.2f}')
        results['CV Std'].append(f'{std_score:.2f}')
        results['Model'].append(model_name)

    # Create DataFrame from results
    metrics_df = pd.DataFrame(results)

    return metrics_df


def save_models(models_dict):
    """
    Saves multiple machine learning models to their respective files.

    Args:
    - models_dict (dict): A dictionary where the keys are model names and the values are tuples
                          containing the model and the corresponding filename.
                          Example: {'model1': (model_object1, 'model1.pkl'),
                                    'model2': (model_object2, 'model2.pkl')}
    """
    for model_name, (model, filename) in models_dict.items():
        joblib.dump(model, filename)
        print(f"Model '{model_name}' saved to {filename}")


###################################
def chi_square_test(df, col1, col2):
    """
    Performs a Chi-Square test for independence between two categorical columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - col1 (str): Name of the first categorical column.
    - col2 (str): Name of the second categorical column.

    Returns:
    - None: Prints the Chi-Square statistic, p-value, and test result.
    """
    # Create a contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])
    
    # Perform the Chi-Square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    print("Chi-Square Test:")
    print(f"Chi2 Statistic: {chi2}")
    print(f"P-value: {p}")
    print(f"Degrees of Freedom: {dof}")
    print(f"Expected Frequencies Table:\n{expected}")
    print("\nObserved Frequencies Table:")
    display(contingency_table)

    # Interpretation
    if p < 0.05:
        print(f"There is a significant association between '{col1}' and '{col2}'.")
    else:
        print(f"There is no significant association between '{col1}' and '{col2}'.")

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

def chi2_feature_selection(df, target_column, categorical_columns, k='all', alpha=0.05):
    """
    Perform Chi-square feature selection on multiple categorical features in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column for classification.
    categorical_columns (list of str): List of categorical feature columns to encode and select.
    k (int or 'all'): Number of top features to select. If 'all', select all features.
    alpha (float): Significance level to determine if a feature is statistically significant.

    Returns:
    pd.DataFrame: DataFrame containing Chi2 scores, p-values, and significance status for each feature.
    """
    # Encode the target variable
    df['target'] = LabelEncoder().fit_transform(df[target_column])
    
    # Initialize lists to store results
    results = []

    for col in categorical_columns:
        # Encode the categorical feature
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        
        # Prepare feature matrix X and target vector y
        X = df[[f'{col}_encoded']]
        y = df['target']
        
        # Perform Chi-square feature selection
        chi2_selector = SelectKBest(chi2, k=k)
        chi2_selector.fit(X, y)
        
        # Get Chi-square scores and p-values
        chi2_scores = chi2_selector.scores_
        p_values = chi2_selector.pvalues_
        
        # Determine significance
        is_significant = p_values[0] < alpha
        
        # Append results
        results.append({
            'Feature': col,
            'Chi2 Score': chi2_scores[0],
            'P-value': p_values[0],
            'Significant': 'Yes' if is_significant else 'No'
        })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    return results_df

def anova_feature_selection(df, target_column, numerical_columns):
    """
    Perform ANOVA for each numerical feature with respect to a categorical target variable.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the categorical target column.
    numerical_columns (list of str): List of numerical feature columns to test.

    Returns:
    pd.DataFrame: DataFrame containing ANOVA F-values and p-values for each feature.
    """
    results = []

    for col in numerical_columns:
        # Group data by the target column and apply ANOVA
        groups = [group[col].values for name, group in df.groupby(target_column)]
        f_value, p_value = stats.f_oneway(*groups)

        # Append results
        results.append({
            'Feature': col,
            'F-value': f_value,
            'P-value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Display results

    return results_df

def kruskal_wallis_feature_selection(df, target_column, numerical_columns):
    """
    Perform Kruskal-Wallis H-test for each numerical feature with respect to a categorical target variable.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the categorical target column.
    numerical_columns (list of str): List of numerical feature columns to test.

    Returns:
    pd.DataFrame: DataFrame containing Kruskal-Wallis H-values and p-values for each feature.
    """
    results = []

    for col in numerical_columns:
        # Group data by the target column and apply Kruskal-Wallis
        groups = [group[col].values for name, group in df.groupby(target_column)]
        h_value, p_value = stats.kruskal(*groups)

        # Append results
        results.append({
            'Feature': col,
            'H-value': h_value,
            'P-value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Display results

    return results_df

def anova_feature_selection_with_best_and_original(df, target_column, numerical_columns):
    """
    Perform ANOVA for each numerical feature with respect to a categorical target variable,
    including both the original and the best transformation based on skewness.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the categorical target column.
    numerical_columns (list of str): List of numerical feature columns to test.

    Returns:
    pd.DataFrame: DataFrame containing ANOVA F-values, p-values, and the best transformation for each feature.
    """
    results = []

    for col in numerical_columns:
        # Calculate skewness for original, log-transformed, and square root-transformed data
        original_skewness = skew(df[col])
        log_skewness = skew(np.log1p(df[col]))  # np.log1p(x) handles zero values
        sqrt_skewness = skew(np.sqrt(df[col]))

        # Determine the best transformation based on skewness closest to zero
        skewness_values = {
            'Original': original_skewness,
            'Log': log_skewness,
            'Square Root': sqrt_skewness
        }
        best_transformation = min(skewness_values, key=lambda k: abs(skewness_values[k]))

        # Apply the original data transformation
        transformed_original = df[col]

        # Apply the best transformation
        if best_transformation == 'Original':
            transformed_best = transformed_original
        elif best_transformation == 'Log':
            transformed_best = np.log1p(df[col])
        elif best_transformation == 'Square Root':
            transformed_best = np.sqrt(df[col])

        # Group data by the target column and apply ANOVA for original data
        groups_original = [group[col].values for name, group in df.assign(**{col: transformed_original}).groupby(target_column)]
        f_value_original, p_value_original = stats.f_oneway(*groups_original)

        # Group data by the target column and apply ANOVA for best transformation
        groups_best = [group[col].values for name, group in df.assign(**{col: transformed_best}).groupby(target_column)]
        f_value_best, p_value_best = stats.f_oneway(*groups_best)

        # Append results for original data
        results.append({
            'Feature': col,
            'Transformation': 'Original',
            'F-value': f_value_original,
            'P-value': p_value_original,
            'Significant': 'Yes' if p_value_original < 0.05 else 'No'
        })

        # Append results for best transformation
        results.append({
            'Feature': col,
            'Transformation': best_transformation,
            'F-value': f_value_best,
            'P-value': p_value_best,
            'Significant': 'Yes' if p_value_best < 0.05 else 'No'
        })

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    return results_df

def ttest_feature_selection(df, target_column, numerical_columns):
    """
    Perform t-test for each numerical feature with respect to a categorical target variable.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the categorical target column.
    numerical_columns (list of str): List of numerical feature columns to test.

    Returns:
    pd.DataFrame: DataFrame containing t-test t-values and p-values for each feature.
    """
    results = []

    # Ensure that the target column has exactly two unique values
    unique_values = df[target_column].unique()
    if len(unique_values) != 2:
        raise ValueError("Target column must have exactly two unique values for t-test.")

    # Split the DataFrame into two groups based on the target column
    group_a = df[df[target_column] == unique_values[0]]
    group_b = df[df[target_column] == unique_values[1]]

    for col in numerical_columns:
        # Perform t-test between the two groups for each numerical column
        t_stat, p_value = stats.ttest_ind(group_a[col], group_b[col])

        # Append results
        results.append({
            'Feature': col,
            't-value': t_stat,
            'P-value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    return results_df


def ttest_feature_selection_with_best_and_original(df, target_column, numerical_columns):
    """
    Perform t-tests for each numerical feature with respect to a categorical target variable,
    including both the original and the best transformation based on skewness.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the categorical target column.
    numerical_columns (list of str): List of numerical feature columns to test.

    Returns:
    pd.DataFrame: DataFrame containing t-test statistics, p-values, and the best transformation for each feature.
    """
    results = []

    for col in numerical_columns:
        # Calculate skewness for original, log-transformed, and square root-transformed data
        original_skewness = skew(df[col])
        log_skewness = skew(np.log1p(df[col]))  # np.log1p(x) handles zero values
        sqrt_skewness = skew(np.sqrt(df[col]))

        # Determine the best transformation based on skewness closest to zero
        skewness_values = {
            'Original': original_skewness,
            'Log': log_skewness,
            'Square Root': sqrt_skewness
        }
        best_transformation = min(skewness_values, key=lambda k: abs(skewness_values[k]))

        # Apply the best transformation and add it to the DataFrame
        if best_transformation == 'Original':
            transformed_best = df[col]
        elif best_transformation == 'Log':
            transformed_best = np.log1p(df[col])
        elif best_transformation == 'Square Root':
            transformed_best = np.sqrt(df[col])

        # Add the transformed column to the DataFrame
        df[f'{col}_transformed_best'] = transformed_best

        # Ensure the target column has exactly two unique values for t-test
        unique_targets = df[target_column].unique()
        if len(unique_targets) != 2:
            raise ValueError("Target column must have exactly two unique values for t-test")

        # Separate the groups for original and best transformed data
        group_1_original = df[df[target_column] == unique_targets[0]][col]
        group_2_original = df[df[target_column] == unique_targets[1]][col]

        group_1_best = df[df[target_column] == unique_targets[0]][f'{col}_transformed_best']
        group_2_best = df[df[target_column] == unique_targets[1]][f'{col}_transformed_best']

        # Apply t-test for original data
        t_stat_original, p_value_original = stats.ttest_ind(group_1_original, group_2_original)

        # Apply t-test for best transformation
        t_stat_best, p_value_best = stats.ttest_ind(group_1_best, group_2_best)

        # Append results for original data
        results.append({
            'Feature': col,
            'Transformation': 'Original',
            'T-statistic': t_stat_original,
            'P-value': p_value_original,
            'Significant': 'Yes' if p_value_original < 0.05 else 'No'
        })

        # Append results for best transformation
        results.append({
            'Feature': col,
            'Transformation': best_transformation,
            'T-statistic': t_stat_best,
            'P-value': p_value_best,
            'Significant': 'Yes' if p_value_best < 0.05 else 'No'
        })

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    return results_df


def plot_normality(df, numerical_columns):
    """
    Plots histograms and Q-Q plots to visually inspect the normality of numerical features,
    and adds skewness information to the Q-Q plots.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numerical_columns (list of str): List of numerical feature columns to inspect.
    """
    num_cols = len(numerical_columns)
    
    fig, axes = plt.subplots(num_cols, 2, figsize=(12, 5 * num_cols))
    
    for i, col in enumerate(numerical_columns):
        # Histogram
        sns.histplot(df[col], kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f'Histogram of {col}')
        
        # Q-Q Plot
        stats.probplot(df[col], dist="norm", plot=axes[i, 1])
        axes[i, 1].set_title(f'Q-Q Plot of {col}')
        
        # Calculate skewness
        skewness = stats.skew(df[col])
        
        # Annotate Q-Q plot with skewness
        axes[i, 1].text(0.5, 0.9, f'Skewness: {skewness:.2f}', transform=axes[i, 1].transAxes,
                        fontsize=12, verticalalignment='top', horizontalalignment='center')
    
    plt.tight_layout()
    plt.show()

from scipy.stats import skew

def calculate_skewness(df, columns):
    """
    Calculate skewness for original, log-transformed, and square root-transformed data for given columns.
    Determine the best transformation for each column.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    columns (list of str): List of column names to calculate skewness for.

    Returns:
    pd.DataFrame: DataFrame with skewness values for each transformation and the best transformation.
    """
    skewness_results = {}

    for col in columns:
        # Calculate skewness for original, log-transformed, and square root-transformed data
        original_skewness = skew(df[col])
        log_skewness = skew(np.log1p(df[col]))  # np.log1p(x) handles zero values
        sqrt_skewness = skew(np.sqrt(df[col]))
        
        # Store the skewness values
        skewness_values = {
            'Original': original_skewness,
            'Log Transformation': log_skewness,
            'Square Root Transformation': sqrt_skewness
        }
        
        # Determine the best transformation based on the skewness closest to zero
        best_transformation = min(skewness_values, key=lambda k: abs(skewness_values[k]))
        
        skewness_results[col] = skewness_values
        skewness_results[col]['Best Transformation'] = best_transformation

    # Convert the results dictionary to a DataFrame for better readability
    skewness_df = pd.DataFrame(skewness_results).T
    skewness_df = skewness_df[['Original', 'Log Transformation', 'Square Root Transformation', 'Best Transformation']]
    
    return skewness_df


##########################
import math

def plot_feature_distributions(df):
    """
    Plots histograms for each feature in the provided DataFrame with KDE.

    Parameters:
    df (pd.DataFrame): DataFrame containing the features to plot.
    """
    # Determine the number of rows and columns needed for the subplots
    num_cols = len(df.columns)
    num_rows = math.ceil(num_cols / 3)
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    # Create a histogram for each feature
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], bins=50, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        if df[col].dtype == 'object':  # Rotate x-axis labels if the column is categorical
            axes[i].tick_params(axis='x', rotation=45)

    # Remove any empty subplots
    for i in range(num_cols, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_numerical_features(df, plot_type='boxplot', ncols=3):
    """
    Plots either boxplots, stripplots, or violinplots for each numerical feature in the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the numerical features to plot.
    - plot_type (str): Type of plot to create ('boxplot', 'stripplot', or 'violinplot'). Default is 'boxplot'.
    - ncols (int): Number of columns for the subplot grid. Default is 3.
    """
    # Select numerical columns
    numerical_columns = df.select_dtypes(include='number').columns
    
    # Determine the number of rows needed based on the number of columns and ncols
    num_cols = len(numerical_columns)
    num_rows = (num_cols // ncols) + (num_cols % ncols > 0)
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(nrows=num_rows, ncols=ncols, figsize=(3 * ncols, 3 * num_rows))
    
    # Flatten axes array if there's only one row
    axes = axes.flatten() if num_rows > 1 or num_cols > 1 else [axes]
    
    # Define plot function based on the plot_type
    plot_funcs = {
        'boxplot': sns.boxplot,
        'stripplot': sns.stripplot,
        'violinplot': sns.violinplot
    }
    
    plot_func = plot_funcs.get(plot_type, sns.boxplot)

    # Create the selected plot type for each numerical feature
    for i, col in enumerate(numerical_columns):
        plot_func(y=df[col], ax=axes[i])
        axes[i].set_title(f'{col}')
    
    # Remove any unused subplots
    for i in range(len(numerical_columns), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_category_distribution(df, column_name, title='Distribution of Categories'):
    """
    Plots the distribution of categories in a specified column of the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column to analyze.
    - title (str): The title of the plot (default: 'Distribution of Categories').
    """
    # Calculate the counts and percentages for each category
    category_counts = df[column_name].value_counts()
    category_percentages = (category_counts / category_counts.sum()) * 100

    # Create a DataFrame for Plotly
    category_df = pd.DataFrame({
        'Category': category_counts.index,
        'Count': category_counts.values,
        'Percentage': category_percentages.values
    })

    # Generate a color sequence from seaborn's 'Blues_r' palette
    num_categories = len(category_df)
    color_palette = sns.color_palette("Blues_r", n_colors=num_categories).as_hex()

    # Create a color mapping
    color_mapping = {category: color_palette[i] for i, category in enumerate(category_df['Category'])}

    # Create a Plotly bar plot
    fig = px.bar(category_df,
                 x='Count',
                 y='Category',
                 orientation='h',
                 text='Percentage',
                 color='Category',
                 color_discrete_map=color_mapping,
                 title=title)

    # Update the trace to show percentages as text
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='inside')

    # Update layout
    fig.update_layout(xaxis_title='Frequency', yaxis_title='Category', showlegend=False)

    # Show the plot
    fig.show()


def plot_category_distribution_by_group(df, col1, col2, title='Distribution of Column 1 by Column 2'):
    """
    Plots the distribution of one column by another using Plotly.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - col1 (str): The name of the column to be plotted on the y-axis.
    - col2 (str): The name of the column to be used for grouping and color-coding.
    - title (str): The title of the plot.
    """
    # Calculate the counts for each combination of col1 and col2
    counts = df.groupby([col1, col2]).size().unstack(fill_value=0)

    # Calculate the percentages
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    # Reset index to use it with Plotly
    percentages = percentages.reset_index()

    # Melt the DataFrame for Plotly
    percentages_melted = percentages.melt(id_vars=col1, var_name=col2, value_name='percentage')

    # Create a Plotly bar plot
    fig = px.bar(percentages_melted,
                 y=col1,
                 x='percentage',
                 color=col2,
                 orientation='h',
                 text='percentage',
                 color_discrete_sequence=px.colors.sequential.Blues_r,
                 title=title)

    # Update layout to show text on bars
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')

    # Update axes labels
    fig.update_layout(xaxis_title='Percentage', yaxis_title=col1)

    # Show the plot
    fig.show()


def plot_numerical_feature_comparison(df, target_col, plot_type='violin'):
    """
    Plots a comparison of numerical features against a target column using the specified plot type.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - target_col (str): The name of the column to be used for grouping (e.g., 'high_traffic').
    - plot_type (str): The type of plot to generate ('violin', 'box', or 'strip').
    """
    # Select numerical columns
    numerical_columns = df.select_dtypes(include='number').columns
    
    # Determine the number of columns and rows for the subplot grid
    num_cols = len(numerical_columns)
    num_rows = (num_cols // 3) + (num_cols % 3 > 0)  # Number of rows, 3 plots per row

    # Set up the matplotlib figure with adaptable figsize
    fig, axes = plt.subplots(nrows=num_rows, ncols=min(num_cols, 3), figsize=(3 * min(num_cols, 3), 3 * num_rows))
    
    # Flatten axes array if there's only one row
    axes = axes.flatten() if num_rows > 1 or num_cols > 1 else [axes]
    
    for i, col in enumerate(numerical_columns):
        if plot_type == 'violin':
            sns.violinplot(x=target_col, y=col, data=df, ax=axes[i])
        elif plot_type == 'box':
            sns.boxplot(x=target_col, y=col, data=df, ax=axes[i])
        elif plot_type == 'strip':
            sns.stripplot(x=target_col, y=col, data=df, ax=axes[i], jitter=True)
        else:
            raise ValueError("Invalid plot_type. Choose from 'violin', 'box', or 'strip'.")
        
        axes[i].set_title(f'{col} vs {target_col}')
    
    # Remove any unused subplots
    for i in range(len(numerical_columns), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
