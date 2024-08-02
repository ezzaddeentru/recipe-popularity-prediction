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


def save_notebook_to_github(notebook_path, repo_url, commit_message, github_token, github_email, github_username):
    # Configure git
    !git config --global credential.helper cache
    !git config --global user.email "{github_email}"
    !git config --global user.name "{github_username}"

    # Set GitHub token as environment variable
    os.environ['GITHUB_TOKEN'] = github_token
    token = os.getenv('GITHUB_TOKEN')

    # Clone the repository
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    !git clone https://{github_username}:{token}@github.com/{github_username}/{repo_name}.git

    # Copy the notebook to the repository folder
    !cp "{notebook_path}" "/content/{repo_name}"

    # Change directory to the cloned repository
    %cd /content/{repo_name}

    # Add, commit, and push the changes
    !git add "{os.path.basename(notebook_path)}"
    !git commit -m "{commit_message}"
    !git push https://{github_username}:{token}@github.com/{github_username}/{repo_name}.git


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
