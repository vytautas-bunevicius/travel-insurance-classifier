"""Utility functions for travel insurance data analysis and modeling.

This module provides a set of functions for analyzing, visualizing, and
modeling travel insurance data. It includes tools for exploratory data
analysis, statistical testing, model evaluation, and feature importance
analysis, tailored specifically for travel insurance datasets.

Functions:
    plot_combined_histograms: Plot histograms for multiple features.
    plot_combined_bar_charts: Create bar charts for categorical features.
    plot_combined_boxplots: Generate boxplots for numerical features.
    plot_correlation_matrix: Visualize correlation between numerical features.
    detect_anomalies_iqr: Detect anomalies using the IQR method.
    chi_square_test: Perform chi-square tests for categorical features.
    confidence_interval: Calculate confidence intervals for a dataset.
    analyze_features: Analyze numerical features with confidence intervals.
    analyze_mannwhitneyu: Conduct Mann-Whitney U tests for numerical features.
    adjust_threshold_for_recall: Adjust classification threshold for target recall.
    evaluate_model: Evaluate a travel insurance prediction model.
    plot_model_performance: Visualize performance metrics for multiple models.
    plot_combined_confusion_matrices: Plot confusion matrices for multiple models.
    extract_feature_importances: Extract feature importances from a model.

Example usage:
    import travel_insurance_utils as tiu

    # Plot combined histograms for age and trip duration
    tiu.plot_combined_histograms(df, ['Age', 'TripDuration'], nbins=40)

    # Evaluate a travel insurance prediction model
    metrics = tiu.evaluate_model('RandomForest', y_true, y_pred, threshold)

Note:
    This module is specifically designed for travel insurance data analysis
    and modeling. It assumes that the input data is structured similarly to
    typical travel insurance datasets, with features such as age, trip
    duration, destination, etc.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance

PRIMARY_COLORS = ["#5684F7", "#3A5CED", "#7E7AE6"]
SECONDARY_COLORS = [
    "#7BC0FF",
    "#B8CCF4",
    "#18407F",
    "#85A2FF",
    "#C2A9FF",
    "#3D3270",
]
ALL_COLORS = PRIMARY_COLORS + SECONDARY_COLORS


def plot_combined_histograms(
    df: pd.DataFrame, features: List[str], nbins: int = 40, save_path: str = None
) -> None:
    """Plots combined histograms for specified features in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to plot.
        features (List[str]): List of features to plot histograms for.
        nbins (int): Number of bins to use in histograms.
        save_path (str): Path to save the image file (optional).
    """
    title = f"Distribution of {', '.join(features)}"
    rows = 1
    cols = len(features)

    fig = sp.make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=features,
        horizontal_spacing=0.1,
    )

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Histogram(
                x=df[feature],
                nbinsx=nbins,
                name=feature,
                marker={
                    "color": PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                    "line": {"color": "#000000", "width": 1},
                },
            ),
            row=1,
            col=i + 1,
        )
        fig.update_xaxes(
            title_text=feature,
            row=1,
            col=i + 1,
            title_font={"size": 14},
        )
        fig.update_yaxes(
            title_text="Count",
            row=1,
            col=i + 1,
            title_font={"size": 14},
        )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font={"size": 20},
        showlegend=False,
        template="plotly_white",
        height=500,
        width=400 * len(features),
        margin={"l": 50, "r": 50, "t": 80, "b": 50},
    )

    fig.show()

    if save_path:
        fig.write_image(save_path)


def plot_combined_bar_charts(
    df: pd.DataFrame,
    features: List[str],
    max_features_per_plot: int = 3,
    save_path: str = None,
) -> None:
    """Plots combined bar charts for specified categorical features in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to plot.
        features (List[str]): List of categorical features to plot bar charts for.
        max_features_per_plot (int): Maximum number of features to display per plot.
        save_path (str): Path to save the image file (optional).
    """
    feature_chunks = [
        features[i : i + max_features_per_plot]
        for i in range(0, len(features), max_features_per_plot)
    ]

    for chunk_index, feature_chunk in enumerate(feature_chunks):
        title = f"Distribution of {', '.join(feature_chunk)}"
        rows = 1
        cols = len(feature_chunk)

        fig = sp.make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[None] * cols,
            horizontal_spacing=0.1,
        )

        for i, feature in enumerate(feature_chunk):
            value_counts = df[feature].value_counts().reset_index()
            value_counts.columns = [feature, "count"]
            fig.add_trace(
                go.Bar(
                    x=value_counts[feature],
                    y=value_counts["count"],
                    name=feature,
                    marker={
                        "color": PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                        "line": {"color": "#000000", "width": 1},
                    },
                ),
                row=1,
                col=i + 1,
            )
            fig.update_xaxes(
                title_text=feature,
                row=1,
                col=i + 1,
                title_font={"size": 14},
                showticklabels=True,
            )
            fig.update_yaxes(
                title_text="Count",
                row=1,
                col=i + 1,
                title_font={"size": 14},
            )

        fig.update_layout(
            title_text=title,
            title_x=0.5,
            title_font={"size": 20},
            showlegend=False,
            template="plotly_white",
            height=500,
            width=400 * len(feature_chunk),
            margin={"l": 50, "r": 50, "t": 80, "b": 150},
        )

        fig.show()

        if save_path:
            file_path = f"{save_path}_chunk_{chunk_index + 1}.png"
            fig.write_image(file_path)


def plot_combined_boxplots(
    df: pd.DataFrame, features: List[str], save_path: str = None
) -> None:
    """Plots combined boxplots for specified numerical features in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to plot.
        features (List[str]): List of numerical features to plot boxplots for.
        save_path (str): Path to save the image file (optional).
    """
    title = f"Boxplots of {', '.join(features)}"
    rows = 1
    cols = len(features)

    fig = sp.make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[None] * cols,
        horizontal_spacing=0.1,
    )

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Box(
                y=df[feature],
                marker={
                    "color": PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                    "line": {"color": "#000000", "width": 1},
                },
                boxmean="sd",
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )
        fig.update_yaxes(
            title_text="Value",
            row=1,
            col=i + 1,
            title_font={"size": 14},
        )
        fig.update_xaxes(
            tickvals=[0],
            ticktext=[feature],
            row=1,
            col=i + 1,
            title_font={"size": 14},
            showticklabels=True,
        )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font={"size": 20},
        showlegend=False,
        template="plotly_white",
        height=500,
        width=400 * len(features),
        margin={"l": 50, "r": 50, "t": 80, "b": 150},
    )

    fig.show()

    if save_path:
        fig.write_image(save_path)


def plot_correlation_matrix(
    df: pd.DataFrame, numerical_features: List[str], save_path: str = None
) -> None:
    """Plots the correlation matrix of numerical features in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        numerical_features (List[str]): Numerical features to include.
        save_path (str, optional): Path to save the image file.
    """

    numerical_df = df[numerical_features]
    correlation_matrix = numerical_df.corr()

    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale=ALL_COLORS,
        title="Correlation Matrix",
    )

    fig.update_layout(
        title={
            "text": "Correlation Matrix",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        title_font={"size": 24},
        template="plotly_white",
        height=800,
        width=800,
        margin={"l": 100, "r": 100, "t": 100, "b": 100},
        xaxis={
            "tickangle": -45,
            "title_font": {"size": 18},
            "tickfont": {"size": 14},
        },
        yaxis={
            "title_font": {"size": 18},
            "tickfont": {"size": 14},
        },
    )

    fig.show()

    if save_path:
        fig.write_image(save_path)


def detect_anomalies_iqr(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Detects anomalies in multiple features using the IQR method.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        features (List[str]): List of features to detect anomalies in.

    Returns:
        pd.DataFrame: DataFrame containing the anomalies for each feature.
    """
    anomalies_list = []

    for feature in features:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame.")
            continue

        if not np.issubdtype(df[feature].dtype, np.number):
            print(f"Feature '{feature}' is not numerical and will be skipped.")
            continue

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        feature_anomalies = df[
            (df[feature] < lower_bound) | (df[feature] > upper_bound)
        ]
        if not feature_anomalies.empty:
            print(f"Anomalies detected in feature '{feature}':")
            print(feature_anomalies)
        else:
            print(f"No anomalies detected in feature '{feature}'.")

        anomalies_list.append(feature_anomalies)

    if anomalies_list:
        anomalies = pd.concat(anomalies_list).drop_duplicates().reset_index(drop=True)
        anomalies = anomalies[features]
    else:
        anomalies = pd.DataFrame(columns=features)

    return anomalies


def chi_square_test(
    df: pd.DataFrame, categorical_features: List[str], target: str
) -> None:
    """Perform Chi-Square tests for association between categorical features and a target variable.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        categorical_features (List[str]): List of categorical feature column names.
        target (str): The target variable column name.

    Raises:
        ValueError: If input arguments are not as expected.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df should be a pandas DataFrame.")
    if not all(isinstance(col, str) for col in categorical_features):
        raise ValueError("categorical_features should be a list of strings.")
    if not isinstance(target, str):
        raise ValueError("target should be a string.")

    for col in categorical_features:
        if col not in df.columns or target not in df.columns:
            print(f"Column '{col}' or target '{target}' not found in DataFrame.")
            continue

        contingency_table = pd.crosstab(df[col], df[target])
        chi2, p, _, _ = chi2_contingency(contingency_table)

        print(f"\nChi-Square test results for '{col}':")
        print(f"Chi2 statistic = {chi2}, p-value = {p}")
        if p < 0.05:
            print(f"Significant association between '{col}' and '{target}'.")
        else:
            print(f"No significant association between '{col}' and '{target}'.")


def confidence_interval(
    data: pd.Series, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate the confidence interval for a given dataset.

    Args:
        data: A pandas Series of numerical data.
        confidence: The confidence level for the interval (default is 0.95).

    Returns:
        A tuple containing the lower and upper bounds of the confidence interval.
    """
    mean = np.mean(data)
    sem_value = stats.sem(data)
    margin = sem_value * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean - margin, mean + margin


def analyze_features(
    travel_df: pd.DataFrame, numerical_features: List[str], target: str
) -> None:
    """Analyze numerical features of a DataFrame by calculating confidence intervals.

    Args:
        travel_df (pd.DataFrame): The DataFrame containing the data.
        numerical_features (List[str]): A list of numerical feature names to analyze.
        target (str): The target column name for the classification.

    Prints:
        The 95% confidence intervals for each feature for insured and not insured groups.
    """
    for feature in numerical_features:
        insured = travel_df[travel_df[target] == 1][feature]
        not_insured = travel_df[travel_df[target] == 0][feature]
        ci_insured = confidence_interval(insured)
        ci_not_insured = confidence_interval(not_insured)
        print(
            f"95% confidence interval for {feature} (insured): {ci_insured}"
        )
        print(
            f"95% confidence interval for {feature} (not insured): {ci_not_insured}"
        )


def analyze_mannwhitneyu(
    travel_df: pd.DataFrame, numerical_features: List[str], target: str
) -> None:
    """Analyze numerical features using the Mann-Whitney U test.

    Args:
        travel_df (pd.DataFrame): The DataFrame containing the data.
        numerical_features (List[str]): A list of numerical feature names to analyze.
        target (str): The target column name for the classification.

    Prints:
        The U-statistic and p-value for the Mann-Whitney U test for each feature,
        and whether there is a significant difference in distributions.
    """
    for feature in numerical_features:
        insured = travel_df[travel_df[target] == 1][feature]
        not_insured = travel_df[travel_df[target] == 0][feature]
        u_stat, p_val = mannwhitneyu(
            insured, not_insured, alternative="two-sided"
        )
        print(
            f"Mann-Whitney U test for {feature}: U-statistic = {u_stat}, "
            f"p-value = {p_val}"
        )
        if p_val < 0.05:
            print(f"Significant difference in distributions for {feature}.")
        else:
            print(f"No significant difference in distributions for {feature}.")


def adjust_threshold_for_recall(
    y_true: np.ndarray, y_proba: np.ndarray, target_recall: float = 1.0
) -> float:
    """Adjusts the classification threshold to achieve a target recall.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_proba (np.ndarray): Array of predicted probabilities.
        target_recall (float): The desired recall value (default: 1.0).

    Returns:
        float: The adjusted threshold value.
    """
    _, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    target_index = np.argmin(np.abs(recalls - target_recall))
    return thresholds[target_index]


def evaluate_model(
    name: str, y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Evaluates a model's performance using various metrics.

    Args:
        name (str): The name of the model.
        y_true (np.ndarray): Array of true labels.
        y_proba (np.ndarray): Array of predicted probabilities.
        threshold (float): The classification threshold.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation metrics.
    """
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_proba),
        "Threshold": threshold,
    }
    print(f"\n{name} Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    return metrics


def plot_model_performance(
    results: Dict[str, Dict[str, float]], metrics: List[str], save_path: str = None
) -> None:
    """Plots and optionally saves a bar chart of model performance metrics with legend on the right.

    Args:
        results (Dict[str, Dict[str, float]]): A dictionary with model
        names as keys and dicts of performance metrics as values.
        metrics (List[str]): List of performance metrics to plot (e.g., 'Accuracy', 'Precision').
        save_path (str): Path to save the image file (optional).
    """
    model_names = list(results.keys())

    data = {
        metric: [results[name][metric] for name in model_names] for metric in metrics
    }

    fig = go.Figure()

    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=data[metric],
                name=metric,
                marker_color=ALL_COLORS[i % len(ALL_COLORS)],
                text=[f"{value:.2f}" for value in data[metric]],
                textposition="auto",
            )
        )

    fig.update_layout(
        barmode="group",
        title={
            "text": "Comparison of Model Performance Metrics",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24},
        },
        xaxis_title="Model",
        yaxis_title="Value",
        legend_title="Metrics",
        font={"size": 14},
        height=500,
        width=1200,
        template="plotly_white",
        legend={"yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02},
    )

    fig.update_yaxes(
        range=[0, 1],
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
    )
    fig.update_xaxes(tickangle=-45)

    fig.show()

    if save_path:
        fig.write_image(save_path)


def _create_confusion_matrix_heatmap(
    y_test: np.ndarray, y_pred: np.ndarray, labels: List[str] = None
) -> go.Heatmap:
    """Creates a heatmap for a confusion matrix.

    Args:
        y_test (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        labels (List[str], optional): Class labels.

    Returns:
        go.Heatmap: The heatmap object for the confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    text = [
        [
            f"TN: {cm[0][0]}<br>({cm_percent[0][0]:.1f}%)",
            f"FP: {cm[0][1]}<br>({cm_percent[0][1]:.1f}%)",
        ],
        [
            f"FN: {cm[1][0]}<br>({cm_percent[1][0]:.1f}%)",
            f"TP: {cm[1][1]}<br>({cm_percent[1][1]:.1f}%)",
        ],
    ]

    heatmap = go.Heatmap(
        z=cm,
        x=labels if labels else ["Class 0", "Class 1"],
        y=labels if labels else ["Class 0", "Class 1"],
        hoverongaps=False,
        text=text,
        texttemplate="%{text}",
        colorscale=[
            [0, ALL_COLORS[2]],
            [0.33, ALL_COLORS[1]],
            [0.66, ALL_COLORS[1]],
            [1, ALL_COLORS[0]],
        ],
        showscale=False,
    )

    return heatmap


def plot_combined_confusion_matrices(
    results: Dict[str, Dict[str, float]],
    y_test: np.ndarray,
    y_pred_dict: Dict[str, np.ndarray],
    labels: List[str] = None,
    save_path: str = None,
) -> None:
    """Plots combined confusion matrices for multiple models.

    Args:
        results (Dict[str, Dict[str, float]]): Model results with names as keys.
        y_test (np.ndarray): True labels of the dataset.
        y_pred_dict (Dict[str, np.ndarray]): Predicted labels for each model.
        labels (List[str], optional): Class labels. Defaults to None.
        save_path (str, optional): Path to save the image. Defaults to None.

    Returns:
        None
    """

    n_models = len(results)
    if n_models > 4:
        print("Warning: Only the first 4 models will be plotted.")
        n_models = 4

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=list(results.keys())[:n_models],
    )

    for i, (name, _) in enumerate(list(results.items())[:n_models]):
        row = i // 2 + 1
        col = i % 2 + 1

        heatmap = _create_confusion_matrix_heatmap(
            y_test, y_pred_dict[name], labels
        )

        fig.add_trace(heatmap, row=row, col=col)

        fig.update_xaxes(
            title_text="Predicted",
            row=row,
            col=col,
            tickfont={"size": 10},
        )
        fig.update_yaxes(
            title_text="Actual",
            row=row,
            col=col,
            tickfont={"size": 10},
        )

    fig.update_layout(
        title_text="Confusion Matrices for All Models",
        title_x=0.5,
        height=500,
        width=1200,
        showlegend=False,
        font={"size": 12},
    )

    fig.show()

    if save_path:
        fig.write_image(save_path)


def extract_feature_importances(
    model, x: pd.DataFrame, y: pd.Series
) -> np.ndarray:
    """Extract feature importances using permutation importance
    for models that do not directly provide them.

    Args:
        model: Trained model.
        x (pd.DataFrame): Feature data.
        y (pd.Series): Target data.

    Returns:
        np.ndarray: Array of feature importances.
    """
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    perm_import = permutation_importance(
        model, x, y, n_repeats=30, random_state=42
    )
    return perm_import.importances_mean
