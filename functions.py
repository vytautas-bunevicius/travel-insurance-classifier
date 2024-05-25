import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from typing import List
import numpy as np

PRIMARY_COLORS = ["#5684F7", "#3A5CED", "#7E7AE6"]
SECONDARY_COLORS = ["#7BC0FF", "#B8CCF4", "#18407F", "#85A2FF", "#C2A9FF", "#3D3270"]
ALL_COLORS = PRIMARY_COLORS + SECONDARY_COLORS


def plot_combined_histograms(
    df: pd.DataFrame, features: List[str], nbins: int = 40, save_path: str = None
) -> None:
    """
    Plots combined histograms for specified features in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame to plot.
    - features (List[str]): List of features to plot histograms for.
    - nbins (int): Number of bins to use in histograms.
    - save_path (str): Path to save the image file (optional).
    """
    title = f"Distribution of {', '.join(features)}"
    rows = 1
    cols = len(features)

    fig = sp.make_subplots(
        rows=rows, cols=cols, subplot_titles=features, horizontal_spacing=0.1
    )

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Histogram(
                x=df[feature],
                nbinsx=nbins,
                name=feature,
                marker=dict(
                    color=PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                    line=dict(color="#000000", width=1),
                ),
            ),
            row=1,
            col=i + 1,
        )
        fig.update_xaxes(title_text=feature, row=1, col=i + 1, title_font=dict(size=14))
        fig.update_yaxes(title_text="Count", row=1, col=i + 1, title_font=dict(size=14))

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font=dict(size=20),
        showlegend=False,
        template="plotly_white",
        height=500,
        width=400 * len(features),
        margin=dict(l=50, r=50, t=80, b=50),
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
    """
    Plots combined bar charts for specified categorical features in the DataFrame, with a maximum number of features per plot.

    Parameters:
    - df (pd.DataFrame): DataFrame to plot.
    - features (List[str]): List of categorical features to plot bar charts for.
    - max_features_per_plot (int): Maximum number of features to display per plot.
    - save_path (str): Path to save the image file (optional).
    """
    # Split the features list into chunks of size max_features_per_plot
    feature_chunks = [
        features[i : i + max_features_per_plot]
        for i in range(0, len(features), max_features_per_plot)
    ]

    for chunk_index, feature_chunk in enumerate(feature_chunks):
        title = f"Distribution of {', '.join(feature_chunk)}"
        rows = 1
        cols = len(feature_chunk)

        fig = sp.make_subplots(
            rows=rows, cols=cols, subplot_titles=feature_chunk, horizontal_spacing=0.1
        )

        for i, feature in enumerate(feature_chunk):
            value_counts = df[feature].value_counts().reset_index()
            value_counts.columns = [feature, "count"]
            fig.add_trace(
                go.Bar(
                    x=value_counts[feature],
                    y=value_counts["count"],
                    name=feature,
                    marker=dict(
                        color=PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                        line=dict(color="#000000", width=1),
                    ),
                ),
                row=1,
                col=i + 1,
            )
            fig.update_xaxes(
                title_text=feature, row=1, col=i + 1, title_font=dict(size=14)
            )
            fig.update_yaxes(
                title_text="Count", row=1, col=i + 1, title_font=dict(size=14)
            )

        fig.update_layout(
            title_text=title,
            title_x=0.5,
            title_font=dict(size=20),
            showlegend=False,
            template="plotly_white",
            height=500,
            width=400 * len(feature_chunk),
            margin=dict(l=50, r=50, t=80, b=50),
        )

        fig.show()

        if save_path:
            chunk_save_path = f"{save_path.rstrip('.png')}_part{chunk_index + 1}.png"
            fig.write_image(chunk_save_path)


def plot_combined_boxplots(
    df: pd.DataFrame, features: List[str], save_path: str = None
) -> None:
    """
    Plots combined boxplots for specified numerical features in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame to plot.
    - features (List[str]): List of numerical features to plot boxplots for.
    - save_path (str): Path to save the image file (optional).
    """
    title = f"Boxplots of {', '.join(features)}"
    rows = 1
    cols = len(features)

    fig = sp.make_subplots(
        rows=rows, cols=cols, subplot_titles=[None] * cols, horizontal_spacing=0.1
    )

    for i, feature in enumerate(features):
        fig.add_trace(
            go.Box(
                y=df[feature],
                marker=dict(
                    color=PRIMARY_COLORS[i % len(PRIMARY_COLORS)],
                    line=dict(color="#000000", width=1),
                ),
                boxmean="sd",  # Show mean and standard deviation
                showlegend=False,  # Disable legend for each trace
            ),
            row=1,
            col=i + 1,
        )
        fig.update_yaxes(title_text="Value", row=1, col=i + 1, title_font=dict(size=14))
        fig.update_xaxes(
            tickvals=[0],  # Dummy tick values
            ticktext=[feature],  # Use feature names as tick text
            row=1,
            col=i + 1,
            title_font=dict(size=14),
            showticklabels=True,
        )

    fig.update_layout(
        title_text=title,
        title_x=0.5,
        title_font=dict(size=20),
        showlegend=False,
        template="plotly_white",
        height=500,
        width=400 * len(features),
        margin=dict(
            l=50, r=50, t=80, b=150
        ),  # Increase bottom margin to accommodate x-axis labels
    )

    fig.show()

    if save_path:
        fig.write_image(save_path)


def plot_correlation_matrix(
    df: pd.DataFrame, numerical_features: list, save_path: str = None
) -> None:
    """
    Plots the correlation matrix of the specified numerical features in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - numerical_features (list): List of numerical features to include in the correlation matrix.
    - save_path (str): Path to save the image file (optional).
    """
    # Select only the specified numerical features
    numerical_df = df[numerical_features]

    # Calculate the correlation matrix
    correlation_matrix = numerical_df.corr()

    # Create the heatmap
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale=ALL_COLORS,
        title="Correlation Matrix",
    )

    # Update layout for better appearance
    fig.update_layout(
        title={
            "text": "Correlation Matrix",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        title_font=dict(size=24),
        template="plotly_white",
        height=800,
        width=800,
        margin=dict(l=100, r=100, t=100, b=100),
        xaxis=dict(tickangle=-45, title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=14)),
    )

    # Show the plot
    fig.show()

    # Save the plot if a path is provided
    if save_path:
        fig.write_image(save_path)
