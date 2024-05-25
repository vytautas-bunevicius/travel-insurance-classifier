import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from typing import List

PRIMARY_COLORS = ["#5684F7", "#3A5CED", "#7E7AE6"]


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


def plot_bar_chart(df: pd.DataFrame, feature: str, save_path: str = None) -> None:
    """
    Plots a bar chart for a categorical feature in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame to plot.
    - feature (str): Feature to plot bar chart for.
    - save_path (str): Path to save the image file (optional).
    """
    value_counts = df[feature].value_counts().reset_index()
    value_counts.columns = [feature, "count"]
    fig = px.bar(
        value_counts,
        x=feature,
        y="count",
        title=f"Distribution of {feature}",
        template="plotly_white",
        color_discrete_sequence=[PRIMARY_COLORS[1]],
    )

    fig.update_layout(xaxis_title=feature, yaxis_title="Count")

    fig.show()
    if save_path:
        fig.write_image(save_path)
