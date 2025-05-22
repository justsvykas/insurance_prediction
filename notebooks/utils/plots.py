"""This module contains functions for plotting data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_categorical_vs_target(
    df: pd.DataFrame,
    categorical_cols: list[str],
    target_col: str,
    palette: dict,
    figsize: tuple[int, int] = (15, 8),
    subplot_layout: tuple[int, int] = (2, 3),
) -> None:
    """Plot relationship between categorical features and target variable."""
    fig, axes = plt.subplots(subplot_layout[0], subplot_layout[1], figsize=figsize)
    axes = axes.flatten()

    legend_handles = []
    legend_labels = []

    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        crosstab = pd.crosstab(df[col], df[target_col])
        percentage = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

        bars = crosstab.plot(kind="bar", ax=ax, legend=False)

        for j, container in enumerate(bars.containers):
            for patch in container.patches:
                patch.set_facecolor(list(palette.values())[j])

        ax.set_title(f"{col} vs {target_col}")
        ax.tick_params(axis="x", rotation=0)
        sns.despine(ax=ax)

        for container in bars.containers:
            category = container.get_label()
            for j, patch in enumerate(container.patches):
                value = percentage.iloc[j][int(category)]
                ax.annotate(
                    f"{value:.1f}%",
                    (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        if i == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.legend(
        legend_handles,
        legend_labels,
        title=target_col,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.2),
        ncol=2,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()


def plot_linearity_check(
    x: pd.DataFrame, y_pred: pd.Series, feature_names: list[str]
) -> None:
    """Plot the relationship between each feature and the logit values."""
    fig, axes = plt.subplots(
        len(feature_names), 1, figsize=(10, 4 * len(feature_names))
    )
    axes = axes if len(feature_names) > 1 else [axes]

    logit_values = np.log(y_pred / (1 - y_pred))

    for i, feature in enumerate(feature_names):
        n_bins = 10
        bins = pd.qcut(x[feature], n_bins, duplicates="drop")
        grouped = pd.DataFrame({"feature": x[feature], "logit": logit_values}).groupby(
            bins
        )
        mean_x = grouped["feature"].mean()
        mean_logit = grouped["logit"].mean()

        axes[i].scatter(mean_x, mean_logit)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Log odds")
        axes[i].set_title(f"Linearity check for {feature}")
        sns.regplot(x=mean_x, y=mean_logit, scatter=False, ax=axes[i], color="red")
        sns.despine(ax=axes[i])

    plt.tight_layout()
    plt.show()
