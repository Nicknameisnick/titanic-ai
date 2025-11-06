# src/viz.py
import matplotlib.pyplot as plt
import seaborn as sns

palette = "Set2"

def _style():
    sns.set_style("whitegrid")
    sns.set_palette(palette)
    sns.set_context("talk")

def plot_histogram(df, col, hue=None):
    _style()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(data=df, x=col, hue=hue, kde=False, ax=ax, bins=20, multiple="stack")
    ax.set_title(f"Histogram: {col}")
    return fig

def plot_boxplot(df, x, y):
    _style()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(f"Boxplot: {y} by {x}")
    return fig

def plot_barplot(df, x, hue=None):
    _style()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x=x, hue=hue, ax=ax)
    ax.set_title(f"Barplot: {x}{' by ' + hue if hue else ''}")
    return fig

def plot_scatter(df, x, y, c=None):
    _style()
    fig, ax = plt.subplots(figsize=(6,4))
    if c is not None:
        sns.scatterplot(data=df, x=x, y=y, hue=c, ax=ax)
    else:
        sns.scatterplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(f"Scatter: {x} vs {y}")
    return fig
