import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from arabic_reshaper import reshape
from bidi.algorithm import get_display


def plot_hourly_trend_from_dfs(data, column, column_label, axis_name, max_yticks=10):
    """
    Plots a line graph for a given column from multiple DataFrames with labels.

    Parameters:
    - data: list of dicts like {'df': DataFrame, 'label': str}
    - column: column name to plot from each DataFrame
    """

    sns.set_theme(style="darkgrid")

    plt.figure(figsize=(12, 6))

    for item in data:
        df = item["df"]
        label = get_display(reshape(item["label"]))
        if column in df.columns:
            sns.lineplot(x="start hour", y=column, data=df, linewidth=2, label=label)

            peak_row = df.loc[df[column].idxmax()]
            peak_hour = peak_row['start hour']
            peak_value = peak_row[column]

            plt.scatter(peak_hour, peak_value, s=100, zorder=5, label=get_display(reshape(f"مقدار اوج: {int(peak_value)}")))
            plt.text(peak_hour + 1, peak_value, '',
                     ha='center', fontsize=12, fontweight='bold')

    plt.title(get_display(reshape(f"روند ساعتی  {column_label} در {axis_name}")))
    plt.xlabel(get_display(reshape("ساعت")))
    plt.ylabel(get_display(reshape(column_label)))
    plt.xticks(range(24))
    plt.legend()
    plt.grid(True)

    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_yticks))

    plt.tight_layout()
    plt.show()


def plot_graph_from_dfs(data, column1, column2):
    """
    Plots a line graph for 2 given columns from multiple DataFrames with labels.
    """

    sns.set_theme(style="darkgrid")

    plt.figure(figsize=(12, 6))

    for item in data:
        df = item["df"]
        label = item["label"]
        sns.lineplot(x=column1, y=column2, data=df, linewidth=2, label=label)

    plt.title(f"{column2} by {column1}")
    plt.xlabel(column2)
    plt.ylabel(column2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


class GraphVisualizer:
    def __init__(self, df):
        """
        Initialize the visualizer with a dataframe.
        """
        self.df = df.copy()

        sns.set_theme(style="darkgrid")

    def plot_hourly_trend(self, column):
        """
        Plot the trend of a numerical column over the hours of the day.
        """
        plt.figure(figsize=(10, 5))
        sns.lineplot(x="start hour", y=column, data=self.df, marker="o", linewidth=2)
        plt.xlabel("Hour of the Day")
        plt.ylabel(column)
        plt.title(f"Hourly Trend of {column}")
        plt.xticks(range(24))
        plt.grid(True)
        plt.show()

    def plot_bar_chart(self, column):
        """
        Create a bar chart for a numerical column over the hours of the day.
        """
        plt.figure(figsize=(10, 5))
        sns.barplot(x="start hour", y=column, data=self.df, hue="start hour", palette="viridis", legend=False)
        plt.xlabel("Hour of the Day")
        plt.ylabel(column)
        plt.title(f"Bar Chart of {column} by Hour")
        plt.xticks(range(24))
        plt.show()

    def plot_histogram(self, column, bins=20):
        """
        Plot a histogram to show the distribution of a numerical column.
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df[column], bins=bins, kde=True, color="blue")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {column}")
        plt.show()

    def plot_scatter(self, x_column, y_column):
        """
        Plot a scatter plot to show relationships between two numerical columns.
        """
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=self.df[x_column], y=self.df[y_column], alpha=0.6)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f"{x_column} vs {y_column}")
        plt.show()

    def plot_vehicle_comparison(self):
        """
        Bar chart comparing different vehicle classes.
        """
        vehicle_types = ["number of Class 1 vehicles", "number of Class 2 vehicles",
                         "number of Class 3 vehicles", "number of Class 4 vehicles",
                         "number of Class 5 vehicles"]

        totals = self.df[vehicle_types].sum()

        plt.figure(figsize=(10, 6))
        ax = totals.plot(kind="bar", color=["blue", "green", "red", "purple", "orange"])

        plt.xlabel("Vehicle Type")
        plt.ylabel("Total Count")
        plt.title("Comparison of Different Vehicle Classes")
        plt.xticks(rotation=45)

        for i, val in enumerate(totals):
            ax.text(i, val * 1.02, str(int(val)), ha='center', fontsize=10)

        plt.show()

    def plot_correlation_heatmap(self):
        """
        Heatmap of feature correlations.
        """
        df_copy = self.df.copy()

        if "year" in df_copy.columns:
            df_copy["year"] = df_copy["year"].astype(int)

        if "month" in df_copy.columns:
            month_order = ["farvardin", "ordibehesht", "khordad", "tir", "mordad", "shahrivar",
                           "mehr", "aban", "azar", "dey", "bahman", "esfand"]
            df_copy["month"] = df_copy["month"].apply(lambda x: month_order.index(x) + 1)

        if "season" in df_copy.columns:
            season_order = {"spring": 1, "summer": 2, "autumn": 3, "winter": 4}
            df_copy["season"] = df_copy["season"].map(season_order)

        if "date" in df_copy.columns:
            df_copy["date"] = pd.to_datetime(df_copy["date"])
            df_copy["weekday"] = df_copy["date"].dt.weekday
            df_copy["is_weekend"] = df_copy["weekday"].apply(lambda x: 1 if x == 5 else 0)

        plt.figure(figsize=(10, 6))

        corr_matrix = df_copy.corr()

        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.show()

    def compare_with(self, other_df, column, label1="Primary Data", label2="Comparison Data"):
        """
        Compare the Hourly Trend of a specified column for both the primary DataFrame and another DataFrame.
        """
        plt.figure(figsize=(12, 6))

        sns.lineplot(x="start hour", y=column, data=self.df, marker="o", linewidth=2, label=label1)
        sns.lineplot(x="start hour", y=column, data=other_df, marker="s", linewidth=2, label=label2, linestyle="dashed")

        plt.xlabel("Hour of the Day")
        plt.ylabel(column)
        plt.title(f"Hourly Trend of {column}")
        plt.xticks(range(24))
        plt.legend()
        plt.grid(True)
        plt.show()
