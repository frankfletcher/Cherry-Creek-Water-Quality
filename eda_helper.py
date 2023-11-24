from config import Config
import matplotlib.pyplot as plt
import seaborn as sns

# import cudf


class EDA:
    def __init__(self, config=None) -> None:
        if config is None:
            config = Config()
        self.config = config

        self.rng = config.rng

    # adapted from mlflow documentation
    def plot_corr_to_target(self, df, save_path=None):
        """
        Plots the correlation of each variable in the dataframe with the 'demand' column.

        Args:
        - df (pd.DataFrame): DataFrame containing the data, including a 'demand' column.
        - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

        Returns:
        - None (Displays the plot on a Jupyter window)
        """

        df = df.select_dtypes(
            exclude=["object", "category"]
        )  # include=["number", "bool"])

        # Compute correlations between all variables and 'demand' a
        correlations = df.corr()["target"].drop("target").sort_values()

        # Generate a color palette from red to green
        colors = sns.diverging_palette(10, 130, as_cmap=True)
        color_mapped = correlations.map(colors)

        # Set Seaborn style
        sns.set_style(
            "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
        )  # Light grey background and thicker grid lines

        # Create bar plot
        fig = plt.figure(figsize=(12, 8))
        _ = plt.barh(correlations.index, correlations.values, color=color_mapped)

        # Set labels and title with increased font size
        plt.title("Correlation with Target", fontsize=18)
        plt.xlabel("Correlation Coefficient", fontsize=16)
        plt.ylabel("Variable", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis="x")

        plt.tight_layout()

        # prevent matplotlib from displaying the chart every time we call this function
        plt.close(fig)

        # Save the plot if save_path is specified
        if save_path:
            plt.savefig(save_path, format="png", dpi=600)

        return fig
