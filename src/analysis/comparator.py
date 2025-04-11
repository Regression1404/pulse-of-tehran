import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


class DataFrameComparer:
    def __init__(self, df1, df2, columns=None):
        """
        Initializes the comparer with two DataFrames for comparison.
        :param df1: First DataFrame
        :param df2: Second DataFrame
        :param columns: List of columns to compare
        """
        if columns is None:
            columns = ['total number of vehicles']
        self.df1 = df1.copy()[['start hour'] + columns]
        self.df2 = df2.copy()[['start hour'] + columns]
        self.columns = columns

    def pearson_correlation(self):
        """
        Calculates the Pearson correlation between two DataFrames for each column.
        :return: Dictionary of Pearson correlation values for each column
        """
        pearson_results = {}
        for column in self.columns:
            pearson_results[column] = self.df1[column].corr(self.df2[column])
        return pearson_results

    def spearman_correlation(self):
        """
        Calculates the Spearman rank correlation between two DataFrames for each column.
        :return: Dictionary of Spearman correlation values for each column
        """
        spearman_results = {}
        for column in self.columns:
            spearman_results[column] = self.df1[column].corr(self.df2[column], method='spearman')
        return spearman_results

    def ratio_analysis(self):
        """
        Checks if one DataFrame is a scaled version of the other by calculating the ratio for each column.
        :return: Dictionary of ratios for each column
        """
        ratio_results = {}
        for column in self.columns:
            with pd.option_context('mode.chained_assignment', None):
                ratio = self.df1[column].values.flatten() / self.df2[column].replace(0, pd.NA).values.flatten()
            ratio_results[column] = ratio
        return ratio_results

    def kolmogorov_smirnov_test(self):
        ks_results = {}
        for column in self.columns:
            stat, p_value = ks_2samp(self.df1[column], self.df2[column])
            ks_results[column] = {"D-statistic": stat, "p-value": p_value}

        return ks_results

    def cosine_similarity(self):
        """
        Calculates the Cosine Similarity between two DataFrames for each column.
        :return: Dictionary of Cosine Similarity values for each column
        """
        cosine_results = {}
        for column in self.columns:
            similarity = cosine_similarity(self.df1[column].values.reshape(1, -1),
                                           self.df2[column].values.reshape(1, -1))
            cosine_results[column] = similarity[0][0]
        return cosine_results

    def evaluate_similarity(self):
        """
        Evaluates the similarity between two DataFrames using multiple methods for each column.
        :return: A dictionary with results from different methods for each column
        """
        results = {
            "Pearson Correlation": self.pearson_correlation(),
            "Spearman Correlation": self.spearman_correlation(),
            "Ratio Analysis": self.ratio_analysis(),
            "KS Results": self.kolmogorov_smirnov_test(),
            "Cosine Similarity": self.cosine_similarity()
        }
        return results
