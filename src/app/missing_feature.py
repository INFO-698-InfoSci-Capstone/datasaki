# missing_data.py

import pandas as pd
import numpy as np
from statsmodels.imputation import mice
from sklearn.impute import SimpleImputer, KNNImputer


class MissingDataAnalyzer:
    def __init__(self, data):
        self.data = data
        self.code_snippets = []

    def check_mcar(self):
        """Perform Little's MCAR test."""
        try:
            mcar_test = mice.MICEData(self.data)
            return mcar_test.mice
        except Exception as e:
            print(f"Error performing MCAR test: {e}")
            return None

    def check_mar(self):
        """Check for MAR by examining correlations of missingness."""
        missingness_correlations = self.data.isnull().corr(self.data.dropna())
        return missingness_correlations

    def check_mnar(self):
        """Placeholder for MNAR detection."""
        return "MNAR detection requires domain knowledge and sensitivity analysis."

    def analyze_missing_data(self):
        """Analyze missing data and determine if it's MCAR, MAR, or MNAR."""
        mcar_result = self.check_mcar()

        if mcar_result is not None:
            return "MCAR detected based on Little's MCAR test."

        mar_correlations = self.check_mar()

        if mar_correlations.isnull().sum().sum() == 0:
            return "Data is likely MAR."

        return self.check_mnar()

    def pairwise_deletion(self):
        """Perform pairwise deletion of missing values."""
        result = self.data.dropna()
        self.code_snippets.append("result = self.data.dropna()")
        return result

    def delete_missing_columns(self):
        """Delete columns based on missing data type."""
        mcar_result = self.check_mcar()
        mar_correlations = self.check_mar()

        if mcar_result is not None:
            columns_to_delete = self.data.columns[self.data.isnull().any()].tolist()
            self.data.drop(columns=columns_to_delete, inplace=True)
            self.code_snippets.append(f"self.data.drop(columns={columns_to_delete}, inplace=True)")
            return f"Columns deleted due to MCAR: {columns_to_delete}"

        if mar_correlations.isnull().sum().sum() == 0:
            columns_to_delete = self.data.columns[self.data.isnull().any()].tolist()
            self.data.drop(columns=columns_to_delete, inplace=True)
            self.code_snippets.append(f"self.data.drop(columns={columns_to_delete}, inplace=True)")
            return f"Columns deleted due to MAR: {columns_to_delete}"

        return "No columns deleted; data may be MNAR."

    def replace_with_value(self, value):
        """Replace missing values with a specified value."""
        self.data.fillna(value, inplace=True)
        self.code_snippets.append(f"self.data.fillna({value}, inplace=True)")
        return self.data

    def replace_with_mean(self):
        """Replace missing values with the mean of each column."""
        self.data.fillna(self.data.mean(), inplace=True)
        self.code_snippets.append("self.data.fillna(self.data.mean(), inplace=True)")
        return self.data

    def replace_with_median(self):
        """Replace missing values with the median of each column."""
        self.data.fillna(self.data.median(), inplace=True)
        self.code_snippets.append("self.data.fillna(self.data.median(), inplace=True)")
        return self.data

    def replace_with_mode(self):
        """Replace missing values with the mode of each column."""
        self.data.fillna(self.data.mode().iloc[0], inplace=True)
        self.code_snippets.append("self.data.fillna(self.data.mode().iloc[0], inplace=True)")
        return self.data

    def forward_fill(self):
        """Forward fill missing values."""
        self.data.fillna(method='ffill', inplace=True)
        self.code_snippets.append("self.data.fillna(method='ffill', inplace=True)")
        return self.data

    def backward_fill(self):
        """Backward fill missing values."""
        self.data.fillna(method='bfill', inplace=True)
        self.code_snippets.append("self.data.fillna(method='bfill', inplace=True)")
        return self.data

    def interpolate(self):
        """Interpolate missing values."""
        self.data.interpolate(inplace=True)
        self.code_snippets.append("self.data.interpolate(inplace=True)")
        return self.data

    def simple_imputer(self, strategy='mean'):
        """Impute missing values using SimpleImputer."""
        imputer = SimpleImputer(strategy=strategy)
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        self.code_snippets.append(
            f"imputer = SimpleImputer(strategy='{strategy}')\nself.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)")
        return self.data

    def knn_imputer(self, n_neighbors=5):
        """Impute missing values using KNNImputer."""
        imputer = KNNImputer(n_neighbors=n_neighbors)
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        self.code_snippets.append(
            f"imputer = KNNImputer(n_neighbors={n_neighbors})\nself.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)")
        return self.data

    def impute_most_frequent(self):
        """Impute missing values with the most frequent value."""
        imputer = SimpleImputer(strategy='most_frequent')
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        self.code_snippets.append(
            "imputer = SimpleImputer(strategy='most_frequent')\nself.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)")
        return self.data

    def add_missing_feature(self):
        """Add a binary feature indicating whether the data was missing."""
        for column in self.data.columns:
            self.data[f"{column}_missing"] = self.data[column].isnull().astype(int)
            self.code_snippets.append(f"self.data['{column}_missing'] = self.data['{column}'].isnull().astype(int)")
        return self.data

    def imputation_advice(self):
        """Provide guidance on when to use each imputation technique."""
        advice = {
            "Replace with Constant": "Use when you have a meaningful constant value to replace missing data.",
            "Mean Imputation": "Use for numerical data with MCAR or MAR. Not recommended for skewed distributions.",
            "Median Imputation": "Use for numerical data when dealing with outliers, especially for MAR.",
            "Mode Imputation": "Use for categorical data when missing values are few; suitable for MCAR or MAR.",
            "Forward Fill": "Use for time series data where previous values are expected to carry forward.",
            "Backward Fill": "Use for time series data where future values are expected to fill previous gaps.",
            "Interpolation": "Use when data has a clear trend or pattern, typically for time series.",
            "Simple Imputer": "Use for general-purpose imputation; suitable for various strategies (mean, median, etc.).",
            "KNN Imputer": "Use for more complex relationships between variables; suitable for MAR.",
            "Most Frequent Imputation": "Use when you want to fill missing values with the most common value, typically for categorical data.",
            "Add Missing Feature": "Always consider this to retain information about missingness in your analysis."
        }
        return advice

    def export_code(self):
        """Export generated code snippets."""
        return "\n".join(self.code_snippets)


# Example usage
if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv('your_dataset.csv')

    analyzer = MissingDataAnalyzer(data)

    # Analyze missing data
    analysis_result = analyzer.analyze_missing_data()
    print(analysis_result)

    # Perform pairwise deletion
    pairwise_result = analyzer.pairwise_deletion()
    print("Data after pairwise deletion:")
    print(pairwise_result)

    # Example of replacing missing values
    replaced_data = analyzer.replace_with_mean()
    print("Data after replacing with mean:")
    print(replaced_data)

    # Example of adding a missing feature
    data_with_missing_feature = analyzer.add_missing_feature()
    print("Data with missing feature:")
    print(data_with_missing_feature)

    # Get imputation advice
    imputation_guidelines = analyzer.imputation_advice()
    for method, guidance in imputation_guidelines.items():
        print(f"{method}: {guidance}")

    # Export generated code
    exported_code = analyzer.export_code()
    print("Exported Code:\n")
    print(exported_code)
