from pydantic import BaseModel, ValidationError, Field

class DynamicJSONSchema(BaseModel):
    document_type: str = Field(..., description="The type/category of the document")
    data: Dict[str, Union[str, int, float, bool, list, dict]] = Field(..., description="Key-value pairs representing the document structure with various data types")

class industry_parser(BaseModel):
    industry: str = Field(description="Suggest name of the industry.")

class target_features(BaseModel):
    features: list[str] = Field(description="Suggest list of the possible target for machine learning model.")

class new_computed_feature(BaseModel):
    new_feature_name : list[str] = Field(description="Name of the new computed feature.")
    description: str = Field(description="How this new feature is derived and why it's useful.")

class features_information(BaseModel):
    name: str =  Field(description="name of the feature.")
    feature_use: str = Field(description=" For each feature, explain its significance in your industry.")
    possible_target: bool = Field(description="Indicate whether the feature can be used as a target variable for any data science tasks specific to your industry.")
    new_computed_features: list[new_computed_feature] =  Field(description="Suggest any new features that could be derived or computed from the existing ones, with an explanation of how these computed features could enhance the model and step to derived")
    alternate_names: list[str] = Field(description="Suggest any alternate names for this feature that is commonly used in your industry.")

class features_information_list(BaseModel):
    features_information_list :list[features_information] = Field(description="list of features_information")


class is_categorical_feature(BaseModel):
    name: str =  Field(description="name of the feature.")
    is_categorical: bool = Field(description="True if it can be considered as categorical and False if it is continuous.")
    reason: str = Field(description="Reason for the selection")


prompts = {
    "find_industry": {
        "title": "find_industry",
        "description": "This template is used to get the industry name from the given data",
        "template": [("system", "You are an expert in feature engineering and your task is to determine the most relevant industry for a given set of feature names"),
                     ("human","{user_query}")],
        "prompt_parser": industry_parser,
        "prompt_args":["user_query"]
    },
    "find_industry_by_msg": {
        "title": "find_industry_by_msg",
        "description": "This template is used to get the industry name from the given user message",
        "template": [("system",
                      "just get industry name from the user message"),
                     ("human", "{user_query}")],
        "prompt_parser": industry_parser,
        "prompt_args": ["user_query"]
    },
    "find_target": {
        "title": "find_target",
        "description": "This template helps identify the best target feature for machine learning based on a provided set of features and a user's specific query.",
        "template": [
            ("system", "You are a highly experienced expert in the {industry} industry. Your task is to analyze the provided feature list and determine the most relevant target feature for machine learning based on the user's query."),
            ("human", "{user_query}")
        ],
        "prompt_parser": target_features,
        "prompt_args": ["industry", "user_query"]
    },
    "features_information": {
        "title": "features_information",
        "description": "This template is used to extract and present feature information related to a specific industry, following the LangGraph Pydantic structure.",
        "template": [
            ("system",
             "You are a highly experienced expert in the {industry} industry. Your task is to analyze the features given by user and provide detailed information as requested by the user. The information should be formatted according to the provided LangGraph Pydantic structure, which includes all relevant details for each feature."),
            ("human", "{user_query}")
        ],
        "prompt_parser": features_information_list,
        "prompt_args": ["industry", "user_query"]
        },
    "is_categorical": {
        "title": "is_categorical",
        "description": "This template is used to classify features as either categorical or continuous based on their type and unique value count.",
        "template": [
            ("system","""You are a skilled Data Scientist and Machine Learning Engineer with extensive experience in the {industry} sector. Your task is to assist in the data cleaning process by determining whether each feature is categorical or continuous. 

        A feature should be classified as categorical if:
        - It is a non-numeric (string, boolean, etc.).
        - If it is numeric (integer), its unique count is less than 10.

        Example of a categorical feature:
        - **Non-numeric (string)**:
            "dtype": "string",
            "feature_use": "This feature is useful in understanding the customer's job stability, which can be used to predict the likelihood of loan defaults.",
            "name": "job",
            "unique_ct": 400
        - **Numeric (integer, unique count less than 10)**:
            "dtype": "int",
            "feature_use": "This feature is useful in understanding the customer's job stability, which can be used to predict the likelihood of loan defaults.",
            "name": "job",
            "unique_ct": 9

        Example of a continuous feature:
        - **Numeric (integer, unique count greater than or equal to 10)**:
            "dtype": "int",
            "feature_use": "This feature is useful in understanding the customer's income, which is useful for predicting their creditworthiness.",
            "name": "income",
            "unique_ct": 150
        Please apply this classification to all features provided and classify them accordingly.""",
             ),
            ("human", "{user_query}")
        ],
        "prompt_parser": is_categorical_feature_list,
        "prompt_args": ["industry", "user_query"]
    },
    "missing_values": {
        "title": "missing_values",
        "description": "This template is used to extract and present feature information related to a specific industry, following the LangGraph Pydantic structure.",
        "template": [
            ("system",
            """ You are a data science engineer and {industry} industry expert. Based on the following advice for handling missing values,
                please recommend from below function with parameter and reason to choose
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
                "delete": "if missing is more than 30% we can delete it but again it depends if it doses on have any significations mostly for continuous data
                "None": "if hasMissing is zero"
                """),
            ("human", "{user_query}")
        ],
        "prompt_parser": missing_value_handel_list,
        "prompt_args": ["industry", "user_query"]
    },
    "missing_values_2": {
        "title": "missing_values_2",
        "description": "This template is used to extract and present feature information related to a specific industry, following the LangGraph Pydantic structure.",
        "template": [
            ("system",
            """ You are a data science engineer and {industry} industry expert. Based on the following advice for handling missing values,
                please recommend from below function with parameter and reason to choose and make sure when hasMissing value is 0 it return None function
                "constant": "Use when you have a meaningful constant value to replace missing data.",
                "mean": "Use for numerical data with MCAR or MAR. Not recommended for skewed distributions.",
                "median": "Use for numerical data when dealing with outliers, especially for MAR.",
                "mode": "Use for categorical data when missing values are few; suitable for MCAR or MAR.",
                "forward_fill": "Use for time series data where previous values are expected to carry forward.",
                "backward_fill": "Use for time series data where future values are expected to fill previous gaps.",
                "Interpolation": "Use when data has a clear trend or pattern, typically for time series.",
                "KNN": "Use for more complex relationships between variables; suitable for MAR.",
                "delete": "if missing is more than 30% we can delete it but again it depends if it doses on have any significations mostly for continuous data
                "None": "if hasMissing value is zero"
                """),
            ("human", "{user_query}")
        ],
        "prompt_parser": missing_value_handel_list,
        "prompt_args": ["industry", "user_query"]
    },
    "outlier_prompt": {
        "title": "outlier_prompt",
        "description": "This template is used to extract and present feature information related to a specific industry, following the Langchain Pydantic structure.",
        "template": [
            ("system",
             """ 
             You are a data scientist and machine learning expert tasked with recommending the most suitable outlier detection method based on the characteristics of a specific feature in a dataset.
             Z-Score: Best for detecting outliers in normally distributed continuous data by measuring how far a point deviates from the mean. Consider this if the feature is continuous, normally distributed, and has low skewness and kurtosis.
             K-Nearest Neighbors (KNN): A versatile method suitable for both continuous and categorical data that identifies outliers based on local density variations. This option may be preferable if the feature is categorical or continuous and exhibits complex distributions.
             Isolation Forest: Effective for high-dimensional datasets, this method isolates anomalies using random partitions. It's a good choice for both continuous and categorical data and can handle large datasets with many features.
             Interquartile Range (IQR): A robust technique for detecting outliers in skewed continuous data by examining the spread of the middle 50% of values. This method is appropriate if the feature is continuous and has high skewness.
             Using the provided attributes, analyze and justify your recommendation for the most suitable outlier detection method from above for the feature.
             """),
            ("human", "{user_query}")
        ],
        "prompt_parser": outlier_handel_list,
        "prompt_args": ["industry", "user_query"]
    },
}


# find_industry_prompt = """
# You are an expert in feature engineering and your task is to determine the most relevant industry for a given set of feature names from provided Json data.
#
# Please analyze the feature names and based on the context, return the most relevant **industry name only**, without any additional text or explanation.
# """
#
#
#
#
# find_target = """
# You are a highly experienced expert in the <industry> industry and your task is to determine the most relevant target feature for a given set of feature names.
#
# Please analyze the feature names and message, return the most relevant **target feature only**, without any additional text or explanation.
# """
#
#
# industry_prompt= """ You are a highly experienced expert in the <industry> industry. Your task is to analyze features from a dataset and provide the following details strictly in **JSON format**:
#
# 1. **Feature Use**: For each feature, explain its significance and how it can be used in a data science context. Also, suggest its potential relationship with the target variable in banking scenarios (e.g., predicting loan defaults, churn, customer segmentation, etc.).
#
# 2. **Possible Target**: Indicate whether the feature can be used as a target variable for any prediction tasks.
#
# 3. **New Computed Features**: Suggest any new features that could be derived or computed from the existing ones, with an explanation of how these computed features could enhance the model.
#
# 4. **Alternate Names**: Suggest any alternate names for this feature commonly used in the industry.
#
# **Output** only the JSON structure without any other text, explanation, or formatting. The output should be formatted as follows:
#
# ```json
# {
#   "features": [
#     {
#       "name": "Feature_1",
#       "feature_use": "Description of how the feature is useful in banking data science and its potential relationship with the target.",
#       "possible_target": "Yes"
#       "new_computed_features": [
#         {
#           "new_feature_name": "Name of the new computed feature",
#           "description": "How this new feature is derived and why it's useful."
#         }
#       ],
#       "alternate_names": ["Alternate_Name_1", "Alternate_Name_2"]
#     }
#   ]
# }
#
# """
#
# categorical_prompt = """
# You are a skilled Data Scientist and Machine Learning Engineer with extensive experience in the <industry> sector. Your task is to assist in the data cleaning process by determining whether each feature is categorical or continuous. A feature should be classified as categorical if its unique count is less than 10, unless industry experts explicitly classify it as continuous based on industry standards.
#
# The JSON data includes the following fields:
#
# Name: The name of the feature
# unique_ct: The number of unique values for this feature
# feature_use: The purpose of this feature in the industry
# Please provide the output strictly in JSON format, without any additional text, explanations, or headings. The expected output format is as follows:
#
# ```json
# {
#   "features": [
#     {
#       "name": "Feature_1",
#       "is_categorical": "Yes"
#     }
#   ]
# }
# """
#
# misssing_prompt = """
# You are a data science engineer and industry expert. Based on the following advice for handling missing values, please recommend the best function when missing value is greather than zero:
# {
#     "Replace with Constant": "Use when you have a meaningful constant value to replace missing data.",
#     "Mean Imputation": "Use for numerical data with MCAR or MAR. Not recommended for skewed distributions.",
#     "Median Imputation": "Use for numerical data when dealing with outliers, especially for MAR.",
#     "Mode Imputation": "Use for categorical data when missing values are few; suitable for MCAR or MAR.",
#     "Forward Fill": "Use for time series data where previous values are expected to carry forward.",
#     "Backward Fill": "Use for time series data where future values are expected to fill previous gaps.",
#     "Interpolation": "Use when data has a clear trend or pattern, typically for time series.",
#     "Simple Imputer": "Use for general-purpose imputation; suitable for various strategies (mean, median, etc.).",
#     "KNN Imputer": "Use for more complex relationships between variables; suitable for MAR.",
#     "Most Frequent Imputation": "Use when you want to fill missing values with the most common value, typically for categorical data.",
#     "Add Missing Feature": "Always consider this to retain information about missingness in your analysis."
#     "delete": "if missing is more than 30% we can delete it but again it depends if it doses on have any significations mostly for continuous data
# }
#
# The dataset includes:
#     Name: The name of the feature
#     unique_ct: The number of unique values for this feature
#     feature_use: The purpose of this feature in the industry
#     hasMissing: count of missing value
#     type: MCAR/MAR/MNAR,
#     is_categorical: Feature is categorical or continuous
#     total_row: Number of records in dataset
#     skew: value of skew,
#     kurt: value of kurt
# Please provide the output strictly in JSON format, without any additional text, explanations, or headings. The expected output format is as follows:
# {
#   "features": [
#     {
#       "name": "Feature_1",
#       "function": "function name" ,
#       "parameter": [],
#     }
#   ]
# }
# """
#
# outlier_prompt = """
# You are a data scientist and machine learning expert tasked with recommending the most suitable outlier detection method based on the characteristics of a specific feature in a dataset. Consider the following attributes of the feature:
#
# Name: The name of the feature.
# unique_ct: The number of unique values for this feature.
# feature_use: The purpose of this feature in the industry (e.g., sales, customer behavior).
# hasMissing: Count of missing values in the feature.
# type: Missing data mechanism (MCAR, MAR, MNAR).
# is_categorical: Indicate whether the feature is categorical or continuous.
# total_row: Total number of records in the dataset.
# skew: Value of skewness (positive, negative, or approximately zero).
# kurt: Value of kurtosis (indicating the tailedness of the distribution).
# Based on these characteristics, evaluate the following outlier detection methods and recommend the most suitable one:
#
# Z-Score: Best for detecting outliers in normally distributed continuous data by measuring how far a point deviates from the mean. Consider this if the feature is continuous, normally distributed, and has low skewness and kurtosis.
#
# K-Nearest Neighbors (KNN): A versatile method suitable for both continuous and categorical data that identifies outliers based on local density variations. This option may be preferable if the feature is categorical or continuous and exhibits complex distributions.
#
# Isolation Forest: Effective for high-dimensional datasets, this method isolates anomalies using random partitions. It's a good choice for both continuous and categorical data and can handle large datasets with many features.
#
# Interquartile Range (IQR): A robust technique for detecting outliers in skewed continuous data by examining the spread of the middle 50% of values. This method is appropriate if the feature is continuous and has high skewness.
#
# Using the provided attributes, analyze and justify your recommendation for the most suitable outlier detection method for the feature.
#
# Provide the output strictly in JSON format without any additional text, explanations, or headings. The format should be as follows:
#
# ```json
# {
#   "features": [
#     {
#       "name": "Feature_1",
#       "outlier_function": "function name from the list"
#     }]
#         """
#
#
# normalization_prompt = """ You are a highly skilled Data Scientist and Engineer a highly experienced expert in the <industry> industry. Your task is to evaluate each feature for the need for normalization and recommend specific `scikit-learn` functions with parameters to address these needs.
#
# For each feature, include:
# 1. **Feature Name**: The name of the feature being evaluated.
# 2. **Action List**: Identify if normalization is required, and if so, specify the recommended `scikit-learn` functions with parameters to apply.
#
# Each action should include:
# - The `scikit-learn` function name.
# - The necessary parameters for that function.
# - A brief explanation of why this function and parameter settings are suitable for this feature.
#
# Return the output strictly in JSON format, without additional text, explanations, or headings. Use the following JSON format:
#
# ```json
# {
#   "features": [
#     {
#       "name": "Feature_1",
#       "actions": [
#         {
#           "issue": "Feature has varying scales, suggesting normalization",
#           "scikit_learn_function": "MinMaxScaler",
#           "parameters": {
#             "feature_range": [0, 1]
#           },
#           "explanation": "MinMaxScaler scales each feature to a given range, commonly 0 to 1, making it useful when features have different ranges."
#         },
#         {
#           "issue": "Feature distribution is highly skewed",
#           "scikit_learn_function": "RobustScaler",
#           "parameters": {
#             "with_centering": true,
#             "with_scaling": true
#           },
#           "explanation": "RobustScaler reduces the effect of outliers by using the interquartile range, making it effective for skewed data."
#         }
#       ]
#     },
#     {
#       "name": "Feature_2",
#       "actions": [
#         {
#           "issue": "Feature shows large variance, indicating need for normalization",
#           "scikit_learn_function": "StandardScaler",
#           "parameters": {
#             "with_mean": true,
#             "with_std": true
#           },
#           "explanation": "StandardScaler standardizes the feature to have zero mean and unit variance, which is suitable when the data has large variance."
#         }
#       ]
#     }
#   ]
# }
# """
#
# new_feature_prompt = """
#    You are a highly skilled Data Scientist and a highly experienced expert in the <industry> industry. Your task is to evaluate the given features to determine if a new feature can be computed.
#
# For each feature, assess whether a new feature can be derived and, if so, suggest a suitable new feature name along with the recommended action for creating it in Python.
#
# For each action, specify:
# - The Python function name.
# - Any required parameters for that function.
# - A brief explanation of why this function and parameter settings are suitable.
#
# Return the output strictly in JSON format without additional text, explanations, or headings. The JSON format should be as follows:
#
# ```json
# {
#   "features": [
#     {
#       "name": "Feature_1",
#       "new_feature_name_suggested": "Derived_Feature_1",
#       "actions": [
#         {
#           "issue": "Feature can benefit from binning for categorical representation",
#           "python_function": "KBinsDiscretizer",
#           "parameters": {
#             "n_bins": 4,
#             "encode": "ordinal",
#             "strategy": "quantile"
#           },
#           "explanation": "KBinsDiscretizer with quantile strategy transforms continuous data into equal-frequency bins, making it suitable for categorical representation."
#         }
#       ]
#     },
#     {
#       "name": "Feature_2",
#       "new_feature_name_suggested": "Log_Feature_2",
#       "actions": [
#         {
#           "issue": "Feature shows high skew, suggesting a log transformation",
#           "python_function": "np.log1p",
#           "parameters": {},
#           "explanation": "Log transformation helps reduce skewness and is useful for features with exponential growth patterns."
#         }
#       ]
#     }
#   ]
# }
# """
#
#
# binning_feature_prompt = """ You are a highly skilled Data Scientist and Engineer and  a highly experienced expert in the <industry> industry. Your task is to assist with the data preparation process by evaluating features for binning suitability and suggesting new, binned feature names where appropriate.
#
# For each feature, suggest a new parameter name and recommend an appropriate action to create it using `scikit-learn`.
#
# For each action, specify:
# - The `scikit-learn` function name.
# - The appropriate parameters for that function.
# - A brief explanation of why this function and parameter settings are suitable for the feature.
#
# Output the response strictly in JSON format without additional text, explanations, or headings. The output format should be as follows:
#
# ```json
# {
#   "features": [
#     {
#       "name": "Feature_1",
#       "new_feature_name_suggested": "Binned_Feature_1",
#       "actions": [
#         {
#           "issue": "Feature requires binning to reduce variance",
#           "python_function": "KBinsDiscretizer",
#           "parameters": {
#             "n_bins": 5,
#             "encode": "ordinal",
#             "strategy": "uniform"
#           },
#           "explanation": "KBinsDiscretizer with uniform strategy is suitable for evenly distributing continuous data into equal-width bins."
#         },
#         {
#           "issue": "Feature has outliers that may affect binning",
#           "python_function": "QuantileTransformer",
#           "parameters": {
#             "n_quantiles": 10,
#             "output_distribution": "uniform"
#           },
#           "explanation": "QuantileTransformer reduces the impact of outliers by creating quantile-based bins that divide data into equal frequencies."
#         }
#       ]
#     },
#     {
#       "name": "Feature_2",
#       "new_feature_name_suggested": "Binned_Feature_2",
#       "actions": [
#         {
#           "issue": "Feature needs equal-frequency bins",
#           "python_function": "KBinsDiscretizer",
#           "parameters": {
#             "n_bins": 4,
#             "encode": "ordinal",
#             "strategy": "quantile"
#           },
#           "explanation": "KBinsDiscretizer with quantile strategy ensures equal frequency in each bin, useful for data with skewed distribution."
#         }
#       ]
#     }
#   ]
# }
# """