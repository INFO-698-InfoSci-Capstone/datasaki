from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from app.datasaki_ai.prompt_manager import *
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

groq_api_key = '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'
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name='llama3-8b-8192')

def get_industry(context):
    user_query = f"Please analyze the feature names and based on the context, return the most relevant industry, Please provide only industry name. context is {context}"
    prompt_template = prompts_db.get_prompt('find_industry')
    groq_chat_s = groq_chat.with_structured_output(prompt_template.output_schema)
    chain = prompt_template.create_prompt() | groq_chat_s
    response = chain.invoke({"user_query": user_query})
    return response.industry

def get_industry_by_msg(user_message):
    user_query = f"Please give me the industry name from the given message, message: {user_message}"
    prompt_template = prompts_db.get_prompt('find_industry_by_msg')
    groq_chat_s = groq_chat.with_structured_output(prompt_template.output_schema)
    chain = prompt_template.create_prompt() | groq_chat_s
    print(chain)
    response = chain.invoke({"user_query": user_query})
    print(response)
    return response.industry

def get_target(context,industry,user_message):
    user_query = f"Please analyze the feature names and message, return the one target feature , without any additional text or explanation.features are {context} and message is {user_message}"
    prompt_template = prompts_db.get_prompt('find_target')
    groq_chat_s = groq_chat.with_structured_output(prompt_template.output_schema)
    chain = prompt_template.create_prompt(industry=industry) | groq_chat_s
    response1= chain.invoke({"user_query":user_query,"industry":industry})
    return response1

def get_feature_details(context,industry):
    user_query = f"""
    The feature list is {context} and i want you to fill the details in provided formate
    """
    prompt_template = prompts_db.get_prompt('features_information')
    groq_chat_s = groq_chat.with_structured_output(prompt_template.output_schema)
    chain = prompt_template.create_prompt(industry=industry) | groq_chat_s
    response2 = chain.invoke({"user_query":user_query,"industry":industry})
    print(response2)
    return response2

def get_missing_values(context,industry):
    user_query = f"""
    The feature list is {context} and i want to find the which function used for missing value handling 
    """
    prompt_template = prompts_db.get_prompt('missing_values_2')
    groq_chat_s = groq_chat.with_structured_output(prompt_template.output_schema)
    chain = prompt_template.create_prompt(industry=industry) | groq_chat_s
    response = chain.invoke({"user_query":user_query,"industry":industry})
    res = []
    for value in response.feature_list:
        value = value.dict()
        res.append({"name":value["name"],"function":value["function"],"replace_with":value["function"],"function_list":["constant","mean","mode","median","knn","forward_fill","backward_fill","None"],"function_args":{"knn":{"n_neighbors":5},"constant":{"value":0}}})
    return res

def get_outliers_values(context,industry):
    user_query = f"""Please provide best outlier function based on given feature details
            The dataset includes:
                name: The name of the feature
                unique_ct: The number of unique values for this feature
                feature_use: The purpose of this feature in the industry
                hasMissing: count of missing value
                type: MCAR/MAR/MNAR,
                is_categorical: categorical if True and continuous if False
                total_row: Number of records in dataset
                skew: value of skew,
                kurt: value of kurt
            and dataset is {context}
    """
    prompt_template = prompts_db.get_prompt('outlier_prompt')
    groq_chat_s = groq_chat.with_structured_output(prompt_template.output_schema)
    chain = prompt_template.create_prompt(industry=industry) | groq_chat_s
    response = chain.invoke({"user_query":user_query,"industry":industry})
    return response.feature_list.dict()

#
# data = """[
#   {
#     "dtype": "string",
#     "feature_use": "This feature is useful in understanding the financial stability of the customer. It can be used to predict loan defaults, as customers with low or no checking balance may be more likely to default on their loans.",
#     "name": "checking_balance",
#     "unique_ct": 4
#   },
#   {
#     "dtype": "int64",
#     "feature_use": "This feature is useful in understanding the customer's credit history and behavior. It can be used to predict the likelihood of loan defaults, as longer loan durations may indicate a higher risk of default.",
#     "name": "months_loan_duration",
#     "unique_ct": 33
#   },
#   {
#     "dtype": "string",
#     "feature_use": "This feature is useful in understanding the customer's credit history and behavior. It can be used to predict the likelihood of loan defaults, as customers with a poor credit history may be more likely to default on their loans.",
#     "name": "credit_history",
#     "unique_ct": 5
#   },
#   {
#     "dtype": "string",
#     "feature_use": "This feature is useful in understanding the purpose of the loan, which can be used to predict the likelihood of loan defaults. For example, loans taken for consumption purposes may have a higher risk of default.",
#     "name": "purpose",
#     "unique_ct": 6
#   },
#   {
#     "dtype": "int64",
#     "feature_use": "This feature is useful in understanding the loan amount, which can be used to predict the likelihood of loan defaults. Larger loans may have a higher risk of default.",
#     "name": "amount",
#     "unique_ct": 921
#   },
#   {
#     "dtype": "string",
#     "feature_use": "This feature is useful in understanding the customer's financial stability, which can be used to predict the likelihood of loan defaults.",
#     "name": "savings_balance",
#     "unique_ct": 5
#   },
#   {
#     "dtype": "string",
#     "feature_use": "This feature is useful in understanding the customer's employment history, which can be used to predict the likelihood of loan defaults. Employees with shorter employment durations may be more likely to default on their loans.",
#     "name": "employment_duration",
#     "unique_ct": 5
#   },
#   {
#     "dtype": "int64",
#     "feature_use": "This feature is useful in understanding the customer's debt-to-income ratio, which can be used to predict the likelihood of loan defaults.",
#     "name": "percent_of_income",
#     "unique_ct": 4
#   },
#   {
#     "dtype": "int64",
#     "feature_use": "This feature is useful in understanding the customer's stability of residence, which can be used to predict the likelihood of loan defaults.",
#     "name": "years_at_residence",
#     "unique_ct": 4
#   },
#   {
#     "dtype": "int64",
#     "feature_use": "This feature is useful in understanding the customer's age, which can be used to predict the likelihood of loan defaults. Older customers may be more likely to default on their loans.",
#     "name": "age",
#     "unique_ct": 53
#   },
#   {
#     "dtype": "string",
#     "feature_use": "This feature is useful in understanding the customer's credit behavior, which can be used to predict the likelihood of loan defaults.",
#     "name": "other_credit",
#     "unique_ct": 3
#   },
#   {
#     "dtype": "string",
#     "feature_use": "This feature is useful in understanding the customer's housing situation, which can be used to predict the likelihood of loan defaults.",
#     "name": "housing",
#     "unique_ct": 3
#   },
#   {
#     "dtype": "int64",
#     "feature_use": "This feature is useful in understanding the customer's existing loan portfolio, which can be used to predict the likelihood of loan defaults.",
#     "name": "existing_loans_count",
#     "unique_ct": 4
#   },
#   {
#     "dtype": "string",
#     "feature_use": "This feature is useful in understanding the customer's job stability, which can be used to predict the likelihood of loan defaults.",
#     "name": "job",
#     "unique_ct": 4
#   },
#   {
#     "dtype": "int64",
#     "feature_use": "This feature is useful in understanding the customer's dependents, which can be used to predict the likelihood of loan defaults.",
#     "name": "dependents",
#     "unique_ct": 2
#   },
#   {
#     "dtype": "string",
#     "feature_use": "This feature is useful in understanding the customer's contact information, which can be used to predict the likelihood of loan defaults.",
#     "name": "phone",
#     "unique_ct": 2
#   },
#   {
#     "dtype": "string",
#     "feature_use": "This feature is the target variable, indicating whether the customer has defaulted on their loan or not.",
#     "name": "default",
#     "unique_ct": 2
#   }
# ]"""
#
# user_query = f""" Please use the json data and fill the details in provided formate
# The JSON data includes the following fields:
# Name: The name of the feature
# unique_ct: The number of unique values for this feature
# type: type of the feature
# feature_use: The purpose of this feature in the industry
# **There should be entry for every data feature in response what ever it is  categorical feature or continuous feature**
# Data : {data}
# """
#
# prompt_template = prompts_db.get_prompt('is_categorical')
# groq_chat_s = groq_chat.with_structured_output(prompt_template.output_schema)
# chain = prompt_template.create_prompt(industry=response.industry) | groq_chat_s
# response3 = chain.invoke({"user_query":user_query,"industry":response.industry})
# print(response3)
#
# data = """
# [
# [
#   {
#     "feature_use": "This feature is useful in understanding the financial stability of the customer. It can be used to predict loan defaults, as customers with low or no checking balance may be more likely to default on their loans.",
#     "hasMissing": 0,
#     "is_categorical": "Yes",
#     "kurt": null,
#     "name": "checking_balance",
#     "skew": null,
#     "unique_ct": 4
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's credit history and behavior. It can be used to predict the likelihood of loan defaults, as longer loan durations may indicate a higher risk of default.",
#     "hasMissing": 0,
#     "is_categorical": "No",
#     "kurt": 0.9197813600546372,
#     "name": "months_loan_duration",
#     "skew": 1.0941841715555418,
#     "unique_ct": 33
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's credit history and behavior. It can be used to predict the likelihood of loan defaults, as customers with a poor credit history may be more likely to default on their loans.",
#     "hasMissing": 0,
#     "is_categorical": "Yes",
#     "kurt": null,
#     "name": "credit_history",
#     "skew": null,
#     "unique_ct": 5
#   },
#   {
#     "feature_use": "This feature is useful in understanding the purpose of the loan, which can be used to predict the likelihood of loan defaults. For example, loans taken for consumption purposes may have a higher risk of default.",
#     "hasMissing": 0,
#     "is_categorical": "Yes",
#     "kurt": null,
#     "name": "purpose",
#     "skew": null,
#     "unique_ct": 6
#   },
#   {
#     "feature_use": "This feature is useful in understanding the loan amount, which can be used to predict the likelihood of loan defaults. Larger loans may have a higher risk of default.",
#     "hasMissing": 0,
#     "is_categorical": "No",
#     "kurt": 4.29259030804851,
#     "name": "amount",
#     "skew": 1.9496276798326209,
#     "unique_ct": 921
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's financial stability, which can be used to predict the likelihood of loan defaults.",
#     "hasMissing": 0,
#     "is_categorical": "Yes",
#     "kurt": null,
#     "name": "savings_balance",
#     "skew": null,
#     "unique_ct": 5
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's employment history, which can be used to predict the likelihood of loan defaults. Employees with shorter employment durations may be more likely to default on their loans.",
#     "hasMissing": 0,
#     "is_categorical": "Yes",
#     "kurt": null,
#     "name": "employment_duration",
#     "skew": null,
#     "unique_ct": 5
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's debt-to-income ratio, which can be used to predict the likelihood of loan defaults.",
#     "hasMissing": 0,
#     "is_categorical": "No",
#     "kurt": -1.2104731179379704,
#     "name": "percent_of_income",
#     "skew": -0.5313481143125486,
#     "unique_ct": 4
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's stability of residence, which can be used to predict the likelihood of loan defaults.",
#     "hasMissing": 0,
#     "is_categorical": "No",
#     "kurt": -1.3814485027493704,
#     "name": "years_at_residence",
#     "skew": -0.2725698140337228,
#     "unique_ct": 4
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's age, which can be used to predict the likelihood of loan defaults. Older customers may be more likely to default on their loans.",
#     "hasMissing": 0,
#     "is_categorical": "No",
#     "kurt": 0.5957795670766881,
#     "name": "age",
#     "skew": 1.0207392686768317,
#     "unique_ct": 53
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's credit behavior, which can be used to predict the likelihood of loan defaults.",
#     "hasMissing": 0,
#     "is_categorical": "Yes",
#     "kurt": null,
#     "name": "other_credit",
#     "skew": null,
#     "unique_ct": 3
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's housing situation, which can be used to predict the likelihood of loan defaults.",
#     "hasMissing": 0,
#     "is_categorical": "Yes",
#     "kurt": null,
#     "name": "housing",
#     "skew": null,
#     "unique_ct": 3
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's existing loan portfolio, which can be used to predict the likelihood of loan defaults.",
#     "hasMissing": 0,
#     "is_categorical": "No",
#     "kurt": 1.6044393724243444,
#     "name": "existing_loans_count",
#     "skew": 1.2725759670020926,
#     "unique_ct": 4
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's job stability, which can be used to predict the likelihood of loan defaults.",
#     "hasMissing": 0,
#     "is_categorical": "Yes",
#     "kurt": null,
#     "name": "job",
#     "skew": null,
#     "unique_ct": 4
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's dependents, which can be used to predict the likelihood of loan defaults.",
#     "hasMissing": 0,
#     "is_categorical": "No",
#     "kurt": 1.6492736936698598,
#     "name": "dependents",
#     "skew": 1.909444721297485,
#     "unique_ct": 2
#   },
#   {
#     "feature_use": "This feature is useful in understanding the customer's contact information, which can be used to predict the likelihood of loan defaults.",
#     "hasMissing": 0,
#     "is_categorical": "Yes",
#     "kurt": null,
#     "name": "phone",
#     "skew": null,
#     "unique_ct": 2
#   },
#   {
#     "feature_use": "This feature is the target variable, indicating whether the customer has defaulted on their loan or not.",
#     "hasMissing": 0,
#     "is_categorical": "Yes",
#     "kurt": null,
#     "name": "default",
#     "skew": null,
#     "unique_ct": 2
#   }
# ]
# """
# user_query = f""" Please provide based on given feature details
#             The dataset includes:
#                 Name: The name of the feature
#                 unique_ct: The number of unique values for this feature
#                 feature_use: The purpose of this feature in the industry
#                 hasMissing: count of missing value
#                 type: MCAR/MAR/MNAR,
#                 is_categorical: Feature is categorical or continuous
#                 total_row: Number of records in dataset
#                 skew: value of skew,
#                 kurt: value of kurt
#             and dataset is {data}"""
#
# prompt_template = prompts_db.get_prompt('missing_values')
# groq_chat_s = groq_chat.with_structured_output(prompt_template.output_schema)
# chain = prompt_template.create_prompt(industry=response.industry) | groq_chat_s
# response4 = chain.invoke({"user_query":user_query,"industry":response.industry})
# print(response4)
#
#
# prompt_template = prompts_db.get_prompt('outlier_prompt')
# groq_chat_s = groq_chat.with_structured_output(prompt_template.output_schema)
# chain = prompt_template.create_prompt(industry=response.industry) | groq_chat_s
# response5 = chain.invoke({"user_query":user_query,"industry":response.industry})
# print(response5)