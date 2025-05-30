import numpy as np
import pandas as pd
from app.arcticdb_utils import ArcticDBInstance
from app.code_export import build_code_export
from app.utils import classify_type, dict_merge


def get_col_groups(data_id, tenant_id,data):
    valid_corr_cols = []
    valid_str_corr_cols = []
    valid_date_cols = []
    data_instance = ArcticDBInstance(dataset_id=data_id,tenant_id=tenant_id)
    for col_info in data_instance.dtypes:
        name, dtype = map(col_info.get, ["name", "dtype"])
        dtype = classify_type(dtype)
        if dtype in ["I", "F"]:
            valid_corr_cols.append(name)
        elif dtype == "S" and col_info.get("unique_ct", 0) <= 50:
            valid_str_corr_cols.append(name)
        elif dtype == "D":
            date_counts = data[name].dropna().value_counts()
            if len(date_counts[date_counts > 1]) > 1:
                valid_date_cols.append(dict(name=name, rolling=False))
            elif date_counts.eq(1).all():
                valid_date_cols.append(dict(name=name, rolling=True))
    return valid_corr_cols, valid_str_corr_cols, valid_date_cols


def build_matrix(data_id, data, cols, code_formatting_vars=None):
    if data[cols].isnull().values.any():
        data = data.corr(method="pearson")
        code = build_code_export(data_id)
        code.append(
            (
                "corr_cols = [\n"
                "\t'{corr_cols}'\n"
                "]\n"
                "corr_data = df[corr_cols]\n"
                "{str_encodings}"
                "corr_data = corr_data.corr(method='pearson')"
            ).format(
                **dict_merge(
                    {"corr_cols": "", "str_encodings": ""}, code_formatting_vars
                )
            )
        )
    else:
        # using pandas.corr proved to be quite slow on large datasets so I moved to numpy:
        # https://stackoverflow.com/questions/48270953/pandas-corr-and-corrwith-very-slow
        data = np.corrcoef(data[cols].astype("float").values, rowvar=False)
        data = pd.DataFrame(data, columns=cols, index=cols)
        code = build_code_export(
            data_id, imports="import numpy as np\nimport pandas as pd\n\n"
        )
        code.append(
            (
                "corr_cols = [\n"
                "\t'{corr_cols}'\n"
                "]\n"
                "corr_data = df[corr_cols]\n"
                "{str_encodings}"
                "corr_data = np.corrcoef(corr_data.values, rowvar=False)\n"
                "corr_data = pd.DataFrame(corr_data, columns=[corr_cols], index=[corr_cols])"
            ).format(
                **dict_merge(
                    {"corr_cols": "", "str_encodings": ""}, code_formatting_vars
                )
            )
        )

    code = "\n".join(code)
    return data, code


def get_analysis(data_id,tenant_id):
    data_instance = ArcticDBInstance(dataset_id=data_id,tenant_id=tenant_id)
    df = data_instance.get_data()
    valid_corr_cols, _, _ = get_col_groups(data_id, df)
    corr_matrix, _ = build_matrix(
        data_id, df, valid_corr_cols, {"corr_cols": "", "str_encodings": ""}
    )
    corr_matrix = corr_matrix.abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

    score = upper.max(axis=1)
    score.name = "score"
    score = score.sort_values(ascending=False)

    upper = upper.loc[score.index]
    column_name = upper.index[0]
    max_score = score.loc[column_name]
    if pd.isnull(max_score):
        max_score = "N/A"
    upper = upper.fillna(0).to_dict(orient="index")

    missing = df[valid_corr_cols].isna().sum()
    missing.name = "missing"

    analysis = pd.concat([score, missing], axis=1)
    analysis.index.name = "column"
    analysis = analysis.fillna("N/A").reset_index().to_dict(orient="records")

    return column_name, max_score, upper, analysis
