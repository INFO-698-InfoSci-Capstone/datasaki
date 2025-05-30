import pandas as pd

from scipy import stats

from app.arcticdb_utils import ArcticDBInstance
import app.pandas_util as pandas_util
from app.query import run_query
from app.utils import make_list


def flatten_columns(df, columns=None):
    if columns is not None:
        return [
            " ".join(
                [
                    "{}-{}".format(c1, str(c2))
                    for c1, c2 in zip(make_list(columns), make_list(col_val))
                ]
            ).strip()
            for col_val in df.columns.values
        ]
    return [
        " ".join([str(c) for c in make_list(col)]).strip() for col in df.columns.values
    ]


class DataReshaper(object):
    def __init__(self, data_id, tenant_id,shape_type, cfg):
        self.tenant_id = tenant_id
        self.data_id = data_id
        if shape_type == "pivot":
            self.builder = PivotBuilder(cfg)
        elif shape_type == "aggregate":
            self.builder = AggregateBuilder(cfg)
        elif shape_type == "transpose":
            self.builder = TransposeBuilder(cfg)
        elif shape_type == "resample":
            self.builder = ResampleBuilder(cfg)
        else:
            raise NotImplementedError(
                "{} data re-shaper not implemented yet!".format(shape_type)
            )
    def reshape(self):
        data_instance = ArcticDBInstance(dataset_id=self.data_id, tenant_id=self.tenant_id)
        data = run_query(
            data_instance.get_data(),
            data_instance.get_query(),
            data_instance.context_variables,
        )
        return self.builder.reshape(data)

    def build_code(self):
        return self.builder.build_code()


class PivotBuilder(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def reshape(self, data):
        index, columns, values, aggfunc = (
            self.cfg.get(p) for p in ["index", "columns", "values", "aggfunc"]
        )
        pivot_data = pd.pivot_table(
            data, values=values, index=index, columns=columns, aggfunc=aggfunc
        )
        if len(values) == 1:
            pivot_data.columns = pivot_data.columns.droplevel(0)
        if self.cfg.get("columnNameHeaders", False):
            pivot_data.columns = flatten_columns(pivot_data, columns=columns)
        else:
            pivot_data.columns = flatten_columns(pivot_data)
        pivot_data = pivot_data.rename_axis(None, axis=1)
        return pivot_data

    def build_code(self):
        index, columns, values, aggfunc = (
            self.cfg.get(p) for p in ["index", "columns", "values", "aggfunc"]
        )
        code = []
        if aggfunc is not None or len(values) > 1:
            code.append(
                "df = pd.pivot_table(df, index='{}', columns='{}', values=['{}'], aggfunc='{}')".format(
                    index, columns, "', '".join(values), aggfunc
                )
            )
            if len(values) > 1:
                code.append(
                    "df.columns = [' '.join([str(c) for c in col]).strip() for col in df.columns.values]"
                )
            elif len(values) == 1:
                code.append("df.columns = df.columns.droplevel(0)")
        else:
            code.append(
                "df = df.pivot(index='{index}', columns='{columns}', values='{values}')".format(
                    index=index, columns=columns, values=values[0]
                )
            )
        code.append("df = df.rename_axis(None, axis=1)")
        return "\n".join(code)


def str_joiner(vals, join_char="|"):
    return join_char.join(vals)


def custom_agg_handler(agg):
    if agg == "gmean":
        return stats.gmean
    if agg == "str_joiner":
        return str_joiner
    return agg


def custom_aggregate_handler(cols):
    return {
        col: [custom_agg_handler(agg) for agg in aggs] for col, aggs in cols.items()
    }


def custom_str_handler(aggs):
    def _handler():
        for agg in aggs:
            if agg == "gmean":
                yield agg
            elif agg == "str_joiner":
                yield "'|'.join"
            else:
                yield "'{}'".format(agg)

    return list(_handler())


class AggregateBuilder(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def reshape(self, data):
        index, agg, dropna = (self.cfg.get(p) for p in ["index", "agg", "dropna"])
        agg_type, func, cols = (agg.get(p) for p in ["type", "func", "cols"])

        if index:
            agg_data = pandas_util.groupby(
                data, index, dropna=dropna if dropna is not None else True
            )
            if agg_type == "func":
                if "count_pct" == func:
                    counts = agg_data.size()
                    return pd.DataFrame(
                        {"Count": counts, "Percentage": (counts / len(data)) * 100}
                    )
                if cols:
                    agg_data = agg_data[cols]
                elif pandas_util.is_pandas2():
                    non_str_cols = [
                        c
                        for c in data.select_dtypes(exclude="object").columns
                        if hasattr(agg_data, c) and c != index
                    ]
                    agg_data = agg_data[non_str_cols]

                return (
                    agg_data.agg(stats.gmean)
                    if func == "gmean"
                    else getattr(agg_data, func)()
                )
            agg_data = agg_data.aggregate(custom_aggregate_handler(cols))
            agg_data.columns = flatten_columns(agg_data)
            return agg_data

        agg_data = data[cols] if cols else data
        if agg_type == "func":
            agg_data = (
                agg_data.apply(stats.gmean)
                if func == "gmean"
                else getattr(agg_data, func)()
            )
            return agg_data.to_frame().T

        agg_data = agg_data.aggregate(custom_aggregate_handler(cols))
        agg_data = agg_data.to_frame().T
        return agg_data

    def build_code(self):
        index, agg, dropna = (self.cfg.get(p) for p in ["index", "agg", "dropna"])
        dropna = dropna if dropna is not None else True
        agg_type, func, cols = (agg.get(p) for p in ["type", "func", "cols"])
        code = []
        if (agg_type == "func" and func == "gmean") or (
            agg_type != "func" and "gmean" in cols.values()
        ):
            code.append("\nfrom scipy.stats import gmean\n\n")

        if index:
            index = "', '".join(index)
            if agg_type == "func":
                if "count_pct" == agg:
                    code.append(
                        (
                            "total_records = len(df)\n"
                            "df = df{groupby}\n"
                            "counts = df.size()\n"
                            "df = pd.DataFrame({'Count': counts, 'Percentage': (counts / total_records) * 100})"
                        )
                    )
                    return code
                agg_str = ".agg(gmean)" if agg == "gmean" else ".{}()".format(agg)
                if cols is not None:
                    code.append(
                        "df = df{groupby}['{columns}']{agg}".format(
                            groupby=pandas_util.groupby_code(index, dropna=dropna),
                            columns="', '".join(cols),
                            agg=agg_str,
                        )
                    )
                    return code
                code.append(
                    "df = df{groupby}{agg}".format(
                        groupby=pandas_util.groupby_code(index, dropna=dropna),
                        agg=agg_str,
                    )
                )
                return code
            code += [
                "df = df{groupby}.aggregate(".format(
                    groupby=pandas_util.groupby_code(index, dropna=dropna)
                )
                + "{",
                ",\n".join(
                    "\t'{col}': ['{aggs}']".format(
                        col=col, aggs=", ".join(custom_str_handler(aggs))
                    )
                    for col, aggs in cols.items()
                ),
                "})",
                "df.columns = [' '.join([str(c) for c in col]).strip() for col in df.columns.values]",
            ]
            return "\n".join(code)

        if cols:
            code.append("df = df[[{}]]".format("', '".join(cols)))
        if agg_type == "func":
            agg_str = ".apply(gmean)" if agg == "gmean" else ".{}()".format(agg)
            code += ["df = df{}".format(agg_str), "df = df.to_frame().T"]
            return code
        code += [
            "df = df.aggregate({"
            + ",\n".join(
                "\t'{col}': ['{aggs}']".format(
                    col=col, aggs=", ".join(custom_agg_handler(aggs))
                )
                for col, aggs in cols.items()
            )
            + "})",
            "df = df.to_frame().T",
        ]
        return code


class TransposeBuilder(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def reshape(self, data):
        index, columns = (self.cfg.get(p) for p in ["index", "columns"])
        t_data = data.set_index(index)
        if any(t_data.index.duplicated()):
            raise Exception(
                "Transposed data contains duplicates, please specify additional index or filtering"
            )
        if columns is not None:
            t_data = t_data[columns]
        t_data = t_data.T
        if len(index) > 1:
            t_data.columns = flatten_columns(t_data)
        t_data = t_data.rename_axis(None, axis=1)
        return t_data

    def build_code(self):
        index, columns = (self.cfg.get(p) for p in ["index", "columns"])

        code = []
        if columns is not None:
            code.append(
                "df = df.set_index('{}')['{}'].T".format(
                    "', '".join(index), "', '".join(columns)
                )
            )
        else:
            code.append("df = df.set_index('{}').T".format("', '".join(index)))
        if len(index) > 1:
            code.append(
                "df.columns = [' '.join([str(c) for c in col]).strip() for col in df.columns.values]"
            )
        code.append("df = df.rename_axis(None, axis=1)")
        return "\n".join(code)


class ResampleBuilder(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def reshape(self, data):
        index, columns, freq, agg = (
            self.cfg.get(p) for p in ["index", "columns", "freq", "agg"]
        )
        t_data = data.set_index(index)
        if columns is not None:
            t_data = t_data[columns]
        t_data = getattr(t_data.resample(freq), agg)()
        if not columns or len(columns) > 1:
            t_data.columns = flatten_columns(t_data)
        t_data.index.name = "{}_{}".format(index, freq)
        t_data = t_data.reset_index()
        return t_data

    def build_code(self):
        index, columns, freq, agg = (
            self.cfg.get(p) for p in ["index", "columns", "freq", "agg"]
        )
        code = []
        if columns is not None:
            code.append(
                "df = df.set_index('{}')['{}'].resample('{}').{}()".format(
                    index, "', '".join(columns), freq, agg
                )
            )
        else:
            code.append(
                "df = df.set_index('{}').resample('{}').{}()".format(index, freq, agg)
            )
        if not columns or len(columns) > 1:
            code.append(
                "df.columns = [' '.join([str(c) for c in col]).strip() for col in df.columns.values]"
            )
        return "\n".join(code)
