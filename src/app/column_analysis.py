import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import pprint
import scipy.stats as sts
import time

from app.arcticdb_utils import ArcticDBInstance
from app.code_export import build_code_export, build_final_chart_code
from app.column_builders import clean, clean_code
from app.describe import load_describe
from app.query import build_query, load_filterable_data
from app.utils import (
    apply,
    classify_type,
    find_dtype,
    find_dtype_formatter,
    find_selected_column,
    get_bool_arg,
    get_int_arg,
    get_str_arg,
    grid_columns,
    grid_formatter,
    json_float,
    json_timestamp,
    make_list,
)


LINE_CFG = "line={'shape': 'spline', 'smoothing': 0.3}, mode='lines'"


def handle_top(df, top):
    if top is not None:
        top = int(top)
        if top > 0:
            return df[:top], top, ["chart = chart[:{}]".format(top)]
        else:
            return df[top:], top, ["chart = chart[{}:]".format(top)]
    elif len(df) > 100:
        top = 100
        return df[:top], top, ["chart = chart[:100]"]
    return df, len(df), []


def pctsum_updates(df, group, ret_col):
    grp_sums = df.groupby(group)[[ret_col]].sum()
    code = (
        "ordinal_data = df.groupby('{col}')[['{ret_col}']].sum()\n"
        "ordinal_data = ordinal_data / ordinal_data.sum()"
    ).format(col=group, ret_col=ret_col)
    return grp_sums / grp_sums.sum(), [code]


def handle_cleaners(s, cleaners):
    cleaner_code = []
    if cleaners:
        for cleaner in cleaners.split(","):
            s = clean(s, cleaner, {})
            cleaner_code += clean_code(cleaner, {})
    return s, cleaner_code


def build_kde(s, hist_labels, selected_col):
    try:
        kde = sts.gaussian_kde(s)
        kde_data = kde.pdf(hist_labels)
        kde_data = [json_float(k, precision=12) for k in kde_data]
        code = [
            "import scipy.stats as sts",
            "kde = sts.gaussian_kde(s['{}'])".format(selected_col),
            "kde_data = kde.pdf(np.linspace(labels.min(), labels.max()))",
        ]
        return kde_data, code
    except np.linalg.LinAlgError:
        return None, []


class ColumnAnalysis(object):
    def __init__(self, data_id,tenant_id, req):
        self.data_id = data_id
        self.tenant_id = tenant_id
        self.data_instance = ArcticDBInstance(dataset_id=data_id,tenant_id=tenant_id)
        self.analysis_type = get_str_arg(req, "type")
        self.query = build_query(data_id, tenant_id=tenant_id,query=self.data_instance.get_query())
        self.data = load_filterable_data(data_id,tenant_id=tenant_id,req=req, query=self.query)
        self.selected_col = find_selected_column(
            self.data, get_str_arg(req, "col", "values")
        )
        if self.analysis_type != "frequency":
            self.data = self.data[~pd.isnull(self.data[self.selected_col])]
        self.dtype = find_dtype(self.data[self.selected_col])
        self.classifier = classify_type(self.dtype)
        self.code = build_code_export(
            data_id,
            tenant_id,
            imports="{}\n".format(
                "\n".join(
                    [
                        "import numpy as np",
                        "import pandas as pd",
                        "import plotly.graph_objs as go",
                    ]
                )
            ),
        )

        if self.analysis_type is None:
            self.analysis_type = (
                "histogram" if self.classifier in ["F", "I", "D"] else "value_counts"
            )

        if self.analysis_type == "geolocation":
            self.analysis = GeolocationAnalysis(req)
        elif self.analysis_type == "histogram":
            self.analysis = HistogramAnalysis(req)
        elif self.analysis_type == "categories":
            self.analysis = CategoryAnalysis(req)
        elif self.analysis_type == "value_counts":
            self.analysis = ValueCountAnalysis(req)
        elif self.analysis_type == "word_value_counts":
            self.analysis = WordValueCountAnalysis(req)
        elif self.analysis_type == "qq":
            self.analysis = QQAnalysis()
        elif self.analysis_type == "frequency":
            self.analysis = FrequencyAnalysis(req)

    def build(self):
        base_code = build_code_export(
            self.data_id,
            self.tenant_id,
            imports="{}\n\n".format(
                "\n".join(
                    [
                        "import numpy as np",
                        "import pandas as pd",
                        "import plotly.graph_objs as go",
                    ]
                )
            ),
        )
        return_data, code = self.analysis.build(self)
        return dict(
            code=build_final_chart_code(base_code + code),
            query=self.query,
            cols=self.data_instance.dtypes,
            dtype=self.dtype,
            chart_type=self.analysis_type,
            timestamp=round(time.time() * 1000),
            **return_data
        )


class HistogramAnalysis(object):
    def __init__(self, req):
        self.bins = req.get('bins',20)
        self.target = req.get('target',None)
        self.density = req.get('density')

    def build_histogram_data(self, series):
        hist_kwargs = {"density": True} if self.density else {"bins": self.bins}
        hist_data, hist_labels = np.histogram(series, **hist_kwargs)
        hist_data = [json_float(h) for h in hist_data]
        decimals = 1
        # drop the first bin because of just a minimum
        labels = [
            "".join(["{0:.", "{}".format(decimals), "f}"]).format(lbl)
            for lbl in hist_labels[1:]
        ]
        while len(set(labels)) < len(labels) or decimals > 10:
            decimals += 1
            labels = [
                "".join(["{0:.", "{}".format(decimals), "f}"]).format(lbl)
                for lbl in hist_labels[1:]
            ]

        return dict(labels=labels, data=hist_data), hist_labels

    def build(self, parent):
        if parent.classifier == "D":
            parent.data.loc[:, parent.selected_col] = apply(
                parent.data[parent.selected_col], json_timestamp
            )
        kde_code = []
        if self.target is None:
            return_data, hist_labels = self.build_histogram_data(
                parent.data[parent.selected_col]
            )
            kde, kde_code = build_kde(
                parent.data[parent.selected_col], hist_labels, parent.selected_col
            )
            if kde is not None:
                return_data["kde"] = kde
        else:
            bin_vals = pd.cut(parent.data[parent.selected_col], bins=self.bins)
            labels = ["{}".format(c) for c in bin_vals.dtype.categories]
            parent.data.loc[:, "bin"] = bin_vals.astype("str")
            return_data = {"targets": [], "labels": labels}
            target_dtype = find_dtype(parent.data[self.target])
            target_formatter = find_dtype_formatter(target_dtype)
            for target, target_data in parent.data[[self.target, "bin"]].groupby(
                self.target
            ):
                target_counts = target_data["bin"].value_counts()
                target_counts = [
                    int(tc) for tc in target_counts.reindex(labels, fill_value=0).values
                ]
                return_data["targets"].append(
                    dict(
                        target=target_formatter(target, as_string=True),
                        data=target_counts,
                    )
                )

        desc, desc_code = load_describe(parent.data[parent.selected_col])


        for p in ["skew", "kurt"]:
            if p in dtype_info:
                desc[p] = dtype_info[p]

        return_data["desc"] = desc
        return return_data, self._build_code(parent, kde_code, desc_code)

    def _build_code(self, parent, kde_code, desc_code):
        pp = pprint.PrettyPrinter(indent=4)

        code = [
            "s = df[~pd.isnull(df['{col}'])][['{col}'{target}]]".format(
                col=parent.selected_col,
                target=",'{}'".format(self.target) if self.target else "",
            )
        ]
        if parent.classifier == "D":
            code.append(
                (
                    "\nimport time\n\n"
                    "s.loc[:, '{col}'] = s['{col}'].apply(\n"
                    "\tlambda x: int((time.mktime(x.timetuple()) + (old_div(x.microsecond, 1000000.0))) * 1000\n"
                    ")"
                ).format(col=parent.selected_col)
            )
        if self.target is None:
            hist_kwargs = (
                "density=True" if self.density else "bins={}".format(self.bins)
            )
            code.append(
                "chart, labels = np.histogram(s['{col}'], {hist_kwargs})".format(
                    col=parent.selected_col, hist_kwargs=hist_kwargs
                )
            )
            code += kde_code + desc_code
            layout = pp.pformat(
                go.Layout(
                    **{
                        "barmode": "group",
                        "legend": {"orientation": "h"},
                        "title": {
                            "text": "{} Histogram (bins: {}) w/ KDE".format(
                                parent.selected_col, self.bins
                            )
                        },
                        "xaxis2": {"anchor": "y", "overlaying": "x", "side": "top"},
                        "yaxis": {"title": {"text": "Frequency"}, "side": "left"},
                        "yaxis2": {
                            "title": {"text": "KDE"},
                            "side": "right",
                            "overlaying": "y",
                        },
                    }
                )
            )
            code += [
                "charts = [",
                "\tgo.Bar(x=labels[1:], y=chart, name='Histogram'),",
                "\tgo.Scatter(",
                "\t\tx=list(range(len(kde_data))), y=kde_data, name='KDE',"
                "\t\tyaxis='y2', xaxis='x2',"
                "\t\t{}".format(LINE_CFG),
                "\t)",
                "]",
            ]
        else:
            layout = pp.pformat(
                go.Layout(
                    **{
                        "barmode": "stack",
                        "legend": {"orientation": "h"},
                        "title": {
                            "text": "{} Histogram (bins: {}) w/ target ({})".format(
                                parent.selected_col, self.bins, self.target
                            )
                        },
                        "yaxis": {"title": {"text": "Frequency"}, "side": "left"},
                    }
                )
            )
            code.append(
                (
                    "bin_vals = pd.cut(s['{col}'], bins={bins})\n"
                    "labels = [str(c) for c in bin_vals.dtype.categories]\n"
                    "s.loc[:, 'bin'] = bin_vals.astype('str')\n"
                    "for target, target_data in s[['{target}', 'bin']].groupby('{target}'):\n"
                    "\ttarget_counts = target_data['bin'].value_counts()\n"
                    "\ttarget_counts = [int(tc) for tc in target_counts.reindex(labels, fill_value=0).values]\n"
                    "\tcharts.append(go.Bar(x=labels, y=target_counts, name=target))"
                ).format(col=parent.selected_col, bins=self.bins, target=self.target)
            )
        code.append(
            "figure = go.Figure(data=charts, layout=go.{layout})".format(layout=layout)
        )
        return code


class CategoryAnalysis(object):
    def __init__(self, req):
        self.category_col = req.get("categoryCol")
        self.category_agg =  req.get("categoryAgg","mean")
        self.aggs = [
            "count",
            "sum" if self.category_agg == "pctsum" else self.category_agg,
        ]
        self.top = req.get("top")

    def build(self, parent):
        hist = pandas_util.groupby(parent.data, self.category_col, dropna=False)[
            [parent.selected_col]
        ].agg(self.aggs)
        hist.columns = hist.columns.droplevel(0)
        hist.columns = ["count", "data"]
        if self.category_agg == "pctsum":
            hist["data"] = hist["data"] / hist["data"].sum()
        hist.index.name = "labels"
        hist = hist.reset_index()
        hist, top, top_code = handle_top(hist, self.top)
        f = grid_formatter(grid_columns(hist), nan_display="NaN")
        return_data = f.format_lists(hist)
        return_data["top"] = top
        return return_data, self._build_code(parent, top_code)

    def _build_code(self, parent, top_code):
        pp = pprint.PrettyPrinter(indent=4)
        layout = pp.pformat(
            go.Layout(
                **{
                    "barmode": "group",
                    "legend": {"orientation": "h"},
                    "title": {
                        "text": "{}({}) Categorized by {}".format(
                            parent.selected_col, self.category_agg, self.category_col
                        )
                    },
                    "xaxis": {"title": {"text": self.category_col}},
                    "yaxis": {
                        "title": {
                            "text": "{} ({})".format(
                                parent.selected_col, self.category_agg
                            )
                        },
                        "side": "left",
                    },
                    "yaxis2": {
                        "title": {"text": "Frequency"},
                        "side": "right",
                        "overlaying": "y",
                    },
                }
            )
        )

        code = [
            "chart = df{groupby}[['{col}']].agg(['{aggs}'])".format(
                groupby=pandas_util.groupby_code(make_list(self.category_col), False),
                col=parent.selected_col,
                aggs="', '".join(self.aggs),
            ),
            "chart.columns = chart.columns.droplevel(0)",
            'chart.columns = ["count", "data"]',
        ]
        if self.category_agg == "pctsum":
            code.append("chart['data'] = chart['data'] / chart['data'].sum()")
        code += ["chart.index.name = 'labels'", "chart = chart.reset_index()"]
        code += top_code
        code += [
            "charts = [",
            "\tgo.Bar(x=chart['labels'].values, y=chart['data'].values),",
            "\tgo.Scatter(",
            "\t\tx=chart['labels'].values, y=chart['count'].values, yaxis='y2',",
            "\t\tname='Frequency', {}".format(LINE_CFG),
            "\t)",
            "]",
            "figure = go.Figure(data=charts, layout=go.{layout})".format(layout=layout),
        ]
        return code


def build_hist(s, code, df_var="chart"):
    code.append("{} = pd.value_counts(s).to_frame(name='data')".format(df_var))
    code.append(
        "{df_var}['percent'] = ({df_var}['data'] / {df_var}['data'].sum()) * 100".format(
            df_var=df_var
        )
    )
    df = pd.value_counts(s).to_frame(name="data")
    df["percent"] = (df["data"] / df["data"].sum()) * 100
    return df


class ValueCountAnalysis(object):
    def __init__(self, req):
        self.top = req.get("top")
        self.ordinal_col = req.get("ordinalCol")
        self.ordinal_agg = req.get("ordinalAgg",'sum')
        self.cleaners = req.get("cleaner")

    def build_hist(self, s, code):
        return build_hist(s, code)

    def setup_ordinal_data(self, parent):
        if self.ordinal_agg == "pctsum":
            return pctsum_updates(parent.data, parent.selected_col, self.ordinal_col)

        ordinal_data = getattr(
            parent.data.groupby(parent.selected_col)[[self.ordinal_col]],
            self.ordinal_agg,
        )()
        ordinal_code = [
            "ordinal_data = df.groupby('{col}')[['{ordinal}']].{agg}()".format(
                col=parent.selected_col, ordinal=self.ordinal_col, agg=self.ordinal_agg
            )
        ]
        return ordinal_data, ordinal_code

    def setup_chart_layout(self, parent):
        pp = pprint.PrettyPrinter(indent=4)
        layout_cfg = {
            "barmode": "group",
            "legend": {"orientation": "h"},
            "title": {"text": "{} Value Counts".format(parent.selected_col)},
            "xaxis": {"title": {"text": parent.selected_col}},
            "yaxis": {"title": {"text": "Frequency"}},
        }
        if self.ordinal_col:
            layout_cfg["yaxis2"] = {
                "title": {"text": "{} ({})".format(self.ordinal_col, self.ordinal_agg)},
                "side": "right",
                "overlaying": "y",
            }
        return pp.pformat(go.Layout(**layout_cfg))

    def build(self, parent):
        code = [
            "s = df[~pd.isnull(df['{col}'])]['{col}']".format(col=parent.selected_col)
        ]
        s, cleaner_code = handle_cleaners(
            parent.data[parent.selected_col], self.cleaners
        )
        code += cleaner_code
        hist = self.build_hist(s, code)

        if self.ordinal_col is not None:
            ordinal_data, ordinal_code = self.setup_ordinal_data(parent)
            code += ordinal_code
            hist["ordinal"] = ordinal_data
            hist.index.name = "labels"
            hist = hist.reset_index().sort_values("ordinal")
            code += [
                "chart['ordinal'] = ordinal_data",
                "chart.index.name = 'labels'",
                "chart = chart.reset_index().sort_values('ordinal')",
            ]
        else:
            hist.index.name = "labels"
            hist = hist.reset_index().sort_values(
                ["data", "labels"], ascending=[False, True]
            )
            code += [
                "chart.index.name = 'labels'",
                "chart = chart.reset_index().sort_values(['data', 'labels'], ascending=[False, True])",
            ]
        hist, top, top_code = handle_top(hist, self.top)
        code += top_code
        col_types = grid_columns(hist)
        f = grid_formatter(col_types, nan_display=None)
        return_data = f.format_lists(hist)
        return_data["top"] = top

        layout = self.setup_chart_layout(parent)
        code.append(
            "charts = [go.Bar(x=chart['labels'].values, y=chart['data'].values, name='Frequency')]"
        )
        if self.ordinal_col:
            code.append(
                (
                    "charts.append(go.Scatter(\n"
                    "\tx=chart['labels'].values, y=chart['ordinal'].values, yaxis='y2',\n"
                    "\tname='{} ({})', {}\n"
                    "))"
                ).format(self.ordinal_col, self.ordinal_agg, LINE_CFG)
            )
        code.append(
            "figure = go.Figure(data=charts, layout=go.{layout})".format(layout=layout)
        )
        return return_data, code


class WordValueCountAnalysis(ValueCountAnalysis):
    def build_hist(self, s, code):
        code.append("chart = pd.value_counts(s.str.split(expand=True).stack())")
        code.append("chart = chart.to_frame(name='data').sort_index()")
        code.append("chart['percent'] = (chart['data'] / chart['data'].sum()) * 100")
        df = (
            pd.value_counts(s.str.split(expand=True).stack())
            .to_frame(name="data")
            .sort_index()
        )
        df["percent"] = (df["data"] / df["data"].sum()) * 100
        return df

    def setup_ordinal_data(self, parent):
        expanded_words = parent.data[parent.selected_col].str.split(expand=True).stack()
        expanded_words.name = "label"
        expanded_words = expanded_words.reset_index()[["level_0", "label"]]
        expanded_words.columns = ["index", "label"]
        expanded_words = pd.merge(
            parent.data[[self.ordinal_col]],
            expanded_words.set_index("index"),
            how="inner",
            left_index=True,
            right_index=True,
        )
        expanded_word_code = [
            (
                "ordinal_data = df['{col}'].str.split(expand=True).stack()\n"
                "ordinal_data.name = 'label'\n"
                "ordinal_data = ordinal_data.reset_index()[['level_0', 'label']]\n"
                "ordinal_data.columns = ['index', 'label']\n"
                "ordinal_data = pd.merge(\n"
                "\tdf[['{ordinal}']],\n"
                "\tordinal_data.set_index('index'),\n"
                "\thow='inner',\n"
                "\tleft_index=True,\n"
                "\tright_index=True,\n"
                ")"
            ).format(col=parent.selected_col, ordinal=self.ordinal_col)
        ]
        if self.ordinal_agg == "pctsum":
            ordinal_data, ordinal_code = pctsum_updates(
                expanded_words, "label", self.ordinal_col
            )
            return ordinal_data, expanded_word_code + ordinal_code

        ordinal_code = expanded_word_code + [
            "ordinal_data = ordinal_data.groupby('label')[['{ordinal}']].{agg}()".format(
                ordinal=self.ordinal_col, agg=self.ordinal_agg
            )
        ]
        ordinal_data = getattr(
            expanded_words.groupby("label")[[self.ordinal_col]], self.ordinal_agg
        )()
        return ordinal_data, ordinal_code

    def setup_chart_layout(self, parent):
        pp = pprint.PrettyPrinter(indent=4)
        layout_cfg = {
            "barmode": "group",
            "legend": {"orientation": "h"},
            "title": {"text": "{} Word Value Counts".format(parent.selected_col)},
            "xaxis": {"title": {"text": parent.selected_col}},
            "yaxis": {"title": {"text": "Frequency"}},
        }
        if self.ordinal_col:
            layout_cfg["yaxis2"] = {
                "title": {"text": "{} ({})".format(self.ordinal_col, self.ordinal_agg)},
                "side": "right",
                "overlaying": "y",
            }
        return pp.pformat(go.Layout(**layout_cfg))


class GeolocationAnalysis(object):
    def __init__(self, req):
        self.lat_col = get_str_arg(req, "latCol")
        self.lon_col = get_str_arg(req, "lonCol")

    def build(self, parent):
        geo = parent.data[[self.lat_col, self.lon_col]].dropna()
        geo.columns = ["lat", "lon"]
        col_types = grid_columns(geo)
        f = grid_formatter(col_types, nan_display=None)
        return_data = f.format_lists(geo)
        return return_data, self._build_code()

    def _build_code(self):
        pp = pprint.PrettyPrinter(indent=4)
        layout = pp.pformat(
            go.Layout(
                **{
                    "autosize": True,
                    "geo": {"fitbounds": "locations", "scope": "world"},
                    "legend": {"orientation": "h"},
                    "margin": {"b": 0, "l": 0, "r": 0},
                    "title": {
                        "text": "Map of Latitude({})/ Longitude({})".format(
                            self.lat_col, self.lon_col
                        )
                    },
                }
            )
        )
        return [
            "chart = df[['{}', '{}']].dropna()".format(self.lat_col, self.lon_col),
            "chart.columns = ['lat', 'lon']",
            (
                "chart = go.Scattergeo(\n"
                "\tlon=chart['lon'].values,\n"
                "\tlat=chart['lat'].values,\n"
                "\tmode='markers',\n"
                "\tmarker={'color': 'darkblue'}\n"
                ")"
            ),
            "figure = go.Figure(data=chart, layout=go.{layout})".format(layout=layout),
        ]


class QQAnalysis(object):
    def build(self, parent):
        s = parent.data[parent.selected_col]
        if parent.classifier == "D":
            s = apply(s, json_timestamp)

        qq_x, qq_y = sts.probplot(s, dist="norm", fit=False)
        qq = pd.DataFrame(dict(x=qq_x, y=qq_y))
        f = grid_formatter(grid_columns(qq), nan_display=None)
        return_data = f.format_lists(qq)

        trend_line = px.scatter(x=qq_x, y=qq_y, trendline="ols").data[1]
        trend_line = pd.DataFrame(dict(x=trend_line["x"], y=trend_line["y"]))
        f = grid_formatter(grid_columns(trend_line), nan_display=None)
        trend_line = f.format_lists(trend_line)
        return_data["x2"] = trend_line["x"]
        return_data["y2"] = trend_line["y"]
        return return_data, self._build_code(parent)

    def _build_code(self, parent):
        pp = pprint.PrettyPrinter(indent=4)
        layout = pp.pformat(
            go.Layout(
                **{
                    "legend": {"orientation": "h"},
                    "title": {"text": "{} QQ Plot".format(parent.selected_col)},
                }
            )
        )
        code = [
            "s = df[~pd.isnull(df['{col}'])]['{col}']".format(col=parent.selected_col)
        ]
        if parent.classifier == "D":
            code.append(
                (
                    "\nimport time\n\n"
                    "s = s['{col}'].apply(\n"
                    "\tlambda x: int((time.mktime(x.timetuple()) + (old_div(x.microsecond, 1000000.0))) * 1000\n"
                    ")"
                ).format(col=parent.selected_col)
            )
        code += [
            "\nimport scipy.stats as sts\nimport plotly.express as px\n",
            'qq_x, qq_y = sts.probplot(s, dist="norm", fit=False)',
            "chart = px.scatter(x=qq_x, y=qq_y, trendline='ols', trendline_color_override='red')",
            "figure = go.Figure(data=chart, layout=go.{layout})".format(layout=layout),
        ]
        return code


"""
http://anschonfel.hn.res.num:9205/dtale/column-analysis/1?col=str_val&top=10&type=frequency&filtered=false
"""


class FrequencyAnalysis(object):
    def __init__(self, req):
        self.top = get_int_arg(req, "top")
        self.split_cols = get_str_arg(req, "splits", "")
        if self.split_cols:
            self.split_cols = self.split_cols.split(",")
        else:
            self.split_cols = []
        self.cleaners = get_str_arg(req, "cleaner")

    def build(self, parent):
        code = []
        if parent.classifier == "S":
            code.append(
                "s = df.fillna('Missing')['{col}']".format(col=parent.selected_col)
            )
            s, cleaner_code = handle_cleaners(
                parent.data[parent.selected_col].fillna("Missing"), self.cleaners
            )
            code += cleaner_code
        else:
            code.append(
                "s = df['{col}'].fillna('Missing').astype(str)".format(
                    col=parent.selected_col
                )
            )
            formatter = find_dtype_formatter(parent.dtype)
            s = parent.data[parent.selected_col].apply(
                lambda x: formatter(x, nan_display="Missing")
            )

        df_var = "base_vals" if len(self.split_cols) else "result"

        base_vals = build_hist(s, code, df_var)
        base_vals.index.name = parent.selected_col
        base_vals = base_vals.rename(
            columns={"data": "Frequency", "percent": "Percent"}
        )
        base_vals = base_vals[base_vals["Frequency"] > 0]
        base_vals = base_vals.reset_index().sort_values(
            ["Frequency", parent.selected_col], ascending=[False, True]
        )
        base_vals = base_vals.head(self.top)

        code += [
            "{}.index.name = '{}'".format(df_var, parent.selected_col),
            "{df_var} = {df_var}.{rename}".format(
                df_var=df_var,
                rename="rename(columns={'data': 'Frequency', 'percent': 'Percent'})",
            ),
            "{df_var} = {df_var}[{df_var}['Frequency'] > 0]".format(df_var=df_var),
            "{df_var} = {df_var}.reset_index().sort_values(['Frequency', '{col}'], ascending=[False, True])".format(
                df_var=df_var, col=parent.selected_col
            ),
            "{df_var} = {df_var}.head(self.top)".format(df_var=df_var),
        ]

        if len(self.split_cols):
            top_vals = base_vals[parent.selected_col]
            val_filter = parent.data[parent.selected_col].isin(top_vals)
            val_filter_code = "val_filter = df['{col}'].isin(top_vals)".format(
                col=parent.selected_col
            )
            if (top_vals == "Missing").any():
                val_filter = val_filter | parent.data[parent.selected_col].isnull()
                val_filter_code = "({val_filter} | df['{col}'].isnull())".format(
                    val_filter=val_filter, col=parent.selected_col
                )
            hist = parent.data[val_filter].groupby([s] + self.split_cols).size()
            hist.name = "Frequency"
            hist = hist.reset_index()
            hist = hist[hist["Frequency"] > 0]
            outer_freq = hist.groupby(parent.selected_col)["Frequency"].transform("sum")
            hist["Percent"] = (hist["Frequency"] / outer_freq) * 100

            code += [
                "top_vals = {df_var}['{col}']".format(
                    df_var=df_var, col=parent.selected_col
                ),
                val_filter_code,
                "result = df[val_filter].groupby([s, '{}']).size()".format(
                    "', '".join(self.split_cols)
                ),
                "result.name = 'Frequency'",
                "result = result.reset_index()",
                "result = result[result['Frequency'] > 0]",
                "outer_freq = result.groupby('{}')['Frequency'].transform('sum')".format(
                    parent.selected_col
                ),
                "result['Percent'] = (result['Frequency'] / outer_freq) * 100",
            ]
        else:
            hist = base_vals

        col_types = grid_columns(hist)
        f = grid_formatter(col_types, nan_display=None)
        return_data = f.format_lists(hist)
        return_data = dict(data=return_data)
        return_data["top"] = self.top
        return return_data, code
