from __future__ import division

import decimal
import json
import os
import socket
import sys
import time
import traceback
import urllib
from builtins import map, object
from logging import getLogger

# from flask import jsonify as _jsonify

import numpy as np
import pandas as pd
# from pkg_resources import parse_version
# from past.utils import old_div
from six import BytesIO, PY3, StringIO
from statsmodels.imputation import mice
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
logger = getLogger(__name__)


def running_with_pytest():
    """
    Checks to see if it has been initiated from test

    :return: `True` if executed from test, `False` otherwise
    :rtype: bool
    """
    return hasattr(sys, "_called_from_test")


def running_with_debug():
    """
    Checks to see if D-Tale has been initiated from Flask

    :return: `True` if executed from test, `False` otherwise
    :rtype: bool
    """
    return os.environ.get("DEBUG") == "true"


def get_url_unquote():
    """
    Returns URL unquote based on whether Python 2 or 3 is being used.
    """
    return urllib.parse.unquote if PY3 else urllib.unquote


def get_url_quote():
    """
    Returns URL quote based on whether Python 2 or 3 is being used.
    """
    return urllib.parse.quote if PY3 else urllib.quote


def get_host(host=None):
    """
    Returns host input if it exists otherwise the output of :func:`python:socket.gethostname`

    :param host: hostname, can start with 'http://', 'https://' or just the hostname itself
    :type host: str, optional
    :return: str
    """

    def is_valid_host(host):
        try:
            socket.gethostbyname(host.split("://")[-1])
            return True
        except BaseException:
            return False

    if host is None:
        socket_host = socket.gethostname()
        if is_valid_host(socket_host):
            return socket_host
        return "localhost"
    if is_valid_host(host):
        return host
    raise Exception("Hostname ({}) is not recognized".format(host))


def build_url(port, host, ssl=False):
    """
    Returns full url combining host(if not specified will use the output of :func:`python:socket.gethostname`) & port

    :param port: integer string for the port to be used by the :class:`flask:flask.Flask` process
    :type port: str
    :param host: hostname, can start with 'http://', 'https://' or just the hostname itself
    :type host: str, optional
    :param ssl: whether the app is being hosted under HTTPS or not
    :type ssl: boolean, optional
    :return: str
    """
    final_port = ":{}".format(port) if port is not None else ""
    if (host or "").startswith("http"):
        return "{}{}".format(host, final_port)
    return "http{}://{}{}".format("s" if ssl else "", host, final_port)


def get_str_arg(r, name, default=None):
    """
    Retrieve argument from :attr:`flask:flask.request` and convert to string

    :param r: :attr:`flask:flask.request`
    :param name: argument name
    :type: str
    :param default: default value if parameter is non-existent, defaults to None
    :return: string argument value
    """
    val = r.get(name,None)
    if val is None or val == "":
        return default
    else:
        try:
            return str(val)
        except BaseException:
            return default


def get_json_arg(r, name, default=None):
    """
    Retrieve argument from :attr:`flask:flask.request` and parse JSON to python data structure

    :param r: :attr:`flask:flask.request`
    :param name: argument name
    :type: str
    :param default: default value if parameter is non-existent, defaults to None
    :return: parsed JSON
    """
    val = r.args.get(name)
    if val is None or val == "":
        return default
    else:
        return json.loads(val)


def get_int_arg(r, name, default=None):
    """
    Retrieve argument from :attr:`flask:flask.request` and convert to integer

    :param r: :attr:`flask:flask.request`
    :param name: argument name
    :type: str
    :param default: default value if parameter is non-existent, defaults to None
    :return: integer argument value
    """
    val = r.args.get(name)
    if val is None or val == "":
        return default
    else:
        try:
            return int(val)
        except BaseException:
            return default


def get_float_arg(r, name, default=None):
    """
    Retrieve argument from :attr:`flask:flask.request` and convert to float

    :param r: :attr:`flask:flask.request`
    :param name: argument name
    :type: str
    :param default: default value if parameter is non-existent, defaults to None
    :return: float argument value
    """
    val = r.args.get(name)
    if val is None or val == "":
        return default
    else:
        try:
            return float(val)
        except BaseException:
            return default


def get_bool_arg(r, name):
    """
    Retrieve argument from :attr:`flask:flask.request` and convert to boolean

    :param r: :attr:`flask:flask.request`
    :param name: argument name
    :type: str
    :return: `True` if lowercase value equals 'true', `False` otherwise
    """
    return r.get(name, "false").lower() == "true"


def json_string(x, nan_display="", **kwargs):
    """
    convert value to string to be used within JSON output

    If a :class:`python.UnicodeEncodeError` occurs then :func:`python:str.encode` will be called on input

    :param x: value to be converted to string
    :param nan_display: if `x` is :attr:`numpy:numpy.nan` then return this value
    :return: string value
    :rtype: str
    """
    if pd.isnull(x):
        return nan_display
    if x or x in ["", False, 0, pd.Timedelta(0)]:
        try:
            return str(x)
        except UnicodeEncodeError:
            return x.encode("utf-8")
        except BaseException as ex:
            logger.exception(ex)
    return nan_display


def json_int(x, nan_display="", as_string=False, fmt="{:,d}"):
    """
    Convert value to integer to be used within JSON output

    :param x: value to be converted to integer
    :param nan_display: if `x` is :attr:`numpy:numpy.nan` then return this value
    :param as_string: return integer as a formatted string (EX: 1,000,000)
    :return: integer value
    :rtype: int
    """
    try:
        if not np.isnan(x) and not np.isinf(x):
            return fmt.format(int(x)) if as_string else int(x)
        return nan_display
    except BaseException:
        return nan_display


# hack to solve issues with formatting floats with a precision more than 4 decimal points
# https://stackoverflow.com/questions/38847690/convert-float-to-string-without-scientific-notation-and-false-precision
DECIMAL_CTX = decimal.Context()
DECIMAL_CTX.prec = 20


def json_float(x, precision=2, nan_display="nan", inf_display="inf", as_string=False):
    """
    Convert value to float to be used within JSON output

    :param x: value to be converted to integer
    :param precision: precision of float to be returned
    :param nan_display: if `x` is :attr:`numpy:numpy.nan` then return this value
    :param inf_display: if `x` is :attr:`numpy:numpy.inf` then return this value
    :param as_string: return float as a formatted string (EX: 1,234.5643)
    :return: float value
    :rtype: float
    """
    try:
        if np.isinf(x):
            return inf_display
        if not np.isnan(x):
            output = float(round(x, precision))
            if as_string:
                str_output = format(
                    DECIMAL_CTX.create_decimal(repr(x)), ",.{}f".format(str(precision))
                )
                # drop trailing zeroes off & trailing decimal points if necessary
                return str_output.rstrip("0").rstrip(".")
            return output
        return nan_display
    except BaseException:
        return nan_display


def json_date(x, fmt="%Y-%m-%d %H:%M:%S.%f", nan_display="", **kwargs):
    """
    Convert value to date string to be used within JSON output

    :param x: value to be converted to date string
    :param fmt: the data string formatting to be applied
    :param nan_display: if `x` is :attr:`numpy:numpy.nan` then return this value
    :return: date string value
    :rtype: str (YYYY-MM-DD)
    """
    try:
        # calling unique on a pandas datetime column returns numpy datetime64
        output = (pd.Timestamp(x) if isinstance(x, np.datetime64) else x).strftime(fmt)
        empty_microseconds = ".000000"
        if output.endswith(empty_microseconds):
            output = output[: -1 * len(empty_microseconds)]
        empty_time = " 00:00:00"
        if output.endswith(empty_time):
            return output[: -1 * len(empty_time)]
        return output
    except BaseException:
        return nan_display


def json_timestamp(x, nan_display="", **kwargs):
    """
    Convert value to timestamp (milliseconds) to be used within JSON output
    :param x: value to be converted to milliseconds
    :param nan_display: if `x` is :attr:`numpy:numpy.nan` then return this value
    :return: millisecond value
    :rtype: bigint
    """
    try:
        output = pd.Timestamp(x) if isinstance(x, np.datetime64) else x
        output = int(
            (time.mktime(output.timetuple()) + (old_div(output.microsecond, 1000000.0)))
            * 1000
        )
        return str(output) if kwargs.get("as_string", False) else output
    except BaseException:
        return nan_display


class JSONFormatter(object):
    """
    Class for formatting dictionaries and lists of dictionaries into JSON compliant data

    :Example:

        >>> nan_display = 'nan'
        >>> f = JSONFormatter(nan_display)
        >>> f.add_int(1, 'a')
        >>> f.add_float(2, 'b')
        >>> f.add_string(3, 'c')
        >>> jsonify(f.format_dicts([dict(a=1, b=2.0, c='c')]))
    """

    def __init__(self, nan_display="", as_string=False):
        self.fmts = []
        self.nan_display = nan_display
        self.as_string = as_string

    def add_string(self, idx, name=None):
        def f(x, nan_display):
            return json_string(x, nan_display=nan_display)

        self.fmts.append([idx, name, f])

    def add_int(self, idx, name=None, as_string=False):
        def f(x, nan_display):
            return json_int(
                x, nan_display=nan_display, as_string=as_string or self.as_string
            )

        self.fmts.append([idx, name, f])

    def add_float(self, idx, name=None, precision=6, as_string=False):
        def f(x, nan_display):
            return json_float(
                x,
                precision,
                nan_display=nan_display,
                as_string=as_string or self.as_string,
            )

        self.fmts.append([idx, name, f])

    def add_timestamp(self, idx, name=None, as_string=False):
        def f(x, nan_display):
            return json_timestamp(
                x, nan_display=nan_display, as_string=as_string or self.as_string
            )

        self.fmts.append([idx, name, f])

    def add_date(self, idx, name=None, fmt="%Y-%m-%d %H:%M:%S.%f"):
        def f(x, nan_display):
            return json_date(x, fmt=fmt, nan_display=nan_display)

        self.fmts.append([idx, name, f])

    def add_json(self, idx, name=None):
        def f(x, nan_display):
            if x is None or pd.isnull(x):
                return None
            return x

        self.fmts.append([idx, name, f])

    def format_dict(self, lst):
        return {
            name: f(lst[idx], nan_display=self.nan_display)
            for idx, name, f in self.fmts
        }

    def format_dicts(self, lsts):
        return list(map(self.format_dict, lsts))

    def format_lists(self, df):
        return {
            name: [f(v, nan_display=self.nan_display) for v in df[name].values]
            for _idx, name, f in self.fmts
            if name in df.columns
        }

    def format_df(self, df):
        formatters = {col: f for _idx, col, f in self.fmts}
        cols = [col for col in df.columns if col in formatters]
        return pd.concat(
            [
                apply(
                    df[col], lambda v: formatters[col](v, nan_display=self.nan_display)
                )
                for col in cols
            ],
            axis=1,
        )


def classify_type(type_name):
    """

    :param type_name: string label for value from :meth:`pandas:pandas.DataFrame.dtypes`
    :return: shortened string label for dtype
        S = str
        B = bool
        F = float
        I = int
        D = timestamp or datetime
        TD = timedelta
    :rtype: str
    """
    lower_type_name = (type_name or "").lower()
    if lower_type_name.startswith("str"):
        return "S"
    if lower_type_name.startswith("bool"):
        return "B"
    if lower_type_name.startswith("float"):
        return "F"
    if lower_type_name.startswith("int"):
        return "I"
    if any([t for t in ["timestamp", "datetime"] if lower_type_name.startswith(t)]):
        return "D"
    if lower_type_name.startswith("timedelta"):
        return "TD"
    return "S"


def retrieve_grid_params(req, props=None):
    """
    Pull out grid parameters from :attr:`flask:flask.request` arguments and return as a `dict`

    :param req: :attr:`flask:flask.request`
    :param props: argument names
    :type props: list
    :return: dictionary of argument/value pairs
    :rtype: dict
    """
    params = dict()
    params["sort_column"] = get_str_arg(req, "sortColumn")
    params["sort_direction"] = get_str_arg(req, "sortDirection")
    sort = get_str_arg(req, "sort")
    if sort:
        params["sort"] = json.loads(sort)
    return params


def sort_df_for_grid(df, params):
    """
    Sort dataframe based on 'sort' property in parameter dictionary. Sort
    configuration is of the following shape:
    {
        sort: [
            [col1, ASC],
            [col2, DESC],
            ...
        ]
    }

    :param df: dataframe
    :type df: :class:`pandas:pandas.DataFrame`
    :param params: arguments from :attr:`flask:flask.request`
    :type params: dict
    :return: sorted dataframe
    :rtype: :class:`pandas:pandas.DataFrame`
    """
    if "sort" in params:
        cols, dirs = [], []
        for col, dir in params["sort"]:
            cols.append(col)
            dirs.append(dir == "ASC")
        return df.sort_values(cols, ascending=dirs)
    return df.sort_index()


def find_dtype(s):
    """
    Helper function to determine the dtype of a :class:`pandas:pandas.Series`
    """
    if s.dtype.name == "object":
        return pd.api.types.infer_dtype(s, skipna=True)
    else:
        return s.dtype.name


def coord_type(s):
    if classify_type(find_dtype(s)) not in ["F", "I"]:
        return None
    is_pandas_1_3 = parse_version(pd.__version__) >= parse_version("1.3.0")
    inclusive = "both" if is_pandas_1_3 else True
    if "lat" in s.name.lower():
        return (
            None if (~s.dropna().between(-90, 90, inclusive=inclusive)).sum() else "lat"
        )
    if "lon" in s.name.lower():
        return (
            None
            if (~s.dropna().between(-180, 180, inclusive=inclusive)).sum()
            else "lon"
        )
    return None

def get_dtypes(df):
    """
    Build dictionary of column/dtype name pairs from :class:`pandas:pandas.DataFrame`
    """

    def _load():
        for col in df.columns:
            yield col, find_dtype(df[col])

    return dict(list(_load()))

def grid_columns(df):
    """
    Build list of {name, dtype} dictionaries for columns in :class:`pandas:pandas.DataFrame`
    """
    data_type_info = get_dtypes(df)
    return [dict(name=c, dtype=data_type_info[c]) for c in df.columns]


DF_MAPPINGS = {
    "I": lambda f, i, c: f.add_int(i, c),
    "D": lambda f, i, c: f.add_date(i, c),
    "F": lambda f, i, c: f.add_float(i, c),
    "S": lambda f, i, c: f.add_string(i, c),
}


def find_dtype_formatter(dtype, overrides=None):
    type_classification = classify_type(dtype)
    if type_classification in (overrides or {}):
        return overrides[type_classification]
    if type_classification == "I":
        return json_int
    if type_classification == "D":
        return json_date
    if type_classification == "F":
        return json_float
    return json_string


def grid_formatter(col_types, nan_display="", overrides=None, as_string=False):
    """
    Build :class:`dtale.utils.JSONFormatter` from :class:`pandas:pandas.DataFrame`
    """
    f = JSONFormatter(nan_display, as_string=as_string)
    mappings = dict_merge(DF_MAPPINGS, overrides or {})
    for i, ct in enumerate(col_types, 1):
        c, dtype = map(ct.get, ["name", "dtype"])
        type_classification = classify_type(dtype)
        mappings.get(type_classification, DF_MAPPINGS["S"])(f, i, c)
    return f


def build_formatters(df, nan_display=None):
    """
    Helper around :meth:`dtale.utils.grid_formatters` that will build a formatter for the data being fed into a chart as
    well as a formatter for the min/max values for each column used in the chart data.

    :param df: dataframe which contains column names and data types for formatters
    :type df: :class:`pandas:pandas.DataFrame`
    :return: json formatters for chart data and min/max values for each column used in the chart
    :rtype: (:class:`dtale.utils.JSONFormatter`, :class:`dtale.utils.JSONFormatter`)
    """
    cols = grid_columns(df)
    data_f = grid_formatter(cols, nan_display=nan_display)
    overrides = {"F": lambda f, i, c: f.add_float(i, c, precision=2)}
    range_f = grid_formatter(cols, overrides=overrides, nan_display=nan_display)
    return data_f, range_f


def format_grid(df, overrides=None):
    """
    Translate :class:`pandas:pandas.DataFrame` to well-formed JSON.  Structure is as follows:
    {
        results: [
            {col1: val1_row1,...,colN: valN_row1},
            ...,
            {col1: val1_rowN,...,colN: valN_rowN},
        ],
        columns: [
            {name: col1, dtype: int},
            ...,
            {name: colN, dtype: float},
        ]
    }

    :param df: dataframe
    :type df: :class:`pandas:pandas.DataFrame`
    :return: JSON
    """
    col_types = grid_columns(df)
    f = grid_formatter(col_types, overrides=overrides)
    return {"results": f.format_dicts(df.itertuples()), "columns": col_types}


def handle_error(error_info):
    """
    Boilerplate exception messaging
    """
    logger.exception(
        "Exception occurred while processing request: {}".format(
            error_info.get("error")
        )
    )


def jsonify(return_data={}, **kwargs):
    """
    Overriding Flask's jsonify method to account for extra error handling

    :param return_data: dictionary of data to be passed to :meth:`flask:flask.jsonify`
    :param kwargs: Optional keyword arguments merged into return_data
    :return: output of :meth:`flask:flask.jsonify`
    """
    if isinstance(return_data, dict) and return_data.get("error"):
        handle_error(return_data)
        return json.dumps(
            dict_merge(dict(success=False), dict_merge(kwargs, return_data))
        )
    if len(kwargs):
        return json.dumps(dict_merge(kwargs, return_data))
    return json.dumps(return_data)


class ChartBuildingError(Exception):
    """
    Exception for signalling there was an issue constructing the data for your chart.
    """

    def __init__(self, error, details=None):
        super(ChartBuildingError, self).__init__("Chart Error")
        self.error = error
        self.details = details


def jsonify_error(e):
    tb = traceback.format_exc()
    if isinstance(e, ChartBuildingError):
        if e.details:
            tb = e.details
        e = e.error
    return jsonify(dict(error=str(e), traceback=str(tb)))


def find_selected_column(data, col):
    """
    In case we come across a series which after reset_index()
    it's columns are [date, security_id, values]
    in which case we want the last column

    :param data: dataframe
    :type data: :class:`pandas:pandas.DataFrame`
    :param col: column name
    :type col: str
    :return: column name if it exists within the dataframe's columns, the last column within the dataframe otherwise
    :rtype: str
    """

    return col if col in data.columns else data.columns[-1]


def make_list(vals):
    """
    Convert a value that is optionally list or scalar
    into a list
    """
    if vals is None:
        return []
    elif isinstance(vals, (list, tuple)):
        return vals
    return [vals]


def dict_merge(d1, d2, *args):
    """
    Merges two dictionaries.  Items of the second dictionary will
    replace items of the first dictionary if there are any overlaps.
    Either dictionary can be None.  An empty dictionary {} will be
    returned if both dictionaries are None.

    :param d1: First dictionary can be None
    :type d1: dict
    :param d2: Second dictionary can be None
    :type d1: dict
    :return: new dictionary with the contents of d2 overlaying the contents of d1
    :rtype: dict
    """

    def _dict_merge(d11, d12):
        if not d11:
            return d12 or {}
        elif not d12:
            return d11 or {}
        return dict(list(d11.items()) + list(d12.items()))

    ret = _dict_merge(d1, d2)
    for d in args:
        ret = _dict_merge(ret, d)
    return ret


def flatten_lists(lists):
    """
    Take an iterable containing iterables and flatten them into one list.
        - [[1], [2], [3, 4]] => [1, 2, 3, 4]
    """
    return [item for sublist in lists for item in sublist]


def divide_chunks(lst, n):
    """
    Break list input 'l' up into smaller lists of size 'n'
    """
    # looping till length l
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class DuplicateDataError(Exception):
    """
    Exception for signalling that similar data is trying to be loaded to D-Tale again.  Is this correct?
    """

    def __init__(self, data_id):
        super(DuplicateDataError, self).__init__("Duplicate Data")
        self.data_id = data_id


def triple_quote(val):
    return '"""{}"""'.format(val)


def export_to_csv_buffer(data, tsv=False):
    kwargs = dict(encoding="utf-8", index=False)
    if tsv:
        kwargs["sep"] = "\t"
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, **kwargs)
    csv_buffer.seek(0)
    return csv_buffer


def export_to_parquet_buffer(data):
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise ImportError(
            "In order to use the parquet exporter you must install pyarrow!"
        )
    kwargs = dict(compression="gzip", index=False)
    parquet_buffer = BytesIO()
    data.to_parquet(parquet_buffer, **kwargs)
    parquet_buffer.seek(0)
    return parquet_buffer


def is_app_root_defined(app_root):
    return app_root is not None and app_root != "/"


def fix_url_path(path):
    while "//" in path:
        path = path.replace("//", "/")
    return path


def apply(df, func, *args, **kwargs):
    try:
        import swifter  # noqa: F401

        return df.swifter.progress_bar(False).apply(func, *args, **kwargs)
    except ImportError:
        return df.apply(func, *args, **kwargs)


def optimize_df(df):
    for col in df.select_dtypes(include=["object"]):
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype("category")
    return df


def read_file(file_path, encoding="utf-8"):
    open_kwargs = {}
    if PY3 and encoding:
        open_kwargs["encoding"] = encoding
    with open(file_path, "r", **open_kwargs) as file:
        output = file.read()
        if not PY3 and encoding:
            return output.decode(encoding)
        return output


def unique_count(s):
    return int(len(s.dropna().unique()))


def format_data(data, inplace=False, drop_index=False):
    """
    Helper function to build globally managed state pertaining to a D-Tale instances data.  Some updates being made:
     - convert all column names to strings
     - drop any indexes back into the dataframe so what we are left is a natural index [0,1,2,...,n]
     - convert inputs that are indexes into dataframes
     - replace any periods in column names with underscores

    :param data: dataframe to build data type information for
    :type data: :class:`pandas:pandas.DataFrame`
    :param allow_cell_edits: If false, this will not allow users to edit cells directly in their D-Tale grid
    :type allow_cell_edits: bool, optional
    :param inplace: If true, this will call `reset_index(inplace=True)` on the dataframe used as a way to save memory.
                    Otherwise this will create a brand new dataframe, thus doubling memory but leaving the dataframe
                    input unchanged.
    :type inplace: bool, optional
    :param drop_index: If true, this will drop any pre-existing index on the dataframe input.
    :type drop_index: bool, optional
    :return: formatted :class:`pandas:pandas.DataFrame` and a list of strings constituting what columns were originally
             in the index
    :raises: Exception if the dataframe contains two columns of the same name
    """
    if isinstance(data, (pd.DatetimeIndex, pd.MultiIndex)):
        data = data.to_frame(index=False)

    if isinstance(data, (np.ndarray, list, dict)):
        try:
            data = pd.DataFrame(data)
        except BaseException:
            data = pd.Series(data).to_frame()

    index = [
        str(i) for i in make_list(data.index.name or data.index.names) if i is not None
    ]
    drop = True
    if not data.index.equals(pd.RangeIndex(0, len(data))):
        drop = False
        if not len(index):
            index = ["index"]

    if inplace:
        data.reset_index(inplace=True, drop=drop_index)
    else:
        data = data.reset_index(drop=drop_index)

    if drop_index:
        index = []

    if drop:
        if inplace:
            data.drop("index", axis=1, errors="ignore", inplace=True)
        else:
            data = data.drop("index", axis=1, errors="ignore")

    def _format_colname(colname):
        if isinstance(colname, tuple):
            formatted_vals = [
                find_dtype_formatter(type(v).__name__)(v, as_string=True)
                for v in colname
            ]
            return "_".join([v for v in formatted_vals if v])
        return str(colname).strip()

    data.columns = [_format_colname(c) for c in data.columns]
    if len(data.columns) > len(set(data.columns)):
        distinct_cols = set()
        dupes = set()
        for c in data.columns:
            if c in distinct_cols:
                dupes.add(c)
            distinct_cols.add(c)
        raise Exception(
            "data contains duplicated column names: {}".format(", ".join(sorted(dupes)))
        )

    for col in data.columns:
        dtype = find_dtype(data[col])
        all_null = data[col].isnull().all()
        if dtype.startswith("mixed") and not all_null:
            try:
                unique_count(data[col])
            except TypeError:
                # convert any columns with complex data structures (list, dict, etc...) to strings
                data.loc[:, col] = data[col].astype("str")
        elif dtype.startswith("period") and not all_null:
            # convert any pandas period_range columns to timestamps
            data.loc[:, col] = data[col].apply(lambda x: x.to_timestamp())
        elif dtype.startswith("datetime") and not all_null:
            # remove timezone information for filtering purposes
            data.loc[:, col] = data[col].dt.tz_localize(None)

    return data, index


def option(v):
    return dict(value=v, label="{}".format(v))

def build_string_metrics(s, col):
    char_len = s.len()

    def calc_len(x):
        try:
            return len(x)
        except BaseException:
            return 0

    word_len = apply(s.replace(r"[\s]+", " ").str.split(" "), calc_len)

    def txt_count(r):
        return s.count(r).astype(bool).sum()

    string_metrics = dict(
        char_min=int(char_len.min()),
        char_max=int(char_len.max()),
        char_mean=json_float(char_len.mean()),
        char_std=json_float(char_len.std()),
        with_space=int(txt_count(r"\s")),
        with_accent=int(txt_count(r"[À-ÖÙ-öù-ÿĀ-žḀ-ỿ]")),
        with_num=int(txt_count(r"[\d]")),
        with_upper=int(txt_count(r"[A-Z]")),
        with_lower=int(txt_count(r"[a-z]")),
        with_punc=int(
            txt_count(
                r'(\!|"|\#|\$|%|&|\'|\(|\)|\*|\+|,|\-|\.|/|\:|\;|\<|\=|\>|\?|@|\[|\\|\]|\^|_|\`|\{|\||\}|\~)'
            )
        ),
        space_at_the_first=int(txt_count(r"^ ")),
        space_at_the_end=int(txt_count(r" $")),
        multi_space_after_each_other=int(txt_count(r"\s{2,}")),
        # with_hidden=int(txt_count(r"[^{}]+".format(printable))),
        word_min=int(word_len.min()),
        word_max=int(word_len.max()),
        word_mean=json_float(word_len.mean()),
        word_std=json_float(word_len.std()),
    )

    punc_reg = (
        """\tr'(\\!|"|\\#|\\$|%|&|\\'|\\(|\\)|\\*|\\+|,|\\-|\\.|/|\\:|\\;|\\<|\\=|"""
        """\\>|\\?|@|\\[|\\\\|\\]|\\^|_|\\`|\\{|\\||\\}|\\~)'"""
    )
    code = [
        "s = data['{}']".format(col),
        "s = s[~s.isnull()].str",
        "char_len = s.len()\n",
        "def calc_len(x):",
        "\ttry:",
        "\t\treturn len(x)",
        "\texcept:",
        "\t\treturn 0\n",
        "word_len = s.replace(r'[\\s]+', ' ').str.split(' ').apply(calc_len)\n",
        "def txt_count(r):",
        "\treturn s.count(r).astype(bool).sum()\n",
        "char_min=char_len.min()",
        "char_max = char_len.max()",
        "char_mean = char_len.mean()",
        "char_std = char_len.std()",
        "with_space = txt_count(r'\\s')",
        "with_accent = txt_count(r'[À-ÖÙ-öù-ÿĀ-žḀ-ỿ]')",
        "with_num = txt_count(r'[\\d]')",
        "with_upper = txt_count(r'[A-Z]')",
        "with_lower = txt_count(r'[a-z]')",
        "with_punc = txt_count(",
        "\t{}".format(punc_reg),
        ")",
        "space_at_the_first = txt_count(r'^ ')",
        "space_at_the_end = txt_count(r' $')",
        "multi_space_after_each_other = txt_count(r'\\s{2,}')",
        "printable = r'\\w \\!\"#\\$%&'\\(\\)\\*\\+,\\-\\./:;<»«؛،ـ\\=>\\?@\\[\\\\\\]\\^_\\`\\{\\|\\}~'",
        "with_hidden = txt_count(r'[^{}]+'.format(printable))",
        "word_min = word_len.min()",
        "word_max = word_len.max()",
        "word_mean = word_len.mean()",
        "word_std = word_len.std()",
    ]

    return string_metrics, code

def build_sequential_diffs(s, col, sort=None):
    if sort is not None:
        s = s.sort_values(ascending=sort == "ASC")
    diff = s.diff()
    diff = diff[diff == diff]  # remove nan or nat values
    min_diff = diff.min()
    max_diff = diff.max()
    avg_diff = diff.mean()
    diff_vals = diff.value_counts().sort_values(ascending=False)
    diff_vals.index.name = "value"
    diff_vals.name = "count"
    diff_vals = diff_vals.reset_index()

    diff_vals_f = grid_formatter(grid_columns(diff_vals), as_string=True)
    diff_fmt = next((f[2] for f in diff_vals_f.fmts if f[1] == "value"), None)
    diff_ct = len(diff_vals)

    code = (
        "sequential_diffs = data['{}'].diff()\n"
        "diff = diff[diff == diff]\n"
        "min_diff = sequential_diffs.min()\n"
        "max_diff = sequential_diffs.max()\n"
        "avg_diff = sequential_diffs.mean()\n"
        "diff_vals = sequential_diffs.value_counts().sort_values(ascending=False)"
    ).format(col)

    metrics = {
        "diffs": {
            "data": diff_vals_f.format_dicts(diff_vals.head(100).itertuples()),
            "top": diff_ct > 100,
            "total": diff_ct,
        },
        "min": diff_fmt(min_diff, "N/A"),
        "max": diff_fmt(max_diff, "N/A"),
        "avg": diff_fmt(avg_diff, "N/A"),
    }
    return metrics, code

def get_column_stats(df):
    """
    Get detailed statistics for each column in a pandas DataFrame.

    :param df: pandas DataFrame
    :return: List of dictionaries containing statistics for each column
    """
    column_stats = []

    for index, column in enumerate(df.columns):
        col_data = df[column]
        stats = {"name": column, "index": index, "hasMissing": col_data.isnull().sum(), "unique_ct": col_data.nunique(), "visible": True}
        stats["is_categorical"] = categorical(df, column)
        stats["missingness_type"] = analyze_missing_data_for_column(df,column)
        if np.issubdtype(col_data.dtype, np.number):
            # Handling numeric columns (both float and int)
            stats["dtype"] = str(col_data.dtype)
            stats["hasOutliers"] = 0  # Placeholder, calculated below
            stats["kurt"] = col_data.kurtosis()
            stats["lowVariance"] = col_data.var() < 1e-5
            stats["max"] = col_data.max()
            stats["min"] = col_data.min()
            stats["skew"] = col_data.skew()
            # Outlier detection using IQR (Interquartile Range)
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            stats["outlierRange"] = {"lower": lower_bound, "upper": upper_bound}
            stats["hasOutliers"] = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

        elif np.issubdtype(col_data.dtype, np.datetime64):
            # Handling datetime columns
            stats["dtype"] = "datetime64[ns]"
            stats["hasOutliers"] = 0
            stats["kurt"] = col_data.dropna().apply(pd.Timestamp.timestamp).kurtosis()  # Convert datetime to timestamp for kurtosis
            stats["skew"] = col_data.dropna().apply(pd.Timestamp.timestamp).skew()  # Skewness based on timestamp

        elif pd.api.types.is_string_dtype(col_data):
            # Handling string columns
            stats["dtype"] = "string"
            stats["hasOutliers"] = 0

        elif pd.api.types.is_bool_dtype(col_data):
            # Handling boolean columns
            stats["dtype"] = "bool"
            stats["hasOutliers"] = 0

        else:
            # Handle other types if needed
            stats["dtype"] = str(col_data.dtype)

        column_stats.append(stats)

    return column_stats


import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return super().encode(bool(obj))
        return super(NpEncoder, self).default(obj)

def dtype_formatter(data, dtypes, data_ranges, prev_dtypes=None):
    """
    Helper function to build formatter for the descriptive information about each column in the dataframe you
    are viewing in D-Tale.  This data is later returned to the browser to help with controlling inputs to functions
    which are heavily tied to specific data types.

    :param data: dataframe
    :type data: :class:`pandas:pandas.DataFrame`
    :param dtypes: column data type
    :type dtypes: dict
    :param data_ranges: dictionary containing minimum and maximum value for column (if applicable)
    :type data_ranges: dict, optional
    :param prev_dtypes: previous column information for syncing updates to pre-existing columns
    :type prev_dtypes: dict, optional
    :return: formatter function which takes column indexes and names
    :rtype: func
    """

    def _formatter(col_index, col):
        visible = True
        dtype = dtypes.get(col)
        if prev_dtypes and col in prev_dtypes:
            visible = prev_dtypes[col].get("visible", True)
        s = data[col]
        dtype_data = dict(
            name=col,
            dtype=dtype,
            index=col_index,
            visible=visible,
            hasOutliers=0,
            hasMissing=1,
        )
        if global_state.is_arcticdb:
            return dtype_data

        dtype_data["unique_ct"] = unique_count(s)
        dtype_data["hasMissing"] = int(s.isnull().sum())
        classification = classify_type(dtype)
        if (
            classification in ["F", "I"] and not s.isnull().all() and col in data_ranges
        ):  # floats/ints
            col_ranges = data_ranges[col]
            if not any((np.isnan(v) or np.isinf(v) for v in col_ranges.values())):
                dtype_data = dict_merge(col_ranges, dtype_data)

            # load outlier information
            o_s, o_e = calc_outlier_range(s)
            if not any((np.isnan(v) or np.isinf(v) for v in [o_s, o_e])):
                dtype_data["hasOutliers"] += int(((s < o_s) | (s > o_e)).sum())
                dtype_data["outlierRange"] = dict(lower=o_s, upper=o_e)
            skew_val = pandas_util.run_function(s, "skew")
            if skew_val is not None:
                dtype_data["skew"] = json_float(skew_val)
            kurt_val = pandas_util.run_function(s, "kurt")
            if kurt_val is not None:
                dtype_data["kurt"] = json_float(kurt_val)

        if classification in ["F", "I"] and not s.isnull().all():
            # build variance flag
            unique_ct = dtype_data["unique_ct"]
            check1 = (unique_ct / len(data[col])) < 0.1
            check2 = False
            if check1 and unique_ct >= 2:
                val_counts = s.value_counts()
                check2 = (val_counts.values[0] / val_counts.values[1]) > 20
            dtype_data["lowVariance"] = bool(check1 and check2)
            dtype_data["coord"] = coord_type(s)

        if classification in ["D"] and not s.isnull().all():
            timestamps = apply(s, lambda x: json_timestamp(x, np.nan))
            skew_val = pandas_util.run_function(timestamps, "skew")
            if skew_val is not None:
                dtype_data["skew"] = json_float(skew_val)
            kurt_val = pandas_util.run_function(timestamps, "kurt")
            if kurt_val is not None:
                dtype_data["kurt"] = json_float(kurt_val)

        if classification == "S" and not dtype_data["hasMissing"]:
            if (
                dtype.startswith("category")
                and classify_type(s.dtype.categories.dtype.name) == "S"
            ):
                dtype_data["hasMissing"] += int(
                    (apply(s, lambda x: str(x).strip()) == "").sum()
                )
            else:
                dtype_data["hasMissing"] += int(
                    (s.astype("str").str.strip() == "").sum()
                )

        return dtype_data

    return _formatter


def categorical(df, feature):
    """
    Function to check whether a given feature in the DataFrame is categorical or non-categorical.
    This function skips rows with missing values in the given feature when checking the feature type.

    Parameters:
    df : pandas.DataFrame
        The DataFrame containing the feature.
    feature : str
        The name of the feature (column) to check.

    Returns:
    str : "Categorical" or "Non-Categorical"
    """

    # Identify the rows with missing values for the given feature (e.g., None, NaN, empty string, 'Unknown')
    missing_values = [None, 'None', 'unknown', 'Unknown', 'N/A', 'na', 'NA', 'NaN', '', ' ']
    mask = df[feature].isin(missing_values) | df[feature].isna()

    # Filter out rows with missing values in the feature
    clean_data = df[~mask][feature]

    # Check the data type of the feature after excluding missing values
    dtype = clean_data.dtype

    # Check for categorical feature
    if dtype == 'object' or dtype.name == 'category':
        return True

    # For numeric columns, we check if the values are discrete or continuous
    elif pd.api.types.is_numeric_dtype(clean_data):
        # If the number of unique values is small, it's likely categorical
        if clean_data.nunique() < 10:  # Threshold for discrete data
            return True
        else:
            return False

    # If it's neither object nor numeric, we categorize it as Categorical
    return True


def check_mcar(data):
    """Perform Little's MCAR test."""
    from statsmodels.imputation import mice
    try:
        mcar_test = mice.MICEData(data)
        return mcar_test.mice
    except Exception as e:
        print(f"Error performing MCAR test: {e}")
        return None


def check_mar(data, column):
    """Check for MAR by examining correlations of missingness for the given column."""
    # Create a missingness indicator (1 if missing, 0 if not) for the column
    missing_indicator = data[column].isnull().astype(int)

    # Analyze correlation between missingness and observed data for the given column
    correlations = {}
    for col in data.columns:
        if col != column and data[col].dtype != 'object':  # Consider only numeric columns for correlation
            corr_value = missing_indicator.corr(data[col])
            correlations[col] = corr_value

    return correlations


def check_mnar(data, column):
    """Placeholder for MNAR detection for a specific column."""
    print(data[column].isnull().astype(int))
    if 1 in data[column].isnull().astype(int):
        return "None"
    else:
        return "MNAR"


def analyze_missing_data_for_column(data, column):
    """Analyze missing data for a specific column and determine if it's MCAR, MAR, or MNAR."""
    # First, perform MCAR test (Little's MCAR Test)
    mcar_result = check_mcar(data[[column]])  # Only use the column for MCAR test
    if mcar_result is not None:
        return f"MCAR"

    # Check for MAR by examining correlations of missingness for the column
    mar_correlations = check_mar(data, column)

    # If all correlations are low (less than 0.1), it's likely MAR
    if all(abs(corr) < 0.1 for corr in mar_correlations.values()):
        return f"MAR"

    # For categorical features, check the missingness correlation with other features
    if data[column].dtype == 'object' or data[column].dtype.name == 'category':
        cat_correlations = check_missingness_for_categorical(data, column)
        if any(p < 0.05 for p in cat_correlations.values()):  # Significance threshold
            return f"MAR"

    # If none of the above conditions hold, assume MNAR
    return check_mnar(data, column)


def check_missingness_for_categorical(data, column):
    """For categorical data, check missingness correlation with other variables."""
    # Create a mask for missing data
    missing_mask = data[column].isnull()

    # Check the relationship with other columns using chi-squared tests
    correlations = {}
    for col in data.columns:
        if col != column:
            # Only use the columns with non-null values for correlation calculation
            contingency_table = pd.crosstab(missing_mask, data[col])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            correlations[col] = p
    return correlations


def get_missingness_for_column(data, column):
    """Get the missingness pattern for a specific column."""
    # Get the number of missing values in the specific column
    missing_by_column = data[column].isnull().sum()
    missing_percentage_by_column = (missing_by_column / len(data)) * 100

    # Display the missing data pattern
    print(f"Missing Data for Column: {column}")
    print(f"Number of Missing Values: {missing_by_column}")
    print(f"Percentage of Missing Values: {missing_percentage_by_column:.2f}%")

    # Visualize the missing data for this column
    visualize_missing_data_for_column(data, column)


def visualize_missing_data_for_column(data, column):
    """Visualize the missing data pattern for a specific column."""
    missing_mask = data[column].isnull()

    # Create a heatmap for the missing data in that column
    plt.figure(figsize=(8, 4))
    sns.heatmap(missing_mask.to_frame(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title(f"Missing Data Pattern in Column: {column}")
    plt.show()

def manage_missing_values(data, strategy="mean", column=None, time_series=False, add_missing_feature=False):
    """
    Function to manage missing values in a pandas DataFrame using various strategies.

    Parameters:
        data (pd.DataFrame): The input dataframe with missing values.
        strategy (str): The strategy to use for imputation (e.g., "mean", "median", "mode", etc.).
        column (str): The specific column to impute (for univariate imputation).
        time_series (bool): Set True if dealing with time-series data (for forward fill/backward fill).
        add_missing_feature (bool): Set True if you want to create a missingness indicator feature.

    Returns:
        pd.DataFrame: The dataframe with missing values imputed.
    """
    if add_missing_feature:
        # Add a feature indicating if the data was missing
        data[f"{column}_missing"] = data[column].isnull().astype(int) if column else data.isnull().astype(int)
        # Handle missing values for a single column
    if strategy == "Replace with Constant":
        constant_value = 0 # Replace with a constant of choice
        data[column].fillna(constant_value, inplace=True)
    elif strategy == "Mean Imputation":
        non_missing_values = data[column].dropna()
        # Impute missing values with the mean of non-missing values
        data[column].fillna(non_missing_values.mean(), inplace=True)
    elif strategy == "Median Imputation":
        data[column].fillna(data[column].median(), inplace=True)
    elif strategy == "Mode Imputation":
        data[column].fillna(data[column].mode()[0], inplace=True)
    elif strategy == "Forward Fill":
        data[column].fillna(method='ffill', inplace=True)
    elif strategy == "Backward Fill":
        data[column].fillna(method='bfill', inplace=True)
    elif strategy == "Interpolation":
        data[column] = data[column].interpolate(method='linear')
    elif strategy == "Simple Imputer":
        imputer = SimpleImputer(strategy='mean')
        data[column] = imputer.fit_transform(data[[column]])
    elif strategy == "KNN Imputer":
        knn_imputer = KNNImputer(n_neighbors=5)
        data[column] = knn_imputer.fit_transform(data[[column]])
    elif strategy == "Most Frequent Imputation":
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        pass
    return data

def manage_missing_values_2(data, strategy="mean", column=None, time_series=False, add_missing_feature=False,**kwargs):
    """
    Function to manage missing values in a pandas DataFrame using various strategies.

    Parameters:
        data (pd.DataFrame): The input dataframe with missing values.
        strategy (str): The strategy to use for imputation (e.g., "mean", "median", "mode", etc.).
        column (str): The specific column to impute (for univariate imputation).
        time_series (bool): Set True if dealing with time-series data (for forward fill/backward fill).
        add_missing_feature (bool): Set True if you want to create a missingness indicator feature.

    Returns:
        pd.DataFrame: The dataframe with missing values imputed.
    """
    if add_missing_feature:
        # Add a feature indicating if the data was missing
        data[f"missing_{column}"] = data[column].isnull().astype(int) if column else data.isnull().astype(int)
        # Handle missing values for a single column
    if strategy == "constant":
        constant_value = kwargs.get("value",0) # Replace with a constant of choice
        data[column].fillna(constant_value, inplace=True)
    elif strategy == "mean":
        non_missing_values = data[column].dropna()
        # Impute missing values with the mean of non-missing values
        data[column].fillna(non_missing_values.mean(), inplace=True)
    elif strategy == "median":
        data[column].fillna(data[column].median(), inplace=True)
    elif strategy == "mode":
        data[column].fillna(data[column].mode()[0], inplace=True)
    elif strategy == "forward_fill":
        data[column].fillna(method='ffill', inplace=True)
    elif strategy == "backward_fill":
        data[column].fillna(method='bfill', inplace=True)
    elif strategy == "Interpolation":
        data[column] = data[column].interpolate(method='linear')
    elif strategy == "Simple Imputer":
        imputer = SimpleImputer(strategy='mean')
        data[column] = imputer.fit_transform(data[[column]])
    elif strategy == "knn":
        knn_imputer = KNNImputer(**kwargs)
        data[column] = knn_imputer.fit_transform(data[[column]])
    elif strategy == "Most Frequent Imputation":
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        pass
    return data