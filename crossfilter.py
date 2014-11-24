from os import path
from datetime import datetime
import json
import math
from functools import wraps
from glob import glob
import urllib
import jinja2
import pandas as pd
import numpy as np
import matplotlib

CDNJS_URLS = {
    "d3": "http://d3js.org/d3.v3.min",
    "crossfilter": "http://cdnjs.cloudflare.com/ajax/libs/crossfilter/1.3.11/crossfilter.min.js",
    "dc": "http://cdnjs.cloudflare.com/ajax/libs/dc/2.0.0-alpha.2/dc.min.js"
}

NOTEBOOK_URLS = {
    "d3": "/nbextensions/d3.min.js",
    "crossfilter": "/nbextensions/crossfilter.min.js",
    "dc": "/nbextensions/dc.min.js",
    "css": "/nbextensions/dc.css"
}

REDUCE_MEAN = """reduce(function(p, v) {{
    p.sum = p.sum + v["{name}"];
    p.count = p.count + 1;
    return p;
}},

function(p, v) {{
    p.sum = p.sum - v["{name}"];
    p.count = p.count - 1;
    return p;
}},

function() {{
    return {{
        sum: 0,
        count: 0
    }};
}})"""


class DCChart(object):

    def __init__(self, dim_column, group_column=None, **kw):
        self._dim_column = dim_column
        self._group_column = group_column if group_column is not None else dim_column
        self.settings = kw

    def get_setting(self, key, default):
        return self.settings.get(key, self.filter.settings.get(key, default))

    @property
    def df(self):
        return self.filter.df

    @property
    def dtype(self):
        return self.df[self.dim_column].dtype

    @property
    def is_number(self):
        return issubclass(self.dtype.type, np.number)

    @property
    def is_datetime(self):
        return self.dtype.name.startswith("datetime64")

    @property
    def dim_column(self):
        return self._dim_column

    @property
    def group_column(self):
        return self._group_column

    @property
    def json_dim_column(self):
        index = self.df.columns.tolist().index(self.dim_column)
        return "c{}".format(index)

    @property
    def json_group_column(self):
        index = self.df.columns.tolist().index(self.group_column)
        return "c{}".format(index)

    @property
    def width(self):
        return self.get_setting("width", 300)

    @property
    def height(self):
        return self.get_setting("height", 300)

    @property
    def title(self):
        return self.get_setting("title", self.dim_column)

    @property
    def min(self):
        return self.df[self.dim_column].min()

    @property
    def max(self):
        return self.df[self.dim_column].max()

    @property
    def reduce(self):
        method = self.reduce_method
        if method == "count":
            return 'reduceCount()'
        elif method == "sum":
            return 'reduceSum(function(d){{return d["{name}"];}})'.format(name=self.json_group_column)
        elif method == "mean":
            return REDUCE_MEAN.format(name=self.json_group_column)
        else:
            raise ValueError("unknown reduce method")

    @property
    def reduce_method(self):
        return self.get_setting("reduce", "count")


class BarChart(DCChart):
    type = "barChart"

    @property
    def scale(self):
        return (self.max - self.min) / float(self.bins)

    @property
    def bins(self):
        return self.get_setting("bins", 30)

    @property
    def dimension(self):
        return 'return Math.floor((+d["{name}"] - {min})/{scale}) * {scale} + {min} + 0.5 * {scale}'.format(
                    name=self.json_dim_column, scale=self.scale, min=self.min)


class RowChart(DCChart):
    type = "rowChart"

    @property
    def dimension(self):
        transform = self.get_setting("transform", None)
        if transform is not None:
            if transform == "year":
                expr = 'return d3.time.year(d["{name}"]).getFullYear()'
            elif transform == "month":
                expr = 'return d["{name}"].getMonth() + 1'
            elif transform == "dayofweek":
                expr = 'var day = d["{name}"].getDay();' \
                       'var name=["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];' \
                        'return day+"."+name[day];'
        elif self.is_number:
            expr = 'return +d["{name}"]'
        else:
            expr = 'return d["{name}"]'
        return expr.format(name=self.json_dim_column)


class PieChart(RowChart):
    type = "pieChart"

    @property
    def radius(self):
        return self.get_setting("radius", min(self.width, self.height) * 0.5)

    @property
    def inner_radius(self):
        return self.get_setting("inner_radius", self.radius * 0.7)


class ChartCommandParser(object):

    def __init__(self, cmd):
        self.chart_type = "|"
        self.dim_column = ""
        self.group_column = ""
        self.dim_transform = None
        self.group_reduce = "count"

        if cmd[1] == ":":
            self.chart_type = cmd[0]
            cmd = cmd[2:]
        idx = cmd.find("~")
        if idx == -1:
            self.dim_column = cmd
        else:
            self.dim_column = cmd[:idx]
        if ">" in self.dim_column:
            self.dim_column, self.dim_transform = self.dim_column.split(">")
        if idx == -1:
            self.group_column = self.dim_column
            self.group_reduce = "count"
        else:
            cmd = cmd[idx+1:]
            idx = cmd.index("(")
            self.group_reduce = cmd[:idx]
            self.group_column = cmd[idx+1:-1]


def create_chart(chart_cmd):
    chart_map = {
        "|": BarChart,
        "=": RowChart,
        "O": PieChart,
    }

    parser = ChartCommandParser(chart_cmd)
    klass = chart_map[parser.chart_type]
    chart = klass(parser.dim_column, parser.group_column,
                  transform=parser.dim_transform,
                  reduce=parser.group_reduce)
    return chart


class CrossFilter(object):

    def __init__(self, df, charts=None, **settings):

        self.charts = [create_chart(chart) if isinstance(chart, (str, unicode)) else chart for chart in charts]
        columns = set()
        for chart in self.charts:
            columns.add(chart.dim_column)
            columns.add(chart.group_column)
        columns = list(columns)

        self.df = df[columns]

        for chart in self.charts:
            chart.filter = self

        self.settings = settings

        now = datetime.now()
        self.timestamp = now.strftime("%y%m%d%H%M%S%f")

    @property
    def datetime_columns(self):
        return ["c{}".format(i) for i, col in enumerate(self.df.columns)
                if self.df[col].dtype.name.startswith("datetime")]

    def to_json(self):
        df = self.df.copy()
        df.columns = ["c{}".format(i) for i in range(self.df.shape[1])]
        df["_I"] = np.arange(df.shape[0])
        return df.to_json(orient="records", date_format="iso")

    def enumerate_charts(self):
        return enumerate(self.charts)


def nb_update(in_name, out_name, index):
    from IPython import display, get_ipython
    shell = get_ipython()
    in_df = shell.user_ns[in_name]
    index = sorted([int(idx) for idx in index.split()])
    out_df = in_df.iloc[index]
    shell.user_ns[out_name] = out_df


def nb_crossfilter(in_name, out_name, **kw):
    from IPython import display, get_ipython
    shell = get_ipython()
    shell.user_ns["__nb_update__"] = nb_update
    df = shell.user_ns[in_name]
    cf = CrossFilter(df, **kw)
    FOLDER = path.dirname(__file__)
    loader = jinja2.FileSystemLoader(FOLDER)
    env = jinja2.Environment(loader=loader)
    template = env.get_template("crossfilter.html")
    urls = NOTEBOOK_URLS

    require_urls = {key:value[:-3] for key, value in urls.items() if value.endswith(".js")}
    css_urls = [value for value in urls.values() if value.endswith(".css")]

    html = template.render(filter=cf,
                           require_urls=json.dumps(require_urls),
                           css_urls=css_urls,
                           in_name=in_name,
                           out_name=out_name)
    display.display_html(html, raw=True)
    return html

#.x(d3.scale.linear().domain([{{filter.col_min(i)}}, {{filter.col_max(i)}}]))