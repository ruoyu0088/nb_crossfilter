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


class DCChart(object):

    def __init__(self, df, dim_column, group_column=None, **kw):
        self.df = df
        self._dim_column = dim_column
        self._group_column = group_column if group_column is not None else dim_column
        self.settings = kw

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
    def width(self):
        return self.settings.get("width", 300)

    @property
    def height(self):
        return self.settings.get("height", 300)

    @property
    def title(self):
        return self.settings.get("title", self.dim_column)

    @property
    def min(self):
        return self.df[self.dim_column].min()

    @property
    def max(self):
        return self.df[self.dim_column].max()


class BarChart(DCChart):
    type = "barChart"

    @property
    def scale(self):
        return (self.max - self.min) / float(self.bins)

    @property
    def bins(self):
        return self.settings.get("bins", 30)

    @property
    def dimension(self):
        return 'return Math.floor((+d["{name}"] - {min})/{scale}) * {scale} + {min} + 0.5 * {scale}'.format(
                    name=self.json_dim_column, scale=self.scale, min=self.min)


class RowChart(DCChart):
    type = "rowChart"

    @property
    def dimension(self):
        dt = self.dt
        if dt is not None:
            if dt == "year":
                expr = 'return d3.time.year(d["{name}"]).getFullYear()'
            elif dt == "month":
                expr = 'return d["{name}"].getMonth() + 1'
            elif dt == "dayofweek":
                expr = 'var day = d["{name}"].getDay();' \
                       'var name=["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];' \
                        'return day+"."+name[day];'
        elif self.is_number:
            expr = 'return +d["{name}"]'
        else:
            expr = 'return d["{name}"]'
        return expr.format(name=self.json_dim_column)

    @property
    def dt(self):
        return self.settings.get("dt", None)


class CrossFilter(object):

    def __init__(self, df, columns=None, charts=None, **settings):
        if columns is None:
            columns = df.columns.tolist()
        self.df = df[[col.split("@")[0] for col in columns]]

        if charts is None:
            self.charts = []

            for column in columns:
                col = column.split("@")[0]
                if "@" in column:
                    if self.df[col].dtype.name.startswith("datetime"):
                        dt = column.split("@")[1]
                    else:
                        dt = None
                    chart = RowChart(self.df, col, dt=dt, **settings)
                else:
                    chart = BarChart(self.df, col, **settings)
                self.charts.append(chart)
        else:
            self.charts = charts

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

#.x(d3.scale.linear().domain([{{filter.col_min(i)}}, {{filter.col_max(i)}}]))