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

def column_setting(func):
    @wraps(func)
    def f(self, col):
        col_name = self.df.columns[col]
        key_name = func.func_name[4:]

        if col_name in self.settings:
            col_settings = self.settings[col_name]
            if key_name in col_settings:
                return col_settings[key_name]
        key = col_name, key_name
        if key in self._col_cache:
            return self._col_cache[key]
        value = func(self, col)
        self._col_cache[key] = value
        return value
    return f


class CrossFilter(object):

    def __init__(self, df, settings=None, min_bin_count=30, width=300, height=300):
        self.df = df
        self.min_bin_count = min_bin_count
        self.settings = {}
        if settings is not None:
            self.settings = settings
        self._col_cache = {}
        self.width = width
        self.height = height
        now = datetime.now()
        self.timestamp = now.strftime("%y%m%d%H%M%S%f")


    def to_json(self):
        df = self.df.copy()
        df.columns = ["c{}".format(i) for i in range(self.df.shape[1])]
        df["_I"] = np.arange(df.shape[0])
        return df.to_json(orient="records")

    def itercolumns(self):
        return enumerate(self.df.columns)

    def dimension(self, col):
        series = self.df.icol(col)
        col_name = self.df.columns[col]
        dtype = series.dtype
        if issubclass(dtype.type, np.number):
            value_count = self.col_value_count(col)
            if value_count < self.min_bin_count:
                return '+d["c{}"]'.format(col)
            else:
                return 'Math.floor((+d["c{name}"] - {min})/{scale}) * {scale} + {min} + 0.5 * {scale}'.format(
                    name=col, scale=self.col_scale(col), min=self.col_min(col))
        else:
            return 'd["{}"]'.format(col_name)

    def group(self, col):
        pass

    @column_setting
    def col_value_count(self, col):
        return self.df.icol(col).unique().shape[0]

    @column_setting
    def col_bin_count(self, col):
        count = self.col_value_count(col)
        return min(count, self.min_bin_count)

    @column_setting
    def col_min(self, col):
        return self.df.icol(col).min()

    @column_setting
    def col_max(self, col):
        return self.df.icol(col).max()

    @column_setting
    def col_scale(self, col):
        scale = (self.col_max(col) - self.col_min(col)) / self.col_bin_count(col)
        return scale #round(scale, int(2 - round(math.log10(scale))))

    @column_setting
    def col_width(self, col):
        return self.width

    @column_setting
    def col_height(self, col):
        return self.height

    @column_setting
    def col_radius(self, col):
        return 0.5 * min(self.col_width(col), self.col_height(col))

    @column_setting
    def col_inner_radius(self, col):
        return self.col_radius(col) * 0.7

    @column_setting
    def col_chart(self, col):
        if self.col_value_count(col) < self.min_bin_count:
            return "rowChart"
        else:
            return "barChart"


settings = {
    "quality":{"chart":"rowChart"},
    "CRIM":{"width":600}
}

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