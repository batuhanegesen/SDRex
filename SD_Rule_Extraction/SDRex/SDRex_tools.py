from BaseClass import Parameter
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from scipy.stats import qmc
import pysd


def convert_to_pyname(name: str):
    py_name = name.lower().lstrip().rstrip().replace(" ", "_")
    return py_name


def convert_to_name(name: str):
    name = name.capitalize().lstrip().rstrip().replace("_", " ")
    return name


def get_python_info():
    import sys
    print(sys.version)


def create_lhs(dimension: int, sample_size: int):
    engine = LatinHypercube(dimension)
    lhs = engine.random(sample_size)
    return lhs


def get_sample_item(row: int, sample_data: pd.DataFrame):
    rowData = []
    for column in sample_data.columns:
        param = Parameter()
        param.name = column
        param.value = sample_data[column][row]
        rowData.append(param)
    return rowData


def builtin_scaling(low_bounds, high_bounds, sample):
    return qmc.scale(sample, low_bounds, high_bounds)


def run_sim_model(model: pysd.pysd.Model, parameters):
    """
    Runs the model with given sample parameters.
    """
    # run the data with using the sample data.
    keys = []
    values = []
    for parameter in parameters:
        keys.append(parameter.name)
        values.append(parameter.value)
    parameterDictionary = dict(zip(keys, values))
    stocks = model.run(parameterDictionary)
    # print(stocks)
    return stocks
