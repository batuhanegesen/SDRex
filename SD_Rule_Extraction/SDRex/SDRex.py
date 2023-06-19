import pysd
import pandas as pd
import SDRex_tools as SDRexT
from BaseClass import ParameterData
import ModelML as ml
from MIRCO import MIRCO
from sklearn.ensemble import RandomForestRegressor


class SDRuleX:

    def __init__(self):
        """
        Simulates system dynamics model to train random forest metamodel. The trained metamodel is used
        to extract rules by using MIRCO (Minimum Rule Covering) approach.
        """
        self.simulation_duration = 0
        self.model = None
        self.param_data = None
        self.param_list = []
        self.scaled_dataset = None
        self.simulation_output_type = "avg"  # or "max" for the maximum of simulations.
        self.simulation_output = None
        self.simulation_result = None
        self.train_features = None
        self.test_features = None
        self.train_labels = None
        self.test_labels = None
        self.trained_model = None
        self.rf_estimators = 100
        self.test_size = 0.25
        self.predicts = None
        self.fitted_MIRCO = None

    def read_vensim(self, mdl_filepath: any):
        model = pysd.read_vensim(mdl_filepath)
        self.model = model
        return model

    # def read_xmile(xmile_file: any):
    #    model = pysd.read_xmile(xmile_file)
    #    self.model = model
    #    return model
    # not implemented yet.

    def load_param_default(self, filepath: str):
        """
        Loads the specified parameters' values and matches the values
        with respective parameters of specified model.
        Supports xls, xlsx, xlsm, xlsb, odf, ods and odt file extensions read from a local filesystem or URL.

        Example
        -----------
        An example excel parameter file is:
        >>>        Param1  Param2  Param3  ... ParamN
        >>> Value   10      5       1       ... x
        """
        df = pd.read_excel(filepath, index_col=False)

    def load_param_range(self, filepath: str):
        """
        Loads the specified parameters' range to simulate for random values.
        Supports xls, xlsx, xlsm, xlsb, odf, ods and odt file extensions read from a local filesystem or URL.

        Example File
        -----------
        An example excel parameter range file is:
        >>> Param1  Param2  Param3  ... ParamN
        >>> 5       10      50      ... x
        >>> 10      20      100     ... x

        First row is the headers,
        Second row is the minimum of range,
        Third row is the maximum of range.
        """
        range_dataframe = pd.read_excel(filepath, index_col=False)

        param_list = []
        for column in range_dataframe.columns:
            param_range = [range_dataframe[column][0], range_dataframe[column][1]]
            pardata = ParameterData()
            pardata.range = param_range
            pardata.name = column
            param_list.append(pardata)
        self.param_list = param_list
        return param_list

    def create_scaled_dataset(self, sample_size):
        '''
        Creates a dataset with a size of "sample_size" within the specified ranges
        "parameter_data" by using LHS(latin hypercube sampling) method.
        '''
        lhs = SDRexT.create_lhs(self.param_list.__len__(), sample_size)
        df = pd.DataFrame()
        header_list = []
        low_range = []
        high_range = []
        for parameter in self.param_list:
            header_list.append(parameter.name)
            low_range.append(parameter.range[0])
            high_range.append(parameter.range[1])
        data_set = []
        # We map the sample data with the corresponding ranges by using the built in scaling method.
        data_set = SDRexT.builtin_scaling(low_range, high_range, lhs)
        df = df.reindex(columns=header_list)
        for row in data_set:
            df.loc[len(df)] = row
        self.scaled_dataset = df
        return df

    def calculate_result(self, stocks, output_name):
        '''
        The average change of the output_data and the range_data parameters' values are saved.
        This data is later used with random forests and the relation between input and output will be seen.
        '''
        df = pd.DataFrame()
        header_list = []
        data_set = []
        if self.simulation_output_type == "avg":
            output = stocks[output_name].mean()  # get average.
        elif self.simulation_output_type == "max":
            output = stocks[output_name].max()  # get max.
        else:
            output = stocks[output_name].mean()  # get average.
            print("Entered wrong output type, taking the average as the default simulation output type.")
        for parameter in self.param_list:
            header_list.append(parameter.name)
            data_set.append(stocks[parameter.name][0])
        header_list.append("avg_" + output_name)
        data_set.append(output)
        df = df.reindex(columns=header_list)
        df.loc[len(df)] = data_set
        self.simulation_result = df
        # print(df)
        return df

    def simulate(self, simulation_output_type, sample_size: int, output_name):
        """
        Run this to simulate for the specified simulation count.
        """
        self.simulation_output_type = simulation_output_type
        self.create_scaled_dataset(sample_size)
        df = pd.DataFrame()
        df_list = []
        for index in range(self.scaled_dataset.shape[0]):
            parameters = SDRexT.get_sample_item(index, self.scaled_dataset)
            stocks = SDRexT.run_sim_model(self.model, parameters)
            df_list.append(self.calculate_result(stocks, output_name))
        df = pd.concat(df_list, ignore_index=True)
        self.simulation_result = df
        return df

    def train(self, estimators, test_size):
        self.rf_estimators = estimators
        self.test_size = test_size
        self.train_features, self.test_features, self.train_labels, self.test_labels = \
            ml.prepare_model(self.simulation_result,
                             self.simulation_result.columns[self.simulation_result.columns.__len__() - 1],
                             t_size=self.test_size)
        print(self.test_size)
        self.trained_model = ml.train_model(self.train_features, self.train_labels, estimators=self.rf_estimators)

    def predict(self, model="RF"):
        if model == "RF":
            self.predicts = ml.predict_model(self.trained_model, self.test_features, self.test_labels)
        elif model == "MIRCO":
            self.predicts = self.fitted_MIRCO.predict(self.test_features)

    def fit_MIRCO(self):
        MRC = MIRCO.MIRCO(self.trained_model)
        MRC_fit = MRC.fit(self.train_features, self.train_labels)
        self.fitted_MIRCO = MRC_fit

    def rule_extract(self):
        self.fit_MIRCO()
        print('\n\nRules obtained by MIRCO')
        self.fitted_MIRCO.exportRules()
        print('\n## NUMBERS OF RULES ##')
        print('Random Forest: ', self.fitted_MIRCO.initNumOfRules)
        print('MIRCO: ', self.fitted_MIRCO.numOfRules)

    def plot(self):
        self.scaled_dataset.plot(kind='scatter',
                                 x=self.param_list[0].name,
                                 y=self.param_list[1].name,
                                 color='blue')

    def show_importance(self):
        ml.visualize_importance(ml.get_importance(self.trained_model, self.simulation_result))

    def print_simdata(self):
        print(self.simulation_result)