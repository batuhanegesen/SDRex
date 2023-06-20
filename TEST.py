import SD_Rule_Extraction.SDRexLib.SDRex as sdRex
import os
import SMTS.predictSMTS as predictSMTS
import SMTS.trainSMTS as trainSMTS
import SMTS.tuneSMTS as tuneSMTS
import pandas as pd
# Add the parent directory to the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))



# reading the vensim model.
SDRuleObj = sdRex.SDRuleX()
SDRuleObj.read_vensim(current_dir+"/teacup.mdl")
# reading the parameter range data.
SDRuleObj.load_param_range(current_dir +"/range.xlsx")
# install fsspec if throws an error. python -m pip install fsspec

# simulate the data for the given output parameter.
SDRuleObj.simulate("avg",100,"Teacup Temperature")
sim_data = SDRuleObj.simulation_result


print(sim_data) 


classes = sim_data.iloc[:, -1:]
train_data = sim_data.iloc[:, :-1]

opt_params = tuneSMTS.tune_SMTS(train_data, classes)
print(opt_params)

# import trainSMTS as trainer
# import predictSMTS as predictor

# trainer.train_SMTS(train_data, classes, opt_params)

# df_t = pd.read_csv("CBF_TEST.csv", index_col=0 )
# classes_t = df_t.iloc[1:, -1:]
# test_data = df_t.iloc[:, :-1]
# pred = predictor.predict_SMTS(test_data,"model_data.joblib")



# observed_classes = list(test_data)
# predicted_classes = list(pred['classPred'])

# # Create Series objects
# observed_series = pd.Series(observed_classes, name='Observed')
# predicted_series = pd.Series(predicted_classes, name='Predicted')

# # Create a DataFrame and compute the cross-tabulation
# table = pd.crosstab(observed_series, predicted_series)
# table
# Display the table
# print(table)



# train the SD Metamodel
# SDRuleObj.train(estimators=100, test_size=0.25)
# extract the rules from the trained SD Metamodel.
# SDRuleObj.rule_extract()

