import SDRex as sdRex
#from gurobipy import gurobi as gp



# reading the vensim model.
SDRuleObj = sdRex.SDRuleX()
SDRuleObj.read_vensim("C://Users/beges/OneDrive/Desktop/TryLocation/teacup.mdl")
# reading the parameter range data.
SDRuleObj.load_param_range("C://Users/beges/OneDrive/Desktop/TryLocation/range.xlsx")
# install fsspec if throws an error. python -m pip install fsspec

# simulate the data for the given output parameter.
SDRuleObj.simulate("avg",100,"Teacup Temperature")
# train the SD Metamodel
SDRuleObj.train(estimators=100, test_size=0.25)
# extract the rules from the trained SD Metamodel.
SDRuleObj.rule_extract()

