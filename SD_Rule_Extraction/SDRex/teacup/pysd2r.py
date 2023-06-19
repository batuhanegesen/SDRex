from pprint import PrettyPrinter, pprint
from unittest import result
from pandas import array
import pysd
import matplotlib.pyplot as plt

model = pysd.read_vensim('teacup.mdl')
# print(model.doc)
# model = pysd.load('Teacup.py')
# model = pysd.read_xmile('Teacup.mdl')
# pprint(model.doc)

# stocks = model.run(progress=True)
stocks = model.run(flatten_output=True)
pprint(stocks)  

stocks["Teacup Temperature"].plot()
plt.title("Teacup Temperature")
plt.ylabel("Degrees F")
plt.xlabel("Minutes")
plt.grid()






# param1,param2,param3,param4,resultAvg = 23



# [[([24.242424,64.973595,35.723663,54.8451531]),(34.532)],
# [([54.344615,43.7454524,32.13533,92.3251155]),(43.453)],
# [([73.264535,27.373515,62.121538,14.9541541]),(25.175)]]


# [(param1,param2,param3,param4),(resultAvg)]




