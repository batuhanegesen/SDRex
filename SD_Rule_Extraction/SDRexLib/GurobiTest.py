from gurobipy import *

#"C://Users//beges//OneDrive//Desktop//School//Mert Edali//iris.csv"
model = Model("mip1")

x = model.addVar(vtype=GRB.BINARY, name="x")
y = model.addVar(vtype=GRB.BINARY, name="y")
z = model.addVar(vtype=GRB.BINARY, name="z")

model.update()

model.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

model.addConstr(x + 2 * y + 3 * z <= 4, "c0")
model.addConstr(x + y >= 1, "c1")

model.optimize()
model.printAttr("X")