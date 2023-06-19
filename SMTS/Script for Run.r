library(jmotif) # Bu paketter kullanılabilecek datasetleri var. Biz CBF'i deneme amaçlı kullanabiliriz (Alternatif: Gun_Point)
require(randomForest)
require(plyr)
# rfmodel <- randomForest(Species ~ ., data = iris, ntree = 10)
# # print(rfmodel$forest$nodestatus)
# print(getTree(rfmodel, 6))""

setwd("D:/GitHub/thesis")	# Fonksiyonların bulunduğu directory
source("tune.SMTS.r")
source("train.SMTS.r")
source("predict.SMTS.r")

optparams = tune.SMTS(trainingdata = as.data.frame(CBF$data_train), classes = as.data.frame(CBF$labels_train))
# print(optparams)
train.SMTS(trainingdata = as.data.frame(CBF$data_train), classes = as.data.frame(CBF$labels_train), tuningParams = optparams)
pred = predict.SMTS(newdata = as.data.frame(CBF$data_test), modelPath = "D:/GitHub/thesis/model.rds")
# print(table(CBF$labels_test, pred$classPred))

# install.packages("devtools")
# library(devtools)
# install_github('jMotif/jmotif-R')
