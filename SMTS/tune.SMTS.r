tune.SMTS <- function(trainingdata, classes, tuningParamLevels = list(noftreelevels = c(10, 25, 50), nofnodelevels =c (10, 25, 50))){
	
	require(randomForest)
	require(plyr)
	
	# setwd("C:/Users/Mert/Desktop/R Package/R Package v2")
	
	if(is.data.frame(classes) == FALSE)
		stop("classes must be a data.frame!")
	if(is.data.frame(trainingdata) == FALSE)
		stop("trainingdata must be a data.frame!")
		
	trclass <- classes
	colnames(trclass) <- "x"
	uniqueclass <- unique(trclass$x)
	uniqueclass <- uniqueclass[order(uniqueclass)] ## BATUHAN, I ADDED THIS
	uniqueclass <- data.frame(x = uniqueclass, ID = c(1:length(uniqueclass)))
	trclass <- join(trclass, uniqueclass, type = "left")

	trainingdata <- as.matrix(cbind(trclass$ID, trainingdata))
	
	# source("trainingdata_preparation.r")
	
	#following codes obtains the training and test matrix in the paper
	#these are the ones that goes into random forest model, RFobs

	datatraintimestart <- proc.time()
	classtrain <- trainingdata[, 1] 				# classes of the training time series
	# print(classtrain)
	noftrain <- nrow(trainingdata) 					# number of training series
	seriesLen <- apply(trainingdata[, 2:ncol(trainingdata)], 1, function(x) sum(!is.na(x)))	# length of each series
	observations <- array(0, sum(seriesLen)-noftrain)	# observation array (storing all observations as a column)
	# print(dim(observations))
	difference <- array(0, sum(seriesLen)-noftrain)		# difference array (storing difference between consecutive observations as a column)
	# print(length(difference))

	#for each time series observations should be standardized and
	#we need to concatenate the observation of each time series as single column
	#as well as the differences 
	st <- 1
	for(i in 1:noftrain){
		curseries <- trainingdata[i, !is.na(trainingdata[i, ])]
		curclass <- curseries[1]
		#standardize if necessary
		numseries <- as.numeric(curseries[2:length(curseries)])
		numseries <- (numseries-mean(numseries))/sd(numseries)
		en <- st+seriesLen[i]-2
		observations[st:en] <- numseries[2:length(numseries)]
		#print the last element of the observation array
		difference[st:en] <- diff(numseries)
		obsclass <- rep(curclass, seriesLen[i]-1)
		if(i == 1){
			allobsclass <- obsclass
		} else {
			allobsclass <- c(allobsclass, obsclass)
		}
		st <- en+1
	}
	# print(length(allobsclass))

	timeindices <- unlist(lapply(seriesLen,function(x) c(2:x))) #create time indices
	#final train matrix stores class,time index,observation and consecutive difference
	finaltrain <- data.frame(Class = allobsclass, timeindices, observations, difference)
	# print(finaltrain)
	ntrainobs <- seriesLen-1
	# print(ntrainobs)
	datatraintimeend <- proc.time()
	datatrainprepdur <- datatraintimeend-datatraintimestart
	datatrainprepdur <- datatrainprepdur[3]

	#algorithm for generating the distribution of symbols
	generatecodebook <- function(nodestatus, terminal, nofterminal, nofobservations) {
		if(!is.loaded("generate_codebook")) dyn.load("mts_functions64bit.dll")
		# print(paste("node_status.shape:", paste(dim(nodestatus), collapse = "x")))
		# print(paste("train_terminal.shape:", paste(dim(train_terminal), collapse = "x")))
		# print(paste("nofnode:", nofnode))
		# print(paste("ntrainobs:", ntrainobs))


		# print((nodestatus[2,]))
		nofseries <- length(nofobservations)
		noftree <- ncol(terminal)
		nofnode <- nrow(nodestatus)
		total <- sum(nofobservations)
		# print(seriesLen)
		# print("Total:") 
		# print(total)
		# print("Terminal:")
		# print(dim(terminal))
		nofentry <- nofseries*nofterminal*noftree
		# print((nofseries))
		# print(length(as.matrix(nodestatus)))
		# print((nofnode))
		# print(length(noftree))
		# print(dim(as.matrix(terminal)))
		# print((nofterminal))
		# print(length(nofobservations))
		# print(nofobservations)
		# print(length(total))
		# print((nofseries))
		# print((result))
		# print(length(as.integer(as.matrix(nodestatus))))
		out <- .C("generate_codebook", as.integer(as.matrix(nodestatus)), as.integer(nofnode), as.integer(noftree), as.integer(as.matrix(terminal)), as.integer(nofterminal), as.integer(nofobservations), as.integer(total), as.integer(nofseries), result = double(nofentry))
		return(out$result)
		# out <- generate_codebook(RFins$forest$nodestatus, nofnode, noftree, train_terminal, nofterminal, nofobservations, total, nofseries)
		# # # print(out)
		# return(out)
	}
	
	noftreelevels <- tuningParamLevels$noftreelevels
	nofnodelevels <- tuningParamLevels$nofnodelevels

	# source("parameter_selection_noparallel.r") 
	
	ntreeRFts <- 50
	t1 <- system.time({
		noftree <- 25
		# print( nofnodelevels)
		OOB_error_rate_node <- array(0, length(nofnodelevels))
		for(nd in 1:length(nofnodelevels)){		# for each nnumber of trees (J_{ins})
			nofnode <- nofnodelevels[nd]

			RFins <- randomForest(as.matrix(finaltrain[, 2:ncol(finaltrain)]), factor(finaltrain[, 1]), ntree = noftree, maxnodes = nofnode)
			num_features <- sqrt(ncol(as.matrix(finaltrain[, 2:ncol(finaltrain)])))
			prediction <- predict(RFins, finaltrain[, 2:ncol(finaltrain)], nodes = TRUE)
			train_terminal <- attr(prediction, "nodes")
			# print(paste("train_terminal.shape:", paste(dim(train_terminal), collapse = "x")))
			# last column of train_terminal is the terminal node number
			# print(train_terminal[,ncol(train_terminal)])
			codebook <- generatecodebook(RFins$forest$nodestatus,train_terminal, nofnode, ntrainobs)
			codetr <- matrix(codebook, noftrain, noftree*nofnode)
			#print the sum of the first row of codetr
			# print(sum(codetr[1,]))
			RFts <- randomForest(codetr, as.factor(classtrain), ntree = ntreeRFts)
			# What is RFts node count
			OOB_error_rate_node[nd] <- 1-sum(predict(RFts, type = "response") == as.factor(classtrain))/noftrain
		}
	})
	# for (identifier in OOB_error_rate_node) {
	#    print(identifier)
	# }
	
	t2 <- system.time({
		OOB_error_rate <- array(0, length(noftreelevels))
		nofnode <- nofnodelevels[which.min(OOB_error_rate_node)]
		for(nd in 1:length(noftreelevels)){		# for each nnumber of trees (J_{ins})
			noftree <- noftreelevels[nd]
			RFins <- randomForest(finaltrain[, 2:ncol(finaltrain)], factor(finaltrain[, 1]), ntree = noftree, maxnodes = nofnode)
			train_terminal <- attr(predict(RFins, finaltrain[, 2:ncol(finaltrain)], nodes = TRUE), "nodes")
			codetr <- matrix(generatecodebook(RFins$forest$nodestatus, train_terminal, nofnode, ntrainobs), noftrain, noftree*nofnode)	
			RFts <- randomForest(codetr, as.factor(classtrain), ntree = ntreeRFts)
			OOB_error_rate[nd] <- 1-sum(predict(RFts, type = "response") == as.factor(classtrain))/noftrain
		}
		
		noftree <- noftreelevels[which.min(OOB_error_rate)]
	})

	optParams = list(noftree = noftree, nofnode = nofnode)
	return(optParams)
	
}

generate_codebook <- function(nodestatus, nofnode, noftree, terminal, nofterminal, nofobservations, total, nofseries) {
  result <- double(nofseries * nofterminal * noftree)  # Allocate the result array
  
  for (i in 1:noftree) {
    temp <- 0
    for (j in 1:nofnode) {
      if (nodestatus[(i - 1) * nofnode + j] < 0) {
        nodestatus[(i - 1) * nofnode + j] <- temp
        temp <- temp + 1
      }
    }
    temp <- 0
    for (k in 1:nofseries) {
      for (j in 1:nofterminal) {
        result[(nofseries * nofterminal * (i - 1)) + (j - 1) * nofseries + k] <- 0
      }
      tmp <- nofobservations[k]
      for (j in 1:nofobservations[k]) {
        ind <- terminal[(total * (i - 1)) + temp + j] - 1
        index <- nodestatus[(i - 1) * nofnode + ind]
        result[(nofseries * nofterminal * (i - 1)) + (index - 1) * nofseries + k] <- result[(nofseries * nofterminal * (i - 1)) + (index - 1) * nofseries + k] + 1 / tmp
      }
      temp <- temp + nofobservations[k]
    }
  }
#   print("Result:")
#   print(result)
  return(result)  # Return the result array
}
