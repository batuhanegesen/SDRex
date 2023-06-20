from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import pandas as pd
from ctypes import cdll, c_int, c_double, POINTER
from collections import Counter
from matplotlib import pyplot as plt
from sklearn import tree
import os

import warnings

warnings.filterwarnings("ignore")

def tune_SMTS(trainingdata, classes, tuningParamLevels={'noftreelevels': [10, 25, 50], 'nofnodelevels': [10, 25, 50]}):
    if not isinstance(classes, pd.DataFrame):
        raise ValueError("classes must be a data.frame!")
    if not isinstance(trainingdata, pd.DataFrame):
        raise ValueError("trainingdata must be a data.frame!")

    trclass = classes.copy()
    trclass.columns = ['x']
    uniqueclass = np.unique(trclass['x'])
    uniqueclass = pd.DataFrame(
        {'x': uniqueclass, 'ID': np.arange(1, len(uniqueclass) + 1)})

    trclass = trclass.merge(uniqueclass, on='x', how='left')
    trainingdata = np.hstack(
        (trclass['ID'].values.reshape(-1, 1), trainingdata))
    # trainingdata = np.matrix(trainingdata)
    trainingdata = np.asarray(trainingdata, dtype=np.float64)

    datatraintimestart = time.process_time()

    # classes of the training time series
    classtrain = trainingdata[:, 0]
    # classtrain = np.ravel(classtrain)
    noftrain = trainingdata.shape[0]               # number of training series
    seriesLen = np.apply_along_axis(lambda x: np.sum(
        ~np.isnan(x)), axis=1, arr=trainingdata[:, 1:])  # length of each series
    # print(seriesLen)
    # observation array (storing all observations as a column)
    observations = np.zeros(sum(seriesLen) - noftrain)
    # difference array (storing difference between consecutive observations as a column)
    difference = np.zeros(sum(seriesLen) - noftrain)

    st = 0
    for i in range(noftrain):
        trainingdata = pd.DataFrame(trainingdata)
        curseries = trainingdata.iloc[i, :][~np.isnan(trainingdata.iloc[i, :])]
        curclass = curseries[0]
        # standardize if necessary
        numseries = np.array(curseries[1:])
        numseries = (numseries - np.mean(numseries)) / np.std(numseries)
        # print(length(numseries))
        en = st + seriesLen[i] - 2

        observations[st:en+1] = numseries[1:]
        #print the last element of observations
        # print(observations[en])
        difference[st:en+1] = np.diff(numseries)
        # print(difference[en])
        #print the length of difference
        # print(len(difference))
        obsclass = np.repeat(curclass, seriesLen[i] - 1)
		#print the last element of the observation class array
        # print(obsclass[len(obsclass)-1])
        if i == 0:
            allobsclass = obsclass
        else:
            allobsclass = np.concatenate((allobsclass, obsclass))
        st = en + 1

    timeindices = np.concatenate([np.arange(1, x) for x in seriesLen])

    finaltrain = pd.DataFrame({
        'Class': allobsclass,
        'timeindices': timeindices,
        'observations': observations,
        'difference': difference
    })
    ntrainobs = seriesLen - 1

    datatraintimeend = time.perf_counter()
    datatrainprepdur = datatraintimeend - datatraintimestart

    noftreelevels = tuningParamLevels['noftreelevels']
    nofnodelevels = tuningParamLevels['nofnodelevels']

    ntreeRFts = 50
    t1 = time.time()
    noftree = 25
    OOB_error_rate_node = np.zeros(len(nofnodelevels))

    # for each nnumber of trees (J_{ins})
    for nd in range(len(nofnodelevels)):
        nofnode = nofnodelevels[nd]
        RFins = RandomForestClassifier(
            n_estimators=noftree, max_leaf_nodes=nofnode, bootstrap=True, oob_score=True)
        RFins.fit(finaltrain.iloc[:, 1:],
                  finaltrain.iloc[:, 0]) 
        # num_features = int(np.sqrt(finaltrain.iloc[:, 1:].shape[1])) #it was 2
        # predictions = RFins.predict(finaltrain.iloc[:, 1:])
        train_terminal = RFins.apply(finaltrain.iloc[:, 1:])
        node_status = make_node_status(RFins)
        codetr = generate_codebook(
            node_status, train_terminal, nofnode, ntrainobs)
        RFts = RandomForestClassifier(
            n_estimators=ntreeRFts, bootstrap=True, oob_score=True)
        RFts.fit(np.asarray(codetr, dtype=np.float64), classtrain)
        # predictions = RFts.predict(codetr)
        # Calculate OOB error rate
        OOB_error_rate_node[nd] = 1 - RFts.oob_score_
        # print("RFins OOB Error Rate: ", OOB_error_rate_node[nd])
        # print("sum of the first row", np.sum(codetr[0,:]))

    
    t1_end = time.time()
    t1 = t1_end - t1
   
    t2 = time.time()
    OOB_error_rate = np.zeros(len(noftreelevels))
    nofnode = nofnodelevels[np.argmin(OOB_error_rate_node)]
    # # for each nnumber of trees (J_{ins})
    for nd in range(len(noftreelevels)):
        noftree = noftreelevels[nd]
        RFins = RandomForestClassifier(n_estimators=noftree, max_leaf_nodes=nofnode,bootstrap=True, oob_score=True).fit(
            finaltrain.iloc[:, 1:], finaltrain.iloc[:, 0])
        train_terminal = RFins.apply(finaltrain.iloc[:, 1:])
        node_status = make_node_status(RFins)
        codetr = generate_codebook(node_status, train_terminal, nofnode, ntrainobs)
        RFts = RandomForestClassifier(
            n_estimators=ntreeRFts,bootstrap=True, oob_score=True).fit(codetr, classtrain)
        OOB_error_rate[nd] = 1 - RFts.oob_score_
        print("RFts OOB Error Rate: ", OOB_error_rate[nd])

    t2_end = time.time()
    t2 = t2_end - t2
    noftree = noftreelevels[np.argmin(OOB_error_rate)]

    optParams = {'noftree': noftree, 'nofnode': nofnode}
    return (optParams)


def make_node_status(forest: RandomForestClassifier):
    node_status = []
    SPLIT_VALUE = 1
    LEAF_VALUE = -1
    MAX_LENGTH = 0
    for decisionTree in forest.estimators_:
        tree = decisionTree.tree_
        # print(clf.node_count)
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right

        # we need to know how deep this forest is in general.
        if (n_nodes > MAX_LENGTH):
            MAX_LENGTH = n_nodes

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=np.int64)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
                is_leaves[node_id] = SPLIT_VALUE
            else:
                is_leaves[node_id] = LEAF_VALUE
        node_status.append(is_leaves)
    newnodestatus = []
    for treestatus in node_status:
        while (len(treestatus) < MAX_LENGTH):
            # treestatus = np.append(treestatus, -1)
            treestatus = np.append(treestatus, 0)
        newnodestatus.append(treestatus)
    # print((node_status[0]))
    node_status = np.array(newnodestatus)
    node_status = np.transpose(node_status)
    # print(node_status[:, 24])
    # print((node_status[1]))
    # print(node_status)
    # print(node_status.shape)
    # tree.plot_tree(estimator)
    # plt.show()
    # print(
    # "The binary tree structure has {n} nodes and has "
    # "the following tree structure:\n".format(n=n_nodes)
    # )
    # print(is_leaves)
    # for i in range(n_nodes):
    #     if is_leaves[i] == 1:
    #         print(
    #             "{space}node={node} is a leaf node.".format(
    #                 space=node_depth[i] * "\t", node=i
    #             )
    #         )
    #     else:
    #         print(
    #             "{space}node={node} is a split node: ".format(
    #                 space=node_depth[i] * "\t",
    #                 node=i
    #             )
    #         )
    return node_status

# def generatecodebook(nodestatus, terminal, nofterminal, nofobservations):

#     index_errors = []
#     nofnode = len(nodestatus)
#     noftree = terminal.shape[1]
#     nofseries = len(nofobservations)
#     total = sum(nofobservations)
#     nofentry = nofseries * nofterminal * noftree
#     result = np.zeros(nofentry, dtype=np.float64)
#     nodestatus = nodestatus.flatten()
#     terminal = terminal.flatten()
#     # print all variables above with their respective names
#     # print("nodestatus length: ", nodestatus.shape[0])
#     # print("terminal: ", terminal.shape[0])
#     # print("nofterminal: ", nofterminal)
#     # print("nofobservations shape: ", nofobservations.shape)
#     # print("nofobservations: ", nofobservations)
#     # print("nofnode: ", nofnode)
#     # print("noftree: ", noftree)
#     # print("nofseries: ", nofseries)
#     # print("total: ", total)
#     # print("nofentry: ", nofentry)
#     # print("result shape: ", result.shape)
#     # print("************************\n")
    

#     for treeIndex in range(noftree):
#         temp = 0
#         for nodeIndex in range(nofnode):
#             if nodestatus[treeIndex*nofnode + nodeIndex] < 0:
#                 nodestatus[treeIndex*nofnode + nodeIndex] = temp
#                 temp += 1

#         temp = 0
#         for seriesIndex in range(nofseries):
#             for nodeIndex in range(nofterminal):
#                 # print("nofseries*(treeIndex*nofterminal + nodeIndex) + seriesIndex: ", nofseries*(treeIndex*nofterminal + nodeIndex) + seriesIndex)
#                 result[nofseries*(treeIndex*nofterminal + nodeIndex) + seriesIndex] = 0

#             tmp = nofobservations[seriesIndex]
#             for nodeIndex in range(nofobservations[seriesIndex]):
#                 ind = terminal[total*treeIndex + temp + nodeIndex] - 1
#                 index = nodestatus[treeIndex*nofnode + ind]
#                 # result[nofseries*(treeIndex*nofterminal + index) + seriesIndex] += 1/tmp
#                 try:
#                     result[nofseries*(treeIndex*nofterminal + index) + seriesIndex] += 1/tmp
#                 except: 
#                     # print("ind: ",ind, " **** index: ", index)
#                     index_errors.append(nofseries*(treeIndex*nofterminal + index) + seriesIndex)
                
#             temp += nofobservations[seriesIndex]
#     if(index_errors.count != 0):
#         # print("index_errors: ", index_errors)
#         # print("index_errors count: ", len(index_errors))
#         # print("unique index_errors: ", np.unique(index_errors))
#         print("unique index_errors count: ", len(np.unique(index_errors)))
#     result = result.reshape((nofseries, nofterminal * noftree))
#     result = np.asmatrix(result)
#     return result


def generate_codebook(nodestatus, terminal, nofterminal, nofobservations):
        lib = cdll.LoadLibrary(os.getcwd()+"/mts_functions64bit.dll")

        # Get dimensions
        nofnode = len(nodestatus)
        noftree = terminal.shape[1]
        nofseries = len(nofobservations)
        total = sum(nofobservations)
        nofentry = nofseries * nofterminal * noftree

        # Define argument and return types for the C function
        generate_codebook = lib.generate_codebook
        generate_codebook.argtypes = [
            POINTER(c_int),  # nodestatus
            POINTER(c_int),  # nofnode
            POINTER(c_int),  # noftree
            POINTER(c_int),  # terminal
            POINTER(c_int),  # nofterminal
            POINTER(c_int),  # nofobservations
            POINTER(c_int),  # total
            POINTER(c_int),  # nofseries
            POINTER(c_double)  # result
        ]
        generate_codebook.restype = None

        # Convert input data to appropriate ctypes types
        
        nodestatus_arr = np.asmatrix(nodestatus, dtype=np.int32)
        terminal_arr = np.asmatrix(terminal, dtype=np.int32)
        nofnode_c = c_int(nofnode)
        noftree_c = c_int(noftree)
        nofterminal_c = c_int(nofterminal)
        nofobservations_arr = np.array(nofobservations, dtype=np.int32)
        total_c = c_int(total)
        nofseries_c = c_int(nofseries)
        result = np.zeros((nofentry,), dtype=np.float64)
        result = np.asmatrix(result, dtype=np.float64)

        # print("nodestatus_arr:")
        # print("Dimensions:", nodestatus_arr.shape)
        # print("Length:", len(nodestatus_arr))
        # print("Values:", nodestatus_arr)
        # print(nodestatus_arr[22])

        # print("\nterminal_arr:")
        # print("Dimensions:", terminal_arr.shape)
        # print("Length:", len(terminal_arr))
        # print("Values:", terminal_arr)

        # print("\nnofnode_c:")
        # print("Value:", nofnode_c.value)

        # print("\nnoftree_c:")
        # print("Value:", noftree_c.value)

        # print("\nnofterminal_c:")
        # print("Value:", nofterminal_c.value)

        # print("\nnofobservations_arr:")
        # print("Dimensions:", nofobservations_arr.shape)
        # print("Length:", len(nofobservations_arr))
        # print("Values:", nofobservations_arr)

        # print("\ntotal_c:")
        # print("Value:", total_c.value)

        # print("\nnofseries_c:")
        # print("Value:", nofseries_c.value)

        # print("\nresult:")
        # print("Dimensions:", result.shape)
        # print("Length:", len(result))
        # print("Values:", result)
        # print("\n************************\n")

        # Call the C function
        generate_codebook(nodestatus_arr.ctypes.data_as(POINTER(c_int)),
                        nofnode_c,
                        noftree_c,
                        terminal_arr.ctypes.data_as(POINTER(c_int)),
                        nofterminal_c,
                        nofobservations_arr.ctypes.data_as(POINTER(c_int)),
                        total_c,
                        nofseries_c,
                        result.ctypes.data_as(POINTER(c_double)))

        # Reshape the result to a 2-dimensional matrix
        result = result.reshape((nofseries, nofterminal * noftree))
        result = np.asmatrix(result)
        return result




# df = pd.read_csv("CBF_TRAIN.csv", header=None)
# classes = df.iloc[:, -1:]
# train_data = df.iloc[:, :-1]

# opt_params = tune_SMTS(train_data, classes)

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
