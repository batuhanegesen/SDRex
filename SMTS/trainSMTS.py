from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import pandas as pd
from ctypes import cdll, c_int, c_double, POINTER
from collections import Counter
from matplotlib import pyplot as plt
from sklearn import tree
import os
from joblib import dump
import pickle
import json
import warnings

warnings.filterwarnings("ignore")


def train_SMTS(trainingdata, classes, tuningParams={"noftree": 50, "nofnode": 10}, params={"maxiter": 20, "noftree_step": 50, "tolerance": 0.05}, saveModel=True, savePath=os.getcwd()):

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
    trainingdata = np.asarray(trainingdata, dtype=np.float64)

    datatraintimestart = time.process_time()

    # classes of the training time series
    classtrain = trainingdata[:, 0]
    # classtrain = np.ravel(classtrain)
    noftrain = trainingdata.shape[0]    # number of training series
    seriesLen = np.apply_along_axis(lambda x: np.sum(
        ~np.isnan(x)), axis=1, arr=trainingdata[:, 1:])  # length of each series
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
        en = st + seriesLen[i] - 2

        observations[st:en+1] = numseries[1:]
        difference[st:en+1] = np.diff(numseries)
        obsclass = np.repeat(curclass, seriesLen[i] - 1)
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

    noftree = tuningParams['noftree']
    nofnode = tuningParams['nofnode']

    RFins = RandomForestClassifier(
        n_estimators=noftree, max_leaf_nodes=nofnode, bootstrap=True, oob_score=True)
    RFins.fit(finaltrain.iloc[:, 1:], finaltrain.iloc[:, 0])
    train_terminal = RFins.apply(finaltrain.iloc[:, 1:])
    node_status = make_node_status(RFins)
    codetr = generate_codebook(node_status, train_terminal, nofnode, ntrainobs)

    noftree_step = params['noftree_step']
    tolerance = params['tolerance']
    maxiter = params['maxiter']

    RFts = RandomForestClassifier(
        n_estimators=noftree_step, bootstrap=True, oob_score=True)
    RFts.fit(np.asarray(codetr, dtype=np.float64), classtrain)

    prev_OOBerror = 1
    cur_OOBerror = 1 - RFts.oob_score_

    iter = 1

    while (iter < 20 and cur_OOBerror < (1-tolerance) * prev_OOBerror):
        prev_OOBerror = cur_OOBerror
        RFtsmid = RandomForestClassifier(
            n_estimators=noftree_step, bootstrap=True, oob_score=True, min_samples_leaf=2)
        RFtsmid.fit(np.asarray(codetr, dtype=np.float64), classtrain)
        RFts = combine_rfs(RFts, RFtsmid)
        cur_OOBerror = 1 - RFts.oob_score_
        iter = iter + 1
    OOB_error = 1 - RFts.oob_score_
    print("train OOB Error: ", OOB_error)
    
    uniqueclass_dict = uniqueclass.to_dict(orient='list')
    model_dict = {"RFins": RFins,
                  "RFts":RFts,
                  "noftree": noftree,
                  "nofnode": nofnode,
                  "classInfo": uniqueclass_dict}
    
    if saveModel:
        dump(model_dict,"model_data.joblib")
    else:
        return model_dict
    print("OOB error is", OOB_error)
    # RFts and RFins models are going to be saved in the same folder
    # params['savePath'] is the path of the folder
    # if saveModel:
    #     dump(RFts, savePath + '/RFts.joblib')
    #     dump(RFins, savePath + '/RFins.joblib')
    #     with open(savePath + '/model_dict.json', 'w') as fp:
    #         json.dump(model_dict, fp)
    # else:
    #     return model_dict, RFts, RFins
    # print("OOB error is", OOB_error)


def make_node_status(forest: RandomForestClassifier):
    node_status = []
    SPLIT_VALUE = 1
    LEAF_VALUE = -1
    MAX_LENGTH = 0
    for decisionTree in forest.estimators_:
        tree = decisionTree.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right

        # we need to know the max node depth of this forest.
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
            treestatus = np.append(treestatus, 0)
        newnodestatus.append(treestatus)
    node_status = np.array(newnodestatus)
    node_status = np.transpose(node_status)
    return node_status


def generate_codebook(nodestatus, terminal, nofterminal, nofobservations):
    lib = cdll.LoadLibrary(
        os.getcwd()+"/mts_functions64bit.dll")

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

# a method to combine two random forests


def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a
