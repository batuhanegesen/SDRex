from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import pandas as pd
from ctypes import cdll, c_int, c_double, POINTER
from collections import Counter
from matplotlib import pyplot as plt
from sklearn import tree
import os
import joblib
import json


def predict_SMTS(newdata: pd.DataFrame, model_dict_path: str):
    # def predict_SMTS(newdata: pd.DataFrame):
    # newdata: a dataframe with the same columns as the training data
    # RFinsModel: the RFins model
    # RFtsModel: the RFts model
    # model_dict: a dictionary with the following keys:
    #     noftree: number of trees in the forest
    #     nofnode: number of nodes in each tree
    #     classInfo: a dictionary. Keys are the class names, values are the
    #                corresponding class indices
    nofnew = newdata.shape[0]
    seriesLen = np.apply_along_axis(
        lambda x: np.sum(~np.isnan(x)), axis=1, arr=newdata)
    # print('seriesLen length:', len(seriesLen))
    # print the newdata shape
    # print('newdata shape:', newdata.shape)
    observations = np.zeros(sum(seriesLen) - nofnew)
    difference = np.zeros(sum(seriesLen) - nofnew)
    st = 0
    for i in range(nofnew):
        # curseries = newdata[i, ~np.isnan(newdata[i])]
        curseries = newdata.iloc[i, np.logical_not(
            newdata.iloc[i].isna().to_numpy())]
        # standardize if necessary
        numseries = np.array(curseries, dtype=float)
        numseries = (numseries - np.mean(numseries)) / np.std(numseries)
        # print(len(numseries))

        # print(numseries.shape)
        en = st + seriesLen[i] - 2
        observations[st:en+1] = numseries[0:len(numseries)-1]
        difference[st:en+1] = np.diff(numseries)
        st = en + 1
    timeindices = np.concatenate([np.arange(1, x) for x in seriesLen])
    finalnew = pd.DataFrame(
        {'timeindices': timeindices, 'observations': observations, 'difference': difference})
    nnewobs = seriesLen - 1

    RFins = RandomForestClassifier()
    RFts = RandomForestClassifier()
    classInfo = {}
    nofnode = 0
    noftree = 0

    model_dict = joblib.load(model_dict_path)
    RFins = model_dict['RFins']
    RFts = model_dict['RFts']
    classInfo = pd.DataFrame(model_dict['classInfo'])
    nofnode = model_dict['nofnode']
    noftree = model_dict['noftree']

    # print("nofnode: ", nofnode)
    # print("noftree: ", noftree)
    # print("classInfo:  \n", classInfo)
    # print(RFins.estimators_[0].tree_.node_count)
    # print(RFts.estimators_[0].tree_.node_count)

    # print(finalnew)
    new_terminal = RFins.apply(finalnew)
    node_status = make_node_status(RFins)
    codenew = generate_codebook(node_status, new_terminal, nofnode, nnewobs)
    codenew = codenew.reshape(nofnew, noftree * nofnode)

    predicted_prob = RFts.predict_proba(codenew)
    class_labels = classInfo['x']
    predicted_prob = pd.DataFrame(predicted_prob, columns=class_labels)
    predicted_class = RFts.predict(codenew)

    predicted_class_new = np.empty_like(predicted_class, dtype=object)

    classPred = predicted_class.copy()
    probVals = predicted_prob.copy()

    for j in range(classInfo.shape[0]):
        classPred[classPred == classInfo['ID'][j]] = classInfo['x'][j]

    return {'classPred': classPred, 'probVals': probVals}


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
