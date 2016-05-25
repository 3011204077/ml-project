from random import shuffle
from ID3 import *
from operator import xor
from parse import parse
import matplotlib.pyplot as plt
import os.path
from pruning import validation_accuracy
import random
import copy

def curve_data(data_set, curve):
    data_set1 = []
    pool = []
    if curve != 1:
        for i in range(0, int(curve * len(data_set))):
            randindex = random.randint(0, int(curve * len(data_set)) - 1)
            while (1):
                if pool.count(randindex) != 0:
                    randindex = random.randint(0, int(curve * len(data_set)) - 1)
                else:
                    break
            data_set1.append(data_set[randindex])
            pool.append(randindex)
        return data_set1
    else:
        return data_set
    
def get_graph_accuracy_partial(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, pct):
    '''
    get_graph_accuracy_partial - Given a training set, attribute metadata, validation set, numerical splits count, and percentage,
    this function will return the validation accuracy of a specified (percentage) portion of the trainging setself.
    '''
    data_set = curve_data(train_set, pct)
    if data_set != []:
        curve_tree = ID3(data_set, attribute_metadata, numerical_splits_count, depth)
        return validation_accuracy(curve_tree, validate_set, attribute_metadata)
    else:
        return 0
    pass

def get_graph_data(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, pcts):
    '''
    Given a training set, attribute metadata, validation set, numerical splits count, iterations, and percentages,
    this function will return an array of the averaged graph accuracy partials based off the number of iterations.
    '''
    pool = []
    graph_data = []
    for pct in pcts: 
        for j in range(0, iterations):
            origin_splits_count = copy.deepcopy(numerical_splits_count)
            pool.append(get_graph_accuracy_partial(train_set, attribute_metadata, validate_set, origin_splits_count, depth, pct))
        graph_data.append(sum(pool) / len(pool))
        #print pool
        pool = []
    return graph_data
    pass

# get_graph will plot the points of the results from get_graph_data and return a graph
def get_graph(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, lower, upper, increment):
    '''
    get_graph - Given a training set, attribute metadata, validation set, numerical splits count, depth, iterations, lower(range),
    upper(range), and increment, this function will graph the results from get_graph_data in reference to the drange
    percentages of the data.
    '''
    pcts = []
    i = lower
    while i <= upper:
        if i == 0:
            i += increment
            continue
        else:
            pcts.append(i)
        i += increment
    graph_data = get_graph_data(train_set, attribute_metadata, validate_set, numerical_splits_count, depth, iterations, pcts)
    plt.plot(pcts, graph_data, 'xb-')
    plt.ylabel('Accuracy')
    plt.xlabel('Curve')
    plt.axis([lower, upper, 0, 100])
    plt.show()
    pass