import math
from node import Node
import sys
import copy
from data_preprocessing import *

def ID3(data_set, attribute_metadata, numerical_splits_count, depth):
    '''
    ========================================================================================================
    Input:  A data_set, attribute_metadata, maximum number of splits to consider for numerical attributes,
	maximum depth to search to (depth = 0 indicates that this node should output a label)
    ========================================================================================================
    Output: The node representing the decision tree learned over the given data set
    ========================================================================================================

    '''
    preprocessing(data_set, attribute_metadata)
    if check_homogenous(data_set) != None:
        root = Node()
        root.label = check_homogenous(data_set)
    else: 
        if depth == 0:
            root = Node()
            root.label = mode(data_set)
        else:
            best = pick_best_attribute(data_set, attribute_metadata, numerical_splits_count)
            if best[0] == False:
                root = Node()
                root.label = mode(data_set)
            else:
                root = Node()
                root.decision_attribute = best[0]
                root.name = attribute_metadata[best[0]]['name']
                depth -= 1
                if str(best[1]) == 'False':
                    root.is_nominal = True
                    root.children = {}
                    subsets = split_on_nominal(data_set, best[0])
                    for splitval in subsets.keys():
                        root.children[splitval] = ID3(subsets[splitval], attribute_metadata, numerical_splits_count, depth)
                else:
                    root.is_nominal = False
                    root.children = []
                    root.splitting_value = best[1]
                    subsets = split_on_numerical(data_set, best[0], best[1])
                    #numerical_splits_count[best[0]] -= 1
                    print numerical_splits_count
                    print depth
                    root.children.append(ID3(subsets[0], attribute_metadata, numerical_splits_count, depth))
                    root.children.append(ID3(subsets[1], attribute_metadata, numerical_splits_count, depth)) 
    return root

def check_homogenous(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Checks if the attribute at index 0 is the same for the data_set, if so return output otherwise None.
    ========================================================================================================
    Output: Return either the homogenous attribute or None
    ========================================================================================================
    '''
    for i in range(0, len(data_set) - 1):
        if data_set[i][0] == data_set[i + 1][0]:
            continue
        else:
            return None
    return data_set[0][0]


def pick_best_attribute(data_set, attribute_metadata, numerical_splits_count):
    '''
    ========================================================================================================
    Input:  A data_set, attribute_metadata, splits counts for numeric
    ========================================================================================================
    Job:    Find the attribute that maximizes the gain ratio. If attribute is numeric return best split value.
            If nominal, then split value is False.
            If gain ratio of all the attributes is 0, then return False, False
            Only consider numeric splits for which numerical_splits_count is greater than zero
    ========================================================================================================
    Output: best attribute, split value if numeric
    ========================================================================================================
    '''
    i = 0
    maxgratio_num = 0
    pool = {}
    #splival_atmax = 0
    for entry in attribute_metadata:
        if entry['name'] == "winner":
            i += 1
            continue
        if entry['is_nominal'] == True:
            gratio = gain_ratio_nominal(data_set, i)
            pool[gratio] = i
            i += 1
        if entry['is_nominal'] == False:
            if numerical_splits_count[i] != 0:
                gratio = gain_ratio_numeric(data_set, i, 1)
                if gratio[0] >= maxgratio_num:
                    maxgratio_num = gratio[0]
                    splival_atmax = gratio[1]
                pool[gratio[0]] = i
                i += 1
            else:
                pool[0] = i
                i += 1
    if pool == {} or max(pool.keys()) == 0:
        return (False, False)
    else: 
        index = pool[max(pool.keys())]
        if attribute_metadata[index]['is_nominal'] == True:
            return (index, False)
        else:
            numerical_splits_count[index] -= 1
            return (index, splival_atmax)


# Uses gain_ratio_nominal or gain_ratio_numeric to calculate gain ratio.

def mode(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Takes a data_set and finds mode of index 0.
    ========================================================================================================
    Output: mode of index 0.
    ========================================================================================================
    '''
    num0 = 0
    num1 = 0
    for entry in data_set:
        if entry[0] == 0:
            num0 += 1
        else:
            num1 += 1
    if num0 > num1:
        return 0
    else:
        return 1

def entropy(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Calculates the entropy of the attribute at the 0th index, the value we want to predict.
    ========================================================================================================
    Output: Returns entropy. 
    ========================================================================================================
    '''
    numEntries = len(data_set)
    freq0 = 0
    freq1 = 0
    for entry in data_set:
        if entry[0] == 0:
            freq0 += 1
        else:
            freq1 += 1
    prob0 = float(freq0)/numEntries
    prob1 = float(freq1)/numEntries
    if prob0 == 0 or prob1 == 0:
        dataEntropy = 0
    else:
        dataEntropy = -prob0 * math.log(prob0, 2) - prob1 * math.log(prob1, 2)
    return dataEntropy


def gain_ratio_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  Subset of data_set, index for a nominal attribute
    ========================================================================================================
    Job:    Finds the gain ratio of a nominal attribute in relation to the variable we are training on.
    ========================================================================================================
    Output: Returns gain_ratio. See https://en.wikipedia.org/wiki/Information_gain_ratio
    ========================================================================================================
    '''
    splitnom = split_on_nominal(data_set, attribute)
    numEntries = len(data_set)
    dataEntropy = entropy(data_set)
    IntrVal = 0
    subsetEntropy = 0
    for splitval in splitnom.keys():
        prob = float(len(splitnom[splitval])) / numEntries
        IntrVal += -prob * math.log(prob, 2)
        subsetEntropy += entropy(splitnom[splitval]) * prob
    InfoGain = dataEntropy - subsetEntropy
    if InfoGain == 0:
        return 0
    else:
        return InfoGain / IntrVal
    return InfoGain

def gain_ratio_numeric(data_set, attribute, steps):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, and a step size for normalizing the data.
    ========================================================================================================
    Job:    Calculate the gain_ratio_numeric and find the best single threshold value
            The threshold will be used to split examples into two sets
                 those with attribute value GREATER THAN OR EQUAL TO threshold
                 those with attribute value LESS THAN threshold
    ========================================================================================================
    Output: This function returns the gain ratio and threshold value
    ========================================================================================================
    '''
    pool = {}
    for i in range(0, len(data_set)):
        if i % steps == 0:
            splitnum = split_on_numerical(data_set, attribute, data_set[i][attribute])
            if splitnum[0] == [] or splitnum[1] == []:
                gnratio = 0
            else:
                templeft = copy.deepcopy(splitnum[0])
                tempright = copy.deepcopy(splitnum[1])
                for entry in templeft:
                    entry[attribute] = 0
                for entry in tempright:
                    entry[attribute] = 1
                new_data_set = templeft + tempright
                gnratio = gain_ratio_nominal(new_data_set, attribute)
            pool[gnratio] = data_set[i][attribute]         #may have bug
    return (max(pool.keys()), pool[max(pool.keys())])

def split_on_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  subset of data set, the index for a nominal attribute.
    ========================================================================================================
    Job:    Creates a dictionary of all values of the attribute.
    ========================================================================================================
    Output: Dictionary of all values pointing to a list of all the data with that attribute
    ========================================================================================================
    '''
    splitnom = {}
    for entry in data_set:
        if splitnom.has_key(entry[attribute]):
            splitnom[entry[attribute]].append(entry)
        else:
            splitnom[entry[attribute]] = [entry]
    return splitnom

def split_on_numerical(data_set, attribute, splitting_value):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, threshold (splitting) value
    ========================================================================================================
    Job:    Splits data_set into a tuple of two lists, the first list contains the examples where the given
	attribute has value less than the splitting value, the second list contains the other examples
    ========================================================================================================
    Output: Tuple of two lists as described above
    ========================================================================================================
    '''
    splitnumleft = []
    splitnumright = []
    for entry in data_set:
        if entry[attribute] < splitting_value:
            splitnumleft.append(entry)
        else:
            splitnumright.append(entry)
    return (splitnumleft, splitnumright)