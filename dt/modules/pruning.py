from node import Node
from ID3 import *
from operator import xor
import copy
from modules.parse import *

def reduced_error_pruning(temproot, temp_originroot, root, originroot, training_set, validation_set, attribute_metadata):
    if temproot.is_nominal == True:
        subset = split_on_nominal(training_set, temproot.decision_attribute)
        #return
        for splitval in temproot.children.keys():
            if temproot.children[splitval].label == None:
                newnode = Node()
                newnode.label = mode(subset[splitval])
                newnode.children = {}
                tempchild = copy.deepcopy(temproot.children[splitval])
                temproot.children[splitval] = newnode
                prune_acc = validation_accuracy(temp_originroot, validation_set, attribute_metadata)
                acc = validation_accuracy(originroot, validation_set, attribute_metadata)
                if prune_acc >= acc:
                    print prune_acc
                    root.children[splitval] = newnode
                else:
                    temproot.children[splitval] = tempchild
                    reduced_error_pruning(temproot.children[splitval], temp_originroot, root.children[splitval], originroot, subset[splitval], validation_set, attribute_metadata)
        #return
    if temproot.is_nominal == False:
        subset = split_on_numerical(training_set, root.decision_attribute, root.splitting_value)
        for i in range(0, 2):
            if temproot.children[i].label == None:
                newnode = Node()
                newnode.label = mode(subset[i])
                newnode.children = {}
                tempchild = copy.deepcopy(temproot.children[i])
                temproot.children[i] = newnode
                prune_acc = validation_accuracy(temp_originroot, validation_set, attribute_metadata)
                acc = validation_accuracy(originroot, validation_set, attribute_metadata)
                if prune_acc >= acc:
                    print prune_acc
                    root.children[i] = newnode
                else:
                    temproot.children[i] = tempchild
                    reduced_error_pruning(temproot.children[i], temp_originroot, root.children[i], originroot, subset[i], validation_set, attribute_metadata) 
        #return
    pass

def validation_accuracy(tree, validate_set, attribute_metadata):
    '''
    takes a tree and a validation set and returns the accuracy of the set on the given tree
    '''
    accuracy = 0
    i = 0
    j = 0
    preprocessing_for_testdata(validate_set, attribute_metadata)
    for entry in validate_set:
        if entry[0] != None:
            if entry[0] == tree.classify(entry):
                accuracy += 1
            i += 1
    return float(accuracy) / i * 100
    pass
