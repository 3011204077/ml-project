from modules.ID3 import *
from modules.parse import *
from modules.pruning import *
from modules.graph import *
from modules.predictions import *
import copy

# DOCUMENATION
# ===========================================
# decision tree driver - takes a dictionary of options and runs the ID3 algorithm.
#   Supports numerical attributes as well as missing attributes. Documentation on the
#   options can be found in README.md

options = {
    'train' : 'data/btrain.csv',
    'validate': 'data/bvalidate.csv',
    'predict': 'data/btest.csv',
    'new': 'data/new.csv',
    'limit_splits_on_numerical': 6,
    'limit_depth': 6,
    'print_tree': True,
    'print_dnf' : True,
    'prune' : 'data/bvalidate.csv',
    'learning_curve' : False,#{
        #'upper_bound' : 0.5,
        #'increment' : 0.05
    #}
}

def decision_tree_driver(train, validate = False, predict = False, new = False, prune = False,
    limit_splits_on_numerical = False, limit_depth = False, print_tree = False,
    print_dnf = False, learning_curve = False):
    
    train_set, attribute_metadata = parse(train, False)
    if limit_splits_on_numerical != False:
        numerical_splits_count = [limit_splits_on_numerical] * len(attribute_metadata)
    else:
        numerical_splits_count = [float("inf")] * len(attribute_metadata)
        
    if limit_depth != False:
        depth = limit_depth
    else:
        depth = float("inf")

    origin_splits_count = copy.deepcopy(numerical_splits_count)
    
    print "###\n#  Training Tree\n###"

    # call the ID3 classification algorithm with the appropriate options
    tree = ID3(train_set, attribute_metadata, numerical_splits_count, depth)
    print 'finish'
    print '\n'

    # call reduced error pruning using the pruning set
    if prune != False:
        print '###\n#  Pruning\n###'
        pruning_set, _ = parse(prune, False)
        temptree = copy.deepcopy(tree)
        temp_origintree = temptree
        origintree = tree
        print 'original acc ' + str(validation_accuracy(tree, pruning_set, attribute_metadata))
        reduced_error_pruning(temptree, temp_origintree, tree, origintree, train_set, pruning_set, attribute_metadata)
        print ''

    # print tree visually
    if print_tree:
        print '###\n#  Decision Tree\n###'
        cursor = open('./output/tree.txt','w+')
        cursor.write(tree.print_tree())
        cursor.close()
        print 'Decision Tree written to /output/tree'
        print ''

    # print tree in disjunctive normalized form
    if print_dnf:
        print '###\n#  Decision Tree as DNF\n###'
        cursor = open('./output/DNF.txt','w+')
        cursor.write(tree.print_dnf_tree())
        cursor.close()
        print 'Decision Tree written to /output/DNF'
        print ''

    # test tree accuracy on validation set
    if validate != False:
        print '###\n#  Validating\n###'
        validate_set, _ = parse(validate, False)
        accuracy = validation_accuracy(tree,validate_set, attribute_metadata)  #add attribute_metadata
        print "Accuracy on validation set: " + str(accuracy)
        print ''

    # generate predictions on the test set
    if predict != False:
        print '###\n#  Generating Predictions on Test Set\n###'
        create_predictions(tree, predict, new)  #add new
        print ''

    # generate a learning curve using the validation set
    if learning_curve and validate:
        print '###\n#  Generating Learning Curve\n###'
        iterations = 3 # number of times to test each size
        get_graph(train_set, attribute_metadata, validate_set, 
            origin_splits_count, depth, 3, 0, learning_curve['upper_bound'],
            learning_curve['increment'])
        print ''

tree = decision_tree_driver( **options )