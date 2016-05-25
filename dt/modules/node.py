# DOCUMENTATION
# =====================================
# Class node attributes:
# ----------------------------
# children - a list of 2 nodes if numeric, and a dictionary (key=attribute value, value=node) if nominal.  
#            For numeric, the 0 index holds examples < the splitting_value, the 
#            index holds examples >= the splitting value
#
# label - is None if there is a decision attribute, and is the output label if there are no other attributes
#       to split on or the data is homogenous
#
# decision_attribute - the index of the decision attribute being split on
#
# is_nominal - is the decision attribute nominal
#
# value - Ignore (not used, output class if any goes in label)
#
# splitting_value - if numeric, where to split
#
# name - name of the attribute being split on

class Node:
    def __init__(self):
        # initialize all attributes
        self.label = None
        self.decision_attribute = None
        self.is_nominal = None
        self.value = None
        self.splitting_value = None
        self.children = {}
        self.name = None

    def classify(self, instance):
        '''
        given a single observation, will return the output of the tree
        '''
        
        if self.label == 1:
            return 1
        if self.label == 0:
            return 0
        if self.label == None:
            if self.is_nominal == True:
                difference = {}
                for splitval in self.children.keys():
                    if splitval == instance[self.decision_attribute]:
                        return self.children[splitval].classify(instance)
                    else:
                        difference[abs(instance[self.decision_attribute] - splitval)] = splitval
                return self.children[difference[min(difference.keys())]].classify(instance)
            else:
                if instance[self.decision_attribute] < self.splitting_value:
                    return self.children[0].classify(instance)
                else:
                    return self.children[1].classify(instance)

    def print_tree(self, indent = 0):
        '''
        returns a string of the entire tree in human readable form
        '''
        output = []
        print_text = ''
        self.gen_tree(indent, output)
        for entry in output:
            print_text += entry
        return print_text

    def gen_tree(self, indent, output):
        if self.is_nominal == True:
            for splitval in self.children.keys():
                if self.children[splitval].label == None:
                    output.append(' ' * indent * 2 + self.name + ' = ' + str(splitval) + '\n')
                    print ' ' * indent * 2 + self.name + ' = ' + str(splitval)
                    self.children[splitval].gen_tree(indent + 1, output)
                else:
                    output.append(' ' * indent * 2 + self.name + ' = ' + str(splitval) + ' : ' + str(self.children[splitval].label) + '\n')
                    print ' ' * indent * 2 + self.name + ' = ' + str(splitval) + ' : ' + str(self.children[splitval].label)
        if self.is_nominal == False:
            if self.children[0].label == None:
                output.append(' ' * indent * 2 + self.name + ' < ' + str(self.splitting_value) + '\n')
                print ' ' * indent * 2 + self.name + ' < ' + str(self.splitting_value)
                self.children[0].gen_tree(indent + 1, output)
            else:
                output.append(' ' * indent * 2 + self.name + ' < ' + str(self.splitting_value) + ' : ' + str(self.children[0].label) + '\n')
                print ' ' * indent * 2 + self.name + ' < ' + str(self.splitting_value) + ' : ' + str(self.children[0].label)
            if self.children[1].label == None:
                output.append(' ' * indent * 2 + self.name + ' >' + '= ' + str(self.splitting_value) + '\n')
                print ' ' * indent * 2 + self.name + ' >' + '= ' + str(self.splitting_value)
                self.children[1].gen_tree(indent + 1, output)
            else:
                output.append(' ' * indent * 2 + self.name + ' >' + '= ' + str(self.splitting_value) + ' : ' + str(self.children[1].label) + '\n')
                print ' ' * indent * 2 + self.name + ' >' + '= ' + str(self.splitting_value) + ' : ' + str(self.children[1].label)
        
    def print_dnf_tree(self):
        '''
        returns the disjunct normalized form of the tree.
        '''
        result = ''
        dnf = []
        total_dnf = ''
        self.gen_dnf(result, dnf)
        for entry in dnf:
            entry = entry[:-4] + ')'
            total_dnf += entry + ' v '
        print total_dnf[:-3]
        return total_dnf[:-3]
        
    def gen_dnf(self, result, dnf):
        if self.label == 1:
            dnf.append('(' + result + ')' )
        if self.is_nominal == True:
            for splitval in self.children.keys():
                result += self.name + '=' + str(splitval) + ' ^ '
                self.children[splitval].gen_dnf(result, dnf)
        if self.is_nominal == False:
            temp = result
            result += self.name + '<' + str(self.splitting_value) + ' ^ '
            self.children[0].gen_dnf(result, dnf)
            result = temp   
            result += self.name + '>' + '=' + str(self.splitting_value) + ' ^ '
            self.children[1].gen_dnf(result, dnf)