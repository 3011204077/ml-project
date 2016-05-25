
def preprocessing(data_set, attribute_metadata):
    pool = []
    j = 0
    for i in range(0, len(data_set)):
        if data_set[i][0] == None:
            pool.append(i)
    for i in pool:
        del data_set[i - j]
        j += 1
    for entry in data_set:
        for attribute in range(1, len(entry)):
            if entry[attribute] == None:
                entry[attribute] = mostval_sametarg(data_set, attribute_metadata, attribute, entry[0])
                
def preprocessing_for_testdata(data_set, attribute_metadata):
    for entry in data_set:
        for attribute in range(1, len(entry)):
            if entry[attribute] == None:
                entry[attribute] = mostval_sametarg(data_set, attribute_metadata, attribute, entry[0])
                
def mostval_sametarg(data_set, attribute_metadata, attribute, targval):
    pool = {}
    sum_val = []
    if attribute_metadata[attribute]['is_nominal'] == True:
        for entry in data_set:
            if entry[attribute] != None and entry[0] == targval:
                if pool.has_key(entry[attribute]):
                    pool[entry[attribute]] += 1
                else:
                    pool[entry[attribute]] = 1
        for key in pool.keys():
            if pool[key] == max(pool.values()):
                return key
    else:
        for entry in data_set:
            if entry[attribute] != None and entry[0] == targval:
                sum_val.append(entry[attribute])
        return sum(sum_val) / len(sum_val)