BASE_URL = "Data/data_batch_"

def unpickle(batch_num):
    import pickle
    file = BASE_URL + str(batch_num)
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


