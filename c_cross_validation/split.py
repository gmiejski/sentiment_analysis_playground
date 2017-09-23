from sklearn.model_selection import KFold


def splitted_data(data, chunks=5):
    kf = KFold(n_splits=chunks)
    current = 1
    for train_index, test_index in kf.split(data):
        print("Fold: {}".format(current))
        current += 1
        yield [data[index] for index in train_index], [data[index] for index in test_index]
