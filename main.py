import gzip
from sentiment_tokenize.tokenizer import Tokenizer
from data_loading import load
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, vstack

dataset_file = "data/reviews_Movies_and_TV_5.json.gz"


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


generator = parse(dataset_file)
tokenizer = Tokenizer()

dataset = load.load(generator, 100)
dataset = list(map(lambda entry: (tokenizer.tokenize(entry[0]), entry[1]), dataset))


def create_dictionary(dataset):
    words = set()
    for entry in dataset:
        for token in entry[0]:
            words.add(token)
    return dict(zip(words, range(len(words))))


words_dictionary = create_dictionary(dataset)

from sklearn.naive_bayes import MultinomialNB

sentences = ['Ala ma kota.',
             'Dominik ma psa',
             'Kot nie jest taki zły',
             'Ala nie może palić',
             "Ale Ala nie jest taka głupia."
             ]
classes = [1, 0, 0, 1, 1]

data = list(map(lambda x: (x[1], classes[x[0]]), zip(range(len(sentences)), sentences)))
words_dictionary = create_dictionary(data)


def c_cross_validation(data):
    return data[:4], data[-1]


training, test = c_cross_validation(data)


def feature_array(entries, words_indexes):
    matrixes = []
    for entry in entries:
        empty = [0] * len(words_indexes)
        for token in entry[0]:
            empty[words_indexes[token]] += 1
        matrix = csr_matrix([empty])
        matrixes.append(matrix)
    return vstack(matrixes)


train_X = feature_array(training, words_dictionary)
test_X = feature_array([test], words_dictionary)

clf = MultinomialNB()

clf.fit(train_X, classes[:len(training)])
y = clf.predict(test_X)
print(y)
