import gzip
from sentiment_tokenize.tokenizer import Tokenizer

dataset_file = "data/reviews_Movies_and_TV_5.json.gz"


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


generator = parse(dataset_file)
tokenizer = Tokenizer()

for a in [x for _, x in zip(range(10), generator)]:
    # print(a)
    text = a['reviewText']
    print(text)
    s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
    print(tokenizer.tokenize(text))

print()
print()
print()

from sklearn.naive_bayes import MultinomialNB

sentences = ['Ala ma kota.',
             'Dominik ma psa',
             'Kot nie jest taki zły',
             'Ala nie może palić'
             ]


def feature_array(sentence, tokenizer, words_indexes):
    tokens = tokenizer.tokenize(sentence)
    for token in tokens:
        if token not in words_indexes.keys():
            words_indexes[token] = len(words_indexes)

    features = [0] * len(words_indexes)
    for token in tokens:
        features[words_indexes[token]] = features[words_indexes[token]] + 1

    return features


def makeMatrix(sentences):
    tokenizer = Tokenizer()
    words_indexes = {}
    result = []

    for x in sentences:
        features = feature_array(x, tokenizer, words_indexes)
        result.append(features)

    for feature_list in result:
        for empty in range((len(words_indexes) - len(feature_list))):
            feature_list.append(0)

    return result, words_indexes


X, words_indexes = makeMatrix(sentences)
test_sentence = "Ale Ala nie jest taka głupia."
test_X = feature_array(test_sentence, tokenizer, words_indexes)

classes = [1, 0, 0, 1]

clf = MultinomialNB()



def align_features(X, words_indexes):
    for feature_list in X:
        for empty in range((len(words_indexes) - len(feature_list))):
            feature_list.append(0)
    return X


X = align_features(X, words_indexes)

clf.fit(X, classes)
y = clf.predict([test_X, test_X])
print(y)