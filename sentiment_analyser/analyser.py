from scipy.sparse import vstack, csr_matrix
from sklearn.naive_bayes import MultinomialNB


class SentimentAnalyser:
    def __init__(self, words_dictionary):
        self.words_dictionary = words_dictionary
        self.train_X = None
        self.train_y = None
        self.model = None

    def __feature_array(self, entries, words_indexes):
        matrixes = []
        for entry in entries:
            empty = [0] * len(words_indexes)
            for token in entry[0]:
                empty[words_indexes[token]] += 1
            matrix = csr_matrix([empty])
            matrixes.append(matrix)
        return vstack(matrixes), list(map(lambda entry: entry[1], entries))

    def train(self, tokenized_train_data):
        """ train_data : [ ([token, token, token, ...], overall), ...]"""
        self.train_X, self.train_y = self.__feature_array(tokenized_train_data, self.words_dictionary)
        self.model = MultinomialNB()
        self.model.fit(self.train_X, self.train_y)

    def predict(self, test_data):
        test_X, test_y = self.__feature_array(test_data, self.words_dictionary)
        return self.model.predict(test_X)
