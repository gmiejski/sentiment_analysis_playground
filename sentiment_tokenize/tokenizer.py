from nltk.tokenize import TweetTokenizer


class Tokenizer:
    def __init__(self):
        self.tknzr = TweetTokenizer()

    def tokenize(self, sentence):
        return self.tknzr.tokenize(sentence)


def create_dictionary(dataset):
    words = set()
    for entry in dataset:
        for token in entry[0]:
            words.add(token)
    return dict(zip(words, range(len(words))))
