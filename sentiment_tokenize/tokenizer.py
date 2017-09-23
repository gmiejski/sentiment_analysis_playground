
from nltk.tokenize import TweetTokenizer


class Tokenizer():

    def __init__(self):
        self.tknzr = TweetTokenizer()


    def tokenize(self, sentence):

        return self.tknzr.tokenize(sentence)