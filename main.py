import gzip

from data_loading import load
from sentiment_tokenize.tokenizer import Tokenizer, create_dictionary
from sentiment_analyser.analyser import SentimentAnalyser

dataset_file = "data/reviews_Movies_and_TV_5.json.gz"

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


generator = parse(dataset_file)
tokenizer = Tokenizer()

dataset = load.load(generator, 100)
tokenized_data = list(map(lambda entry: (tokenizer.tokenize(entry[0]), entry[1]), dataset))

def c_cross_validation(data):
    return data[:95], data[95:]


tokenized_training_data, test = c_cross_validation(tokenized_data)

words_dictionary = create_dictionary(tokenized_data)

sa = SentimentAnalyser(words_dictionary)
sa.train(tokenized_training_data)

predicted = sa.predict(test)
print(predicted)
print(list(map(lambda x: x[1], test)))
