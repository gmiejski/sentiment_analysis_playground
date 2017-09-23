import gzip

from data_loading import load
from sentiment_tokenize.tokenizer import Tokenizer, create_dictionary
from sentiment_analyser.analyser import SentimentAnalyser
from c_cross_validation.split import splitted_data

dataset_file = "data/reviews_Movies_and_TV_5.json.gz"


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


generator = parse(dataset_file)
tokenizer = Tokenizer()

dataset = load.load(generator, 1000)
tokenized_data = list(map(lambda entry: (tokenizer.tokenize(entry[0]), entry[1]), dataset))

splitted_data(tokenized_data)
words_dictionary = create_dictionary(tokenized_data)

folds_results = []

for cross in splitted_data(tokenized_data):
    tokenized_training_data = cross[0]
    test = cross[1]
    sa = SentimentAnalyser(words_dictionary)
    sa.train(tokenized_training_data)
    predicted = sa.predict(test)
    original = list(map(lambda x: x[1], test))
    test_fild_size = len(test)
    correct_predictions = len(
        list(filter(lambda x: x == True, [predicted[index] == original[index] for index in range(0, test_fild_size)])))
    folds_results.append((correct_predictions, test_fild_size))


def calculate_accuracy(folds_results):
    folds_acc = [(x[0] / x[1] * 1.0) for x in folds_results]
    return sum(folds_acc) / len(folds_acc)

print(calculate_accuracy(folds_results))
