
from sentiment_tokenize.tokenizer import Tokenizer, create_dictionary
from sentiment_analyser.analyser import SentimentAnalyser


sentences = ['Ala ma kota.',
             'Dominik ma psa',
             'Kot nie jest taki zły',
             'Ala nie może palić',
             "Ale Ala nie jest taka głupia."
             ]
classes = [1, 0, 0, 1, 1]

data = list(map(lambda x: (x[1], classes[x[0]]), zip(range(len(sentences)), sentences)))

tokenizer = Tokenizer()
tokenized_data = list(map(lambda entry: (tokenizer.tokenize(entry[0]), entry[1]), data))

def c_cross_validation(data):
    return data[:4], [data[-1]]

tokenized_training_data, test = c_cross_validation(tokenized_data)

words_dictionary = create_dictionary(tokenized_data)

sa = SentimentAnalyser(words_dictionary)
sa.train(tokenized_training_data)

predicted = sa.predict(test)
print(predicted)