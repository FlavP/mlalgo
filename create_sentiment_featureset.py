import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmi = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(pos, neg):
    lexicon = []
    with open(pos, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l.lower())
            lexicon += list(all_words)
    with open(neg, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l.lower())
            lexicon += list(all_words)
    lexicon = [lemmi.lemmatize(i) for i in lexicon]
    w_count = Counter(lexicon)
    #print(w_count)
    l2 = []
    for w in w_count:
        #print(w_count[w])
        if 1000 > w_count[w] > 50:
            l2.append(w)
    print(len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmi.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def feature_sets_and_labels(pos, neg, step_size = 0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1, 0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    random.shuffle(features)
    features = np.array(features)
    # the questin is does tf.argmax['expectations'] == tf.argmax['']
    testing_size = int(len(features) * step_size)
    #so this means that featureset is an array of lists [[[features], [label]], [[features], [label]]] that looks like this [[[0, 1, 0, 1, 1], [0, 1]]]
    # and we want all the 0th element, meaning the first from each group and we take this until the last 10% of the testing
    #size
    train_x = list(features[:,0][:-testing_size])
    #train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    return  train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = feature_sets_and_labels("pos.txt", "neg.txt")
    with open("sentiment_set.pickle", "wb") as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)