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
            all_words = word_tokenize(l)
            lexicon += list(all_words)
    with open(neg, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)
    lexicon = [lemmi.lemmatize(i) for i in lexicon]
    w_count = Counter(lexicon)
    #print(w_count)
    l2 = []
    for w in w_count:
        #print(w_count[w])
        if 1000 > w_count[w] > 50:
            l2.append(w)
    #print(w)
    return l2
