import nltk
import os
import sys


punct_list = "./!:;@#$%^&*()"

def punct_features(tokens,i):
	return {'next-word-capitalized':tokens[i+1][0].isupper(),
			'prevword': tokens[i-1].lower(),
			'punct':tokens[i],
			'prev-word-is-one-char':len(tokens[i-1])==1,
			'is_punct':(tokens[i] in punct_list)}

def segment_sentences(words):
	start = 0
	sents = []
	for i, word in words:
		if word in '.?!' and classifier.classify(words, i ) == True:
			sents.append(words[start:i+1])
			start = i+1
		if start < len(words):
			sents.append(start:)

sents = nlltk.corpus.treebank_raw_sents()
tokens = []
boundaries = set()
offset = 0
for sent in sents:
	tokens.extend(sent)
	offset += len(sent)
	boundaries.add(offset-1)

fs = [(punct_features(tokens,i), (i in boundaries))
		for i in range(1, len(tokens)-1)
		if tokens[i] in ['.','?','!',';',':','and','now','after','before','then']]


size = int(len(fs)*0.1)
train_set, test_set = fs[size:], fs[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)



