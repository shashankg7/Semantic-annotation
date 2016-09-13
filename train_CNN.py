#D1 : Evaluate with papers dataset
import sys
from gensim.models import Doc2Vec
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import STOPWORDS
from gensim import utils
#from read_wiki import stream
from representation import lda, lsi, doc2vec, doc2vec_model
from CNN import abstract_model
import pdb
import json
import re
import random
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression as lr
from keras.preprocessing.sequence import pad_sequences
from string import punctuation
#read in papers dataset
# Arg1 is path to CS_Citation_Network Arg2 is type of representation to be used

file=sys.argv[1];
rep = sys.argv[2]
embed_dim = int(sys.argv[3])

wiki_path = './enwiki-latest-pages-articles.xml.bz2'


def tokenize(text):
	text = text.lower()
	filtered = re.sub(r'[^\w\s]', " ", text)
	filtered = re.sub('\d', ' DIGIT ', filtered)
	tokens = filtered.split()
	tokens = [token for token in tokens if token not in STOPWORDS]
	return tokens


x = "Data mining, is a technique!!! For learning patterns@"
print(x, tokenize(x))
pdb.set_trace()

#Read the keyword and abstract
uniqueFields = {}
keyAbs=[]
with open(file) as infile:
	field = None;
	abstract = None;
	for line in infile:
		#if len(line)==1 :
		if line in ['\n', '\r\n']:
			if abstract != None:
				keyAbs.append([field,abstract])
				#print abstract
				#print field
			field = None
			abstract = None
		else:
			#if line[1]=='*' : # field is title
				#field = line.replace("#*","")
			if line[1]=='f' : #field is keyword
				field = line.replace("#f","")
				field = field.replace("\n","")
				field = field.replace("\r","")
				field = field.replace("_"," ")
				field = field.capitalize()
				#print field
				uniqueFields[field]=0
			if line[1]=='!' :
				abstract = line.replace("#!","")
print 'Citations loaded = '+str(len(keyAbs))


training_samples = []
test = []
train_abstracts = keyAbs[:int(0.8 *len(keyAbs))]
test_abstracts = keyAbs[int(0.8 * len(keyAbs)):]
vocab = {}
#tokens = []
idx = 0
# generate vocab for abstracts
for abstract in train_abstracts:
	title, text = abstract[0], abstract[1]
	text = tokenize(text)
	#tokens.extend(text)
	for token in text:
		if token not in vocab:
			vocab[token] = idx
			idx += 1

vocab["UNK"] = idx
# (OPTIONAL) Filter voacb based on frequency
#  				###################
#			          fitlering code
# 				##################

# Create embedding matrix table 
# initializing embedding matrix with values randomly sampled from [-0.25, 0.25]
i = 0
embedding = np.random.uniform(low=-0.25, high=0.25, size=(len(vocab), embed_dim))
for token, idx in vocab.items():
	try:
		embedding[idx, :] = doc2vec_model[token]
	except Exception as e:
		i += 1
		#print(str(e))
		
print("no of words missing from vocab %d"%i)
# Generate training sequence for CNN
X_train = []
y_train = []
Y_train = []
X_test = []
y_test = []

f_wiki = open('wikiCatArticles.json', 'r')
wiki_data = json.load(f_wiki)
keys = wiki_data.keys()
key2int = dict(zip(keys, range(len(keys))))
avg_len = []

#pdb.set_trace()
#pdb.set_trace()
print("Preparing training data")
for abstract in train_abstracts:
	temp = []
	title, text = abstract[0], abstract[1]
	if title in keys:
		text = tokenize(text)
		avg_len.append(len(text))
		for token in text:
			temp.append(vocab[token])
		X_train.append(temp)
		y_train.append(title)		

avg_len = int(float(sum(avg_len))/len(avg_len))

#pdb.set_trace()
print("Training data done, preparing test data")

for abstract in test_abstracts:
	temp = []
	title, text = abstract[0], abstract[1]
	text = tokenize(text)
	if title in keys:
		for token in text:
			temp.append(vocab.get(token, vocab["UNK"]))
		X_test.append(temp)
		y_test.append(key2int[title])


X_train = pad_sequences(X_train, maxlen=avg_len)
Y_train = np.zeros((len(X_train), len(key2int)))
X_test = pad_sequences(X_test, maxlen=avg_len)
for ind, label in enumerate(y_train):
	Y_train[ind, key2int[label]] = 1


pdb.set_trace()
X = np.hstack((X_train, Y_train))
np.random.shuffle(X)
X_train = X[:, :avg_len]
Y_train = X[:, avg_len:]

print("data prep done")
pdb.set_trace()



