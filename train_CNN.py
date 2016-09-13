#D1 : Evaluate with papers dataset
import sys
from gensim.models import Doc2Vec
from gensim.corpora.dictionary import Dictionary
from gensim import utils
#from read_wiki import stream
from representation import lda, lsi, doc2vec
import pdb
import json
import re
import random
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression as lr
from string import punctuation
#read in papers dataset
# Arg1 is path to CS_Citation_Network Arg2 is type of representation to be used

file=sys.argv[1];
rep = sys.argv[2]
wiki_path = './enwiki-latest-pages-articles.xml.bz2'


def tokenize(text):
	text = text.lower()
	filtered = re.sub(r'[^\w\s]', " ", text)
	filtered = re.sub('\d', ' DIGIT ', filtered)
	return filtered.split()

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
tokens = []
idx = 0
# generate vocab for abstracts
for abstract in train_abstracts:
	title, text = abstract[0], abstract[1]
	text = tokenize(text)
	tokens.extend(text)
	for token in text:
		if token not in vocab:
			vocab[token] = idx
			idx += 1



pdb.set_trace()



