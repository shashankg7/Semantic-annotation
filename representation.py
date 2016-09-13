# Collection of models for inferring different representations (lda, lsi etc.) for a given document

import gensim
import numpy as np
from gensim import corpora, utils, similarities
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.doc2vec import Doc2Vec
from read_wiki import stream
import pdb

dir_path = './'
doc2vec_path = dir_path + 'data/' + 'wiki_model2.doc2vec'
doc2vec_model = Doc2Vec.load(doc2vec_path)
	

def tokenize(text):
	return [token for token in utils.simple_preprocess(text) if token not in STOPWORDS]


def lda(text):
	dict_path = dir_path + 'cs_lda6.dict'
	lda_path = dir_path + 'wiki_model6.ldamodel'
	dictionary = corpora.Dictionary.load(dict_path)
	model = gensim.models.ldamodel.LdaModel.load(lda_path)
	print(model)
	nvect = dictionary.doc2bow(tokenize(text))
	print(nvect)
	#print(model[nvect])
	return model[nvect]


def lsi(text):
	pass


def doc2vec(text):
	return doc2vec_model.infer_vector(text.split(), alpha=0.1, min_alpha=0.0001, steps=5)


if __name__ == "__main__":
	test = "Social media mining is the process of representing, analyzing , and extracting actionable patterns from social media data. Social media mining introduces basic concepts and principal algorithms suitable for investing massive social media data; it discusses theories and methodologies from different disciplines such as computer science, data mining, machine learning, social network analysis, optimization and mathematics. It encompasses the tools to formaly represent, measure, model and mine meaningful patterns from large-scale data"
	print(doc2vec(test))
