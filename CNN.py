
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Flatten, Merge, Lambda, Masking
from keras.layers import Input
from keras.layers.convolutional import Convolution1D, ZeroPadding1D
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential


def abstract_model(vocab_size, embed_dim, n_filters, n_gram, abs_len, embed_matrix, dense, dense_dim):
    ''' Encoding model(CNN) for abstract
    Inputs:
        vocab_size : size of the vocabulary
        embed_dim : Dimensions of the word embeddings to be uses
        n_filters : number of output filter
        n_gram : width of the convolution filter (which n-gram to use)
        abs_len : length of abstract sentence to use
        embed_matrix : lookup table storing pre-trained word embeddings
        dense : flag to indicate whether or not to use dense layer before merging
        dense_dim : dimensions for dense layer
    '''
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, weights=[embed_matrix], \
                        input_length=abs_len, trainable=False))
    model.add(Convolution1D(n_filters, n_gram, activation='relu'))
    model.add(MaxPooling1D(n_gram))
    model.add(Flatten())
    if dense:
        model.add(Dense(dense_dim))
    return model


def main(vocab_size, embed_dim, n_filters, n_gram, abs_len,wiki_len,embed_matrix, dense, dense_dim, last_dim):
    return abstract_model(vocab_size, embed_dim, n_filters, n_gram, abs_len, embed_matrix, dense, dense_dim)
     

if __name__ == "__main__":
    model = main(10000, 25, 100, 5, 100, 10, np.random.rand(10000,25), False, 100, 25)
    print(model)

