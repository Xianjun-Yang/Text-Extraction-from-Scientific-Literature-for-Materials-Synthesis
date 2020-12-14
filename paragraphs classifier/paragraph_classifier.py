#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 00:07:44 2020

@author: xianjunyang
"""
import numpy as np
from gensim.models import keyedvectors

from keras import backend as K
from keras.layers import (GRU, Bidirectional, Concatenate, Dense, Dropout,
                          Embedding, Input, Lambda)
from keras.layers.merge import Concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing import sequence
from unidecode import unidecode_expect_nonascii
import re
import spacy

class ParagraphClassifier(object):
  def __init__(self, seq_maxlen=300):
    self._seq_maxlen = seq_maxlen
    self.para_classes = {
      0: 'null',
      1: 'abstract',
      2: 'intro',
      3: 'recipe',
      4: 'nonrecipe_methods',
      5: 'results',
      6: 'conclusions',
      7: 'caption'
    }
    try:
        self.nlp = spacy.load('en')
    except:
        self.nlp = spacy.load('en_core_web_sm')

  def build_nn_model(self, nn_model=None, rnn_size=128):
    self._load_embeddings()

    input_word_ids = Input(shape=(self._seq_maxlen,))
    paragraph_position = Input(shape=(1,))

    emb_matrix =  Embedding(
      input_dim=self.emb_weights.shape[0],
      output_dim=self.emb_weights.shape[1],
      input_length=self._seq_maxlen,
      weights=[self.emb_weights],
      trainable=False,
      mask_zero=True
      )

    emb_word = emb_matrix(input_word_ids)
    drop_1 = Dropout(0.25)(emb_word)
    rnn_1 = Bidirectional(GRU(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=False))(drop_1) 

    merge_1 = Concatenate()([paragraph_position, rnn_1])
    dense_out = Dense(len(self.para_classes), activation="softmax")(merge_1)

    self.model = Model(inputs=[input_word_ids, paragraph_position], outputs=[dense_out])
    self.model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
                  metrics=['accuracy'])

  # featurize is a function to return the embedding value of each word and punctuation of sentence
  # example: featurize('January 10, 2020'), result is: array([   1,   75,    3, 8532])
  # some bugs?:  © 2020 American Chemical Society converted into array([10,34,7,8532,1,1,1]), for unknown word, give its embedding as 1
  def featurize(self, text):
    text = self._normalize_string(text)
    tokens = self.nlp(text)
    emb_vector = []

    for token in tokens:
      tok = token.lemma_
      if tok in self.emb_vocab:
        emb_vector.append(self.emb_vocab[tok])
      else:
        #
        emb_vector.append(1)

    return np.array(emb_vector)

  # loading embedding model first for following use
  # fasttext_embeddings-MINIFIED.model download: https://figshare.com/articles/Pre-trained_FastText_for_materials_science/7441274/1
  def _load_embeddings(self, fpath='/Users/xianjunyang/Desktop/Quantum/paragraphs_classifier/fasttext_pretrained_matsci/fasttext_embeddings-MINIFIED.model'):
    embeddings = keyedvectors.KeyedVectors.load(fpath)
    embeddings.bucket = 2000000
    self.emb_vocab = dict([('<null>', 0), ('<oov>', 1)] +
                          [(k, v.index+2) for k, v in embeddings.vocab.items()])
    self.emb_weights = np.vstack([np.zeros((1,100)), np.ones((1,100)), np.array(embeddings.syn0)])

  def train(self, X_train, Y_train, batch_size=16, num_epochs=20, verbosity=1):
    self.model.fit(
      x=X_train,
      y=Y_train,
      batch_size=batch_size,
      epochs=num_epochs,
      verbose=verbosity
    )

  #def predict_one(self, paragraph_text, paragraph_position, section_text, supsection_text):
  def predict_one(self, paragraph_text, paragraph_position):
    paragraph_feature_vector = self.featurize(paragraph_text) #paragraph_text should be a string
    # do padding to the string of a sentence, thus normalize it
    padded_vec = sequence.pad_sequences([paragraph_feature_vector],
                                        maxlen=self._seq_maxlen, padding='post', truncating='post')
    # 3 inputs for the fast_predict: padded_vec, np.array, 0
    return self.para_classes[np.argmax(self.fast_predict([
      padded_vec, np.array(paragraph_position).reshape(1, -1), 0])[0][0])]
  #reshape(1,-1): convert array into one line array

  #input_matrix does not work, why?
  def predict(self, input_matrix):
    return self.fast_predict(input_matrix + [0])

  def save(self, filename):
    self.model.save(filename)

  #def load(self, filename='bin/paragraph_classifier.model'):
  #paragraph_classifier.model download: https://figshare.com/s/1a07d18ad20008ddd562
  def load(self, filename='/Users/xianjunyang/Desktop/Quantum/paragraphs_classifier/paragraph_classifier.model'):
    self.model = load_model(filename)
    self.fast_predict = K.function(
         self.model.inputs + [K.learning_phase()],
      ## model.inputs: [<tf.Tensor 'input_1:0' shape=(None, 300) dtype=float32>, <tf.Tensor 'input_4:0' shape=(None, 1) dtype=float32>]
      ## K.learning_phase(): <tf.Tensor 'learning_phase:0' shape=() dtype=int32>
         [self.model.layers[-1].output]
      ## model.layers[-1].output: <tf.Tensor 'dense_1/Softmax:0' shape=(None, 8) dtype=float32>
         )

    self._load_embeddings()

  # normalize string, and if the string contains [Α-Ωα-ωÅ], keep it. Otherwise, normalize it by unidecode_expect_nonascii
  # for some strange char such as ©, convert it into (c)
  """
  input: string(a sentence)
  """
  def _normalize_string(self, string):
    ret_string = ''
    for char in string:
      if re.match('[Α-Ωα-ωÅ]', char) is not None:
        ret_string += char
      else:
        ret_string += unidecode_expect_nonascii(char)

    return ' '.join(ret_string.split())





