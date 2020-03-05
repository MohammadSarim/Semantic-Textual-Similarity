# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:32:08 2020

@author: Muhammad Sarim
"""

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('C:/Users/Muhammad Sarim/Downloads/similarity/Text_Similarity_Dataset.csv')
X=dataset.iloc[:,1:2].values

#Importing the libraries for Word2Vec implementation
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

#Load Google's pre-trained Word2vec model.
word2vec_model=gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Muhammad Sarim/Downloads/similarity/GoogleNews-vectors-negative300.bin.gz', binary=True)


#list containing names of words in the vocabulary
index2word_set=set(word2vec_model.index2word)

#function to average all words vectors in a given paragraph
def avg_sentence_vector(words, model, num_feature, index2word_set):

    featureVec= np.zeros((num_feature,), dtype="float64")
    nwords = 0
    
    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])
            
    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    
    return featureVec

#Importing the library
from scipy import spatial

list=[]


for i in range(0,4023):
    
    #get average vector for text 1
    sentence_1 = dataset['text1'][i]
    sentence_1_avg_vector = avg_sentence_vector(sentence_1.split(), model=word2vec_model, num_feature=300,index2word_set=index2word_set)
    sentence_1_avg_vector=sentence_1_avg_vector.reshape(-1,1)
    
    #get average vector for sentence 1
    sentence_2 = dataset['text2'][i]
    sentence_2_avg_vector = avg_sentence_vector(sentence_2.split(), model=word2vec_model, num_feature=300,index2word_set=index2word_set)
    sentence_2_avg_vector=sentence_2_avg_vector.reshape(-1,1)
    
    
    #get cosine similarity between text 1 and text 2
    sen1_sen2_similarity = 1-spatial.distance.cosine(sentence_1_avg_vector,sentence_2_avg_vector)
    
    #Creating list of similarity score
    list.append(sen1_sen2_similarity)
    
   
#Creating csv file for the output( unique id and similarity score )   
output = pd.DataFrame(( list), columns=['Semantic_Similarity'])
output.to_csv('semantic_similarity_score.csv')





