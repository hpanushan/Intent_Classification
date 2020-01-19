import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk

from ReadDataset import readDataset
from CleaningSentences import cleaningSentences
from Tokenizer import tokenizer
from MaxLength import maxLength
from EncodingListOfSentences import encodingDoc
from Padding import paddindDoc
from OneHotEncode import oneHotEncoder
from CreateModel import createModel

def main():
    # Attributes
    filename = 'Dataset.csv'

    # Reading the dataset
    intents, uniqueIntents, sentences = readDataset(filename)

    # Cleaning the sentences and tokenizing
    cleanedWords = cleaningSentences(sentences)
    
    # Indexing 
    wordTokenizer = tokenizer(cleanedWords)
    vocabSize = len(wordTokenizer.word_index) + 1
    length = maxLength(cleanedWords)

    # Encoding the sentences
    encodedDoc = encodingDoc(wordTokenizer, cleanedWords)

    # Making equal length
    paddedDoc = paddindDoc(encodedDoc, length)

    # For intents
    # Tokenizer with filter changed
    outputTokenizer = tokenizer(uniqueIntents, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')

    # Encodeing the intents with unique intents
    encodedOutput = encodingDoc(outputTokenizer, intents)

    # Creating a array for each intent
    encodedOutput = np.array(encodedOutput).reshape(len(encodedOutput), 1)
    
    # One hot encoding (This creates 2D array with columns=Unique intents, rows=intents)
    outputOneHot = oneHotEncoder(encodedOutput)

    # Now dataset cleaning is finished!!!!
    # Splitting the dataset
    trainX, valX, trainY, valY = train_test_split(paddedDoc, outputOneHot, shuffle = True, test_size = 0.2)

    # Model creation
    model = createModel(vocabSize, length)
    
    # Checking model (Layers)
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    model.summary()

    # Start model training
    filename = 'model.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    hist = model.fit(trainX, trainY, epochs = 200, batch_size = 16, validation_data = (valX, valY), callbacks = [checkpoint])



if __name__ == "__main__":
    main()