import re
import numpy as np

from keras.models import load_model
from nltk.tokenize import word_tokenize

from ReadDataset import readDataset
from CleaningSentences import cleaningSentences
from Tokenizer import tokenizer
from MaxLength import maxLength
from Padding import paddindDoc
from GetSetOfIntents import getSetOfIntents

def predictions(text,wordTokenizer,length):
    # Model loading
    model = load_model("model.h5")

    # Processing the new text
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
    test_word = word_tokenize(clean)
    test_word = [w.lower() for w in test_word]
    test_ls = wordTokenizer.texts_to_sequences(test_word)
    print(test_word)
    #Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls))
    
    test_ls = np.array(test_ls).reshape(1, len(test_ls))
 
    x = paddindDoc(test_ls, length)
  
    pred = model.predict_proba(x)
  
    return pred


def main():
    # Attributes
    filename = 'Dataset.csv'

    # Reading the dataset
    intents, uniqueIntents, sentences = readDataset(filename)

    # Cleaning the sentences and tokenizing
    cleanedWords = cleaningSentences(sentences)
    
    # Indexing 
    wordTokenizer = tokenizer(cleanedWords)

    # Maximum set of words
    length = maxLength(cleanedWords)

    # Prediction
    text = 'What to do if my business category is not in the options?'
    pred = predictions(text,wordTokenizer,length)

    # Getting final output
    getSetOfIntents(pred,uniqueIntents)


if __name__ == "__main__":
    main()