
import re
from nltk.tokenize import word_tokenize


def cleaningSentences(sentences):
    # Removing punctuation and special characters then tokenized sentences into words
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
        w = word_tokenize(clean)
        # Lowercasing 
        words.append([i.lower() for i in w])
    
    return words  

