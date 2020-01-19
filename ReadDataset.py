import pandas as pd

def readDataset(filename):
    # Reading the dataset
    
    df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
    intent = df["Intent"]
    unique_intent = list(set(intent))
    sentences = list(df["Sentence"])
  
    return (intent, unique_intent, sentences)

