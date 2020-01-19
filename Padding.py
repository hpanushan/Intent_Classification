from keras.preprocessing.sequence import pad_sequences

def paddindDoc(encoded_doc, max_length):
    # Make equal length
    return (pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))