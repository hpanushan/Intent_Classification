from sklearn.preprocessing import OneHotEncoder

def oneHotEncoder(encode):
    o = OneHotEncoder(sparse = False)
    return (o.fit_transform(encode))