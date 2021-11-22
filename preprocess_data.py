# Import Libraries
import nltk
import tensorflow as tf
import re
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


from app import sentence

def text_preprocess(text):

   #removing the special characters
    news = re.sub('[^a-zA-Z]',' ',text)
    news = news.lower()
    news = news.split()

    # Stemming + removing stopwords
    ps = PorterStemmer()
    news = [ps.stem(word) for word in news if not word in stopwords.words("english")]
    news = ' '.join(news)
    
    # one hot representation
    voc_size=5000
    one_hot_rep = np.array(one_hot(news,voc_size))
    one_hot_rep=one_hot_rep.reshape(1,one_hot_rep.shape[0])

    # GIVING PADDING TO THE ONE-HOT-ENCODED VALUES
    max_sent_len = 40
    embedded_docs = pad_sequences(one_hot_rep,padding='pre',maxlen=max_sent_len)
    
    # Input to model
    X = np.array(embedded_docs)
    
    return X

def predict(preprocessed_text,model=None):
    model = load_model("Classifier Model/")
    predictions = model.predict(preprocessed_text)[0][0]
    print(predictions)
    if predictions>=0.5:
        prediction = "THIS NEWS IS FAKE !!!"
        print("THIS NEWS IS FAKE !!!")
    else:
        prediction = "THIS NEWS IS ABSOLUTELY TRUE ❤️"
        print("THIS NEWS IS ABSOLUTELY TRUE ❤️")

    return prediction

    