import streamlit as st
from tensorflow.keras.models import load_model
import preprocess_data as pd

model = load_model("Classifier Model/")

st.set_page_config("FAKE NEWS📰 CLASSIFIER APP")
st.markdown("<h1 style='text-align: center;'>FAKE NEWS CALSSIFIER APP</h1>", unsafe_allow_html=True)
st.image('img.png')
st.write("\n",key=555)

sentence = st.text_input("ENTER YOUR NEWS HEADLINES HERE ",key=111)
st.write("\n",key=333)

if sentence:
    # predict
    preprocessed_text = pd.text_preprocess(sentence)
    st.title(pd.predict(preprocessed_text))