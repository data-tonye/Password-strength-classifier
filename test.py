from pandas.io.parsers import read_csv
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from urllib.error import URLError


def load_data():
    data = "https://drive.google.com/uc?export=download&id=19XHbRlvcq2svrpUobxbtnIK8_pF95Ck9"
    df = pd.read_csv(data, error_bad_lines = False)
    df.dropna(inplace=True)
    return df

try:
    st.write("""
    # Password Strength Classifier project
    """
    )
    df = load_data()
    password_tuple = np.array(df)
    random.shuffle(password_tuple)
    x = [labels[0] for labels in password_tuple]
    y = [labels[1] for labels in password_tuple]

    def split_word(inputs):
        character = []
        for i in inputs:
            character.append(i)
        return character

    vectorizer = TfidfVectorizer(tokenizer=split_word)
    
    X = vectorizer.fit_transform(x)
    first_document = X[0]
    first_document.T.todense()
    feature_names = vectorizer.get_feature_names()
    df2 = pd.DataFrame(first_document.T.todense(), index = feature_names, columns = ['TF-IDF'])

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    clf = LogisticRegression(random_state = 0, multi_class= 'multinomial')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    with st.form(key='my_form'):
        text_input = st.text_input(label='Enter a Password')
        if text_input:
            dt = np.array([text_input])
        
        submit_button = st.form_submit_button(label='Check')
        if submit_button:
            pred = vectorizer.transform(dt)
            result = clf.predict(pred)
            if result[0] == 0:
                st.write('Weak password!')
            elif result[0] == 1:
                st.write('Average password!')
            else:
                st.write('Strong password!')
except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )
