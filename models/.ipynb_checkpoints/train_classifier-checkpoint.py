import sys
import re
import pandas as pd
import numpy as np
import nltk
import pickle
import sqlite3 as sq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

def partition_msg_cat(df):
    messages = df.message
    categories = df[df.columns[3:]]
    
    return messages,categories

def load_data(database_filepath):
    db = sq.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM messages_categories",db)
    X, Y = partition_msg_cat(df)
    categories = Y.columns
    return X,Y,categories

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9]'," ",text)
    
    tokens = word_tokenize(text.lower())
    
    stem_tk = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return stem_tk

def build_model():
    vect = CountVectorizer(tokenizer = tokenize)
    tfidf = TfidfTransformer(smooth_idf=False)
    claf = RandomForestClassifier()
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer(smooth_idf=False)),
        ('claf', RandomForestClassifier())
        ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    accuracy = ((Y_test == Y_pred).mean()).mean()
    print('Accuracy of Model is : {}'.format(accuracy))

def save_model(model, model_filepath):
    try:
        pickle.dump(model,open(model_filepath,'wb'))
    except:
        print("Failed dumping")


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()