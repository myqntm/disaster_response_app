from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
from pprint import pprint
import re
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
import time
import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse_Table', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns.tolist()
    return X, y, category_names   


def tokenize(text):
    # Tokenize the data
    tokens = word_tokenize(text)
    # Normalize words
    tokens = [token.lower() for token in tokens]
    # Remove stop words
    stop_words = set(stopwords.words('english'))  
    tokens = [w for w in tokens if not w in stop_words]  
    # Lemmatize words
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100]
        } 
    model = GridSearchCV(pipeline, param_grid=parameters)    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    # predict
    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))



def save_model(model, model_filepath):
    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
        
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


# X, Y, category_names = load_data('data/DisasterResponse.db')
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
# model = build_model()
# model.fit(X_train, Y_train)