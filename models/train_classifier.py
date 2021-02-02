import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report




def load_data(database_filepath):
    '''
    Function to that loads the database
    Input: sql database
    Output: messages, categories and category names
    '''
    # Create sql engine using database
    engine = create_engine('sqlite:///' + database_filepath)
    # Read processed data from table
    df = pd.read_sql_table('DisasterResponse_Table', engine)
    # Load messages column to X value
    X = df['message']
    # Load categories to y value
    y = df.iloc[:,4:]
    # Save category names
    category_names = y.columns.tolist()
    return X, y, category_names   


def tokenize(text):
    '''
    Function that normalizes, removes stop words and tokenizes text
    Input: text content
    Output: tokenize text
    '''
    # Tokenize the text
    tokens = word_tokenize(text)
    # Normalize the text to lowercase
    tokens = [token.lower() for token in tokens]
    # Load English stopwords
    stop_words = set(stopwords.words('english'))  
    # Remove stopwords
    tokens = [w for w in tokens if not w in stop_words]  
    # Lemmatize text
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    return tokens


def build_model():
    '''
    Function that builds the model
    Input: none
    Output: machine learning model
    '''
    # Create ML pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    # Load parameters (tfidf)
    parameters = {'tfidf__use_idf': (True, False), 'clf__estimator__n_estimators': [50, 60, 70]
    }
    # Perform gridsearch
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function that evaluates the model
    Input: model, X_test, Y_test, category_names
    Output: model classification and accuracy
    '''
    # Make predictions
    y_pred = model.predict(X_test)
    # Test for each combination
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i = i + 1
    # Determine accuracy
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    '''
    Function that saves the model to a pickle file
    Input: model, model filepath
    Output: save model to pickle file
    '''
    # Save to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


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



