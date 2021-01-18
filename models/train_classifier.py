import sys


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = meta.columns
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
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 60, 70]
    }

    model = GridSearchCV(estimator=pipeline, param_grid=parameters)    

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