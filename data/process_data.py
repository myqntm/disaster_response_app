import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np



def load_data(messages_filepath, categories_filepath):
    '''
    Function to load the data
    Input: messages file path, categories file path
    Output: Returns the merged dataframe
    '''
    # Load messages data set from csv
    messages = pd.read_csv(messages_filepath)
    # Load categories data set from csv
    categories = pd.read_csv(categories_filepath)
    # Combine messages and categories
    df = pd.merge(messages,categories,on='id')
    return df 
 

def clean_data(df):
    '''
    Function to clean the data
    Input: merged dataframe
    Output: Returns a cleaned dataframe with categories split into columns
    '''
    # Split categories into separate category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # Grab first row
    row = categories.iloc[[1]]
    # Extract category name from values
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    # Apply category name to column names
    categories.columns = category_colnames
    # Convert category values to number
    for column in categories:
    # Extract last character of the string
        categories[column] = categories[column].str[-1]
    # Change type string to numeric
        categories[column] = categories[column].astype(np.int)
    # Drop old category olumn
    df = df.drop(['categories'], axis=1)
    # Concatenate df with new split categories
    df = pd.concat([df,categories],axis=1)
    return df

def save_data(df, database_filename):
    '''
    Function to clean the data
    Input: merged dataframe
    Output: Returns a cleaned dataframe with categories split into columns
    '''
    # Create sql engine with database
    engine = create_engine('sqlite:///'+ database_filename)
    # Save dataframe to database
    df.to_sql('DisasterResponse_Table', engine, index=False, if_exists='replace')
  
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()