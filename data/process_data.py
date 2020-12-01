import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """
    Read csv files of two dataframes and returns a concatenated dataframe
    of both the datasets
    
    Input: messages_filepath: path of messages('disaster_messages.csv')
        `` categories_filepath: path of categories('disaster_categories.csv')

    Output: df: concatenated dataframe object of the two datasets, messages and categories
    """

    # read data from csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge both the datasets and return the same
    df = pd.merge(messages,categories,how = 'left', on='id')
    return df


def clean_data(df):

    """
    Cleans the dataframe by splitting the category names from categories column
    and drops the duplicates, if any.

    Input: df: dataframe from load_data()
    Output: df: clean dataframe df

    """
    
    # split the categories column to extract the category names
    categories = df['categories'].str.split(';',expand=True)
    rows = categories.iloc[0]
    columns = rows.apply(lambda x:x[:-2])

    # set each value to be the last character of the string
    categories.columns = columns
    for column in categories:
      categories[column] = categories[column].apply(lambda x: x[-1]).astype(int)

    # drop duplicates for column 'id' 
    df = pd.concat((df, categories), axis=1).drop('categories', axis=1).drop_duplicates(subset='id')
    return df

def save_data(df, database_filename):

    """
    Stores the dataset into a sqlite database

    Input: df: clean disaster dataframe
    database_filename: file name of database file

    """

    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('Response', engine, if_exists='replace', index=False)
    engine.dispose()  


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