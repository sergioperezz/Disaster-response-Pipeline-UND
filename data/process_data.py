import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    ETL: 1-5
    function to load files, create categorical values, 
    Args: the function gets both path to files to message and categories
    return: A dataframe merged
    """
    # load the files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merge files
    df = messages.merge(categories,on=['id'])

    #make categorical values from categories
    categories = df["categories"].str.split(";",expand=True)
    category_colnames = categories.iloc[0].replace('-.+','',regex=True)
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].replace(column+"-","",regex=True)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop(columns="categories",inplace=True)
    df = pd.concat([df,categories],axis=1)
    
    #hot encoding the genre feature
    genre = pd.get_dummies(df["genre"])
    df = pd.concat([df, genre], axis=1)
    df=df.drop(columns=["direct"])
    
    return df
        
def clean_data(df):
    """
    Remove duplicates values from the df
    Args: df
    Return: df without duplicates values
    """
    return df.drop_duplicates()


def save_data(df, database_filename):
    """
    save the result into a dataframe
    Args: dataFrame and database name
    return: 
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('df', engine, index=False)


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