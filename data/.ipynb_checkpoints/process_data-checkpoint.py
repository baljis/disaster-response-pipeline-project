import sys
import pandas as pd
import numpy as np
import sqlite3 as sq

def col_data_to_row(df, col, outer_split = None, inner_split = None):
    try:
        col_data = df[col][0] 

        split_col_data = col_data.split(outer_split) #split row values using outer split symbol

        new_cols = list(map(lambda col: col.split(inner_split)[0], split_col_data))#get new columns 

        df[new_cols] = None

        final_ls = []

        def split_row_vals(row_val):
            row_split = row_val.split(outer_split)
            vals = list(map(lambda x : x.split(inner_split)[-1], row_split))
            final_ls.append(vals)

        df[col].apply(lambda x : split_row_vals(x))
        df[list(df.columns[2:])] = final_ls

        return df.drop([col],axis = 1)
    except:
        print("Task not achieved in col_data_to_row() ")
    
    return None


def load_data(messages_filepath, categories_filepath):
    """
    input : message_filepath - csv file containing original message and their english counterpart
            categories_filepath - csv file containing categories a message belongs to
    output : dataframe 
    """
    try:
        df_messages = pd.read_csv(messages_filepath)
        df_categories = pd.read_csv(categories_filepath)
        
        return [df_messages,df_categories]
    except:
        print("Load Failed in load_data()")
    
    return None

def clean_data(df):
    """
    input : pandas dataframe
    output : cleaned pandas dataframe
    """
    try:
        for mini_df in df:
            for col in mini_df.columns:
                if mini_df[col].isnull().sum()/mini_df.shape[0] > 0.4:
                    mini_df.drop([col], axis = 1, inplace = True)

        df[-1] = col_data_to_row(df[-1], 'categories', ';', '-')

        return df[0].set_index('id').join(df[1].set_index('id'))
    except:
        print("cleaning failed in clean_data()")
        
    return None

def save_data(df, database_filename):
    try:
        db = sq.connect(database_filename)
    
        df.to_sql('messages_categories', db, if_exists = 'replace')
    except:
        print("saving process failed in save_data()")


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