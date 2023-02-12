import pandas as pd

# Description: This script preprocesses the IBM HR Analytics Employee Attrition & Performance dataset and saves it to a csv file.

## Helper functions

def load_csv(path):
    '''Loads a csv file into a pandas dataframe'''
    return pd.read_csv(path)

def columns_drop(df, columns):
    '''Drops columns from a dataframe'''
    return df.drop(columns, axis=1)

def categorical_columns(df):
    '''Returns a list of columns that are categorical'''
    return df.select_dtypes(include=['object']).columns.tolist()

def one_hot_encode(df, columns):
    '''One-hot encodes a list of columns'''
    return pd.get_dummies(df, columns=columns)

def encode_gender(gender):
    '''Encodes gender column into 1 or 0'''
    if gender == 'Male':
        return 1
    else:
        return 0

def encode_yes_no(column):
    '''Encodes a Yes or No column into 1 or 0'''
    if column == 'Yes':
        return 1
    else:
        return 0

## Main function

def main():
    # Load IBM HR Analytics Employee Attrition & Performance dataset from data.csv
    print('Loading data.csv...')
    df = load_csv('data.csv')
    print('Commencing Data Preprocessing...')

    # Drop columns that are not needed for the analysis
    df = columns_drop(df, ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'])
    
    ## We are dropping these columns because EmployeeCount, Over18, StandardHours 
    ## have only one value; and EmployeeNumber is just an identifier for each employee.

    # Get categorical and numerical columns
    cat_columns = categorical_columns(df)

    # Get categorical columns with only two unique values
    cat_columns_two_unique = [col for col in cat_columns if df[col].nunique() == 2]
    # Remove columns in cat_columns_two_unique from cat_columns
    cat_columns = [col for col in cat_columns if col not in cat_columns_two_unique]
    
    # Encode categorical columns with only two unique values ['Gender', 'OverTime']
    df['Gender'] = df['Gender'].apply(encode_gender)
    df['OverTime'] = df['OverTime'].apply(encode_yes_no)
    # One-hot encode categorical columns
    df = one_hot_encode(df, cat_columns)
    # Encode target column
    df['Attrition'] = df['Attrition'].apply(encode_yes_no)
    
    print('Final Result:')
    print(df.info())
    print('Data Preprocessing Completed')

    # Save preprocessed data to csv file
    df.to_csv('data_preprocessed.csv', index=False)

## Run main function only if this file is run as a script

if __name__ == '__main__':
    main()