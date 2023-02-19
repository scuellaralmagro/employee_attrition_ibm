# IBM HR Analytics Employee Attrition & Performance prediction model
# Description: This model predicts, based on the features of an employee, if they will leave the company or not.
# Author: @scuellaralmagro
# Date: February 2023

import pandas as pd
import joblib
import random

## Helper functions

def get_model(file):
    '''Loads the model from file'''
    return joblib.load(file)

def get_scaler(file):
    '''Loads the scaler from file'''
    return joblib.load(file)

def get_employee_data():
    '''Gets employee data from user input. Returns two
    dictionaries: one with numerical data and the other with
    categorical data. Also return employee_id, which will only
    be used to print the employee ID in the output.'''

    print('-- Please enter the following employee data --')

    employee_id = int(input('Employee ID: ')) # This is not used in the model

    # Personal data
    age = int(input('Age: '))
    gender = int(input('Gender: '))
    marital_status = input('Marital status: ')
    distance_from_home = int(input('Distance from home (miles): '))
    education = int(input('Education level: '))
    education_field = input('Education field: ')

    # Financial data
    daily_rate = int(input('Daily rate: '))
    hourly_rate = int(input('Hourly rate: '))
    monthly_rate = int(input('Monthly rate: '))
    monthly_income = int(input('Monthly income: '))
    percent_salary_hike = int(input('Percent salary hike: '))
    stock_option_level = int(input('Stock option level: '))
    
    # Satisfaction data
    environment_satisfaction = int(input('Environment satisfaction: '))
    job_satisfaction = int(input('Job satisfaction: '))
    relationship_satisfaction = int(input('Relationship satisfaction: '))
    work_life_balance = int(input('Work life balance: '))
    job_involvement = int(input('Job involvement: '))

    # Work & Performance data
    job_level = int(input('Job level: '))
    performance_rating = int(input('Performance rating: '))
    job_role = input('Job role: ')
    department = input('Department: ')
    overtime = input('Overtime: ')
    num_companies_worked = int(input('Number of companies worked for: '))
    total_working_years = int(input('Total working years: '))
    training_times_last_year = int(input('Training times last year: '))
    years_at_company = int(input('Years at company: '))
    years_in_current_role = int(input('Years in current role: '))
    years_since_last_promotion = int(input('Years since last promotion: '))
    years_with_curr_manager = int(input('Years with current manager: '))
    business_travel = input('Business travel: ')

    print('-- Employee data entered --')

    # Create a dictionary with the employee data
    employee_data_numerical = {
        'Age': age,
        'DailyrandRate': daily_rate,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': gender,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobSatisfaction': job_satisfaction,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies_worked,
        'OverTime': overtime,
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrent Role': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }

    employee_data_categorical = {
        'BusinessTravel': business_travel,
        'Department': department,
        'EducationField': education_field,
        'JobRole': job_role,
        'MaritalStatus': marital_status
    }

    return employee_data_numerical, employee_data_categorical, employee_id

def generate_random_employee_data():
    '''Generates random employee data. Returns two
    dictionaries: one with numerical data and the other with
    categorical data. Also return employee_id, which will only
    be used to print the employee ID in the output.'''

    print('-- Generating random employee data --')
    employee_id = random.randint(1, 9999) # This is not used in the model

    # Personal data
    age = random.randint(18, 65)
    gender = random.randint(0, 1)
    marital_status = random.choice(['Single', 'Married', 'Divorced'])
    distance_from_home = random.randint(1, 100)
    education = random.randint(1, 5)
    education_field = random.choice(['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])

    # Financial data
    daily_rate = random.randint(100, 1500)
    hourly_rate = random.randint(30, 100)
    monthly_rate = random.randint(2000, 25000)
    monthly_income = random.randint(1000, 20000)
    percent_salary_hike = random.randint(6, 25)
    stock_option_level = random.randint(0, 3)

    # Satisfaction data
    environment_satisfaction = random.randint(1, 4)
    job_satisfaction = random.randint(1, 4)
    relationship_satisfaction = random.randint(1, 4)
    work_life_balance = random.randint(1, 4)
    job_involvement = random.randint(1, 4)

    # Work & Performance data
    job_level = random.randint(1, 5)
    performance_rating = random.randint(1, 4)
    job_role = random.choice(['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                                'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
    department = random.choice(['Sales', 'Research & Development', 'Human Resources'])
    overtime = random.randint(0, 1)
    num_companies_worked = random.randint(0, 10)
    total_working_years = random.randint(0, 40)
    training_times_last_year = random.randint(0, 6)
    years_at_company = random.randint(0, 40)
    years_in_current_role = random.randint(0, 18)
    years_since_last_promotion = random.randint(0, 15)
    years_with_curr_manager = random.randint(0, 18)
    business_travel = random.choice(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
    print('-- Random employee data generated --')

    # Create a dictionary with the employee data
    employee_data_numerical = {
        'Age': age,
        'DailyRate': daily_rate,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': gender,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobSatisfaction': job_satisfaction,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies_worked,
        'OverTime': overtime,
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }
    employee_data_categorical = {
        'BusinessTravel': business_travel,
        'Department': department,
        'EducationField': education_field,
        'JobRole': job_role,
        'MaritalStatus': marital_status
    }

    return employee_data_numerical, employee_data_categorical, employee_id

## Main function

def main():
    print('\n-- Loading model and scaler --\n')
    model = get_model('logreg_cv.joblib')
    scaler = get_scaler('scaler.joblib')

    ## Get employee data
    while True:
        choice = input('Do you want to enter the employee data manually or generate random data? (manual/random): ')
        if choice == 'manual':
            dict_num, dict_cat, employee_id = get_employee_data()
            break
        elif choice == 'random':
            print('\n-- ATTENTION -- \n')
            print('The generated data is completely random, and therefore not representative of reality.\nThis feature should only be used for testing purposes.\n')
            dict_num, dict_cat, employee_id = generate_random_employee_data()
            break
        else:
            print('Invalid choice. Please try again.')

    ## Create a dataframe with the numerical data
    df_num = pd.DataFrame(dict_num, index=[0])
    
    # Create a dataframe with the categorical data
    # Define the possible values for each categorical feature
    possible_values = {
        'BusinessTravel': ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'],
        'Department': ['Human Resources', 'Research & Development', 'Sales'],
        'EducationField': ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'],
        'JobRole': ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 
                     'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'],
        'MaritalStatus': ['Divorced', 'Married', 'Single']
    }

    # Create an empty dataframe with the possible values for each categorical feature

    one_hot_df = pd.DataFrame(columns=[f'{col}_{val}' for col, vals in possible_values.items() for val in vals])

    # Create a DataFrame with a single row of data to be one-hot encoded

    df_cat = pd.DataFrame([dict_cat])

    # One-hot encode the categorical data

    df_cat = pd.get_dummies(df_cat, columns=possible_values.keys())
    df_cat = df_cat.reindex(columns=one_hot_df.columns, fill_value=0)

    # Concatenate the numerical and categorical dataframes

    df = pd.concat([df_num, df_cat], axis=1)

    # Scale the data

    df = scaler.transform(df)

    # Make a prediction

    prediction = model.predict(df)[0]

    # Print the results

    print('\n-- Prediction --\n')
    print(f'Employee ID: {employee_id}')
    print('\n-- Employee Data --\n')
    print(dict_num)
    print(dict_cat)
    print('\n-- Results --\n')
    if prediction == 0:
        print(f'Employee {employee_id} will not leave the company')
    else:
        print(f'Employee {employee_id} will leave the company')

    print('\n-- Accuracy of the model: 89.11% --\n')


## Run main function only if this file is executed, not imported

if __name__ == '__main__':
    main()