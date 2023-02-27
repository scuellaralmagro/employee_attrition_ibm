# Case Study: IBM HR Analytics Employee Attrition & Performance

## **Introduction**

Employee attrition is an important issue for companies to understand, as high turnover can be costly and impact company culture and productivity. This project is designed for educational and practice purposes, with the goal of analyzing the IBM HR Analytics Employee Attrition & Performance dataset and building a machine learning model to predict employee attrition. The data utilized for this project is publicly available on Kaggle, and can be found [here](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset).

In this documentation, we will walk through the various stages of the data analysis and machine learning process, including data exploration, ETL, machine learning model building and model deployment. Each section wil provide readers with a step-by-step guide and insights into the data and models used.

---
## Table of Contents

1. [Introduction](#introduction)
2. [File Structure](#file-structure)
3. [Requirements](#requirements)
4. [Data](#data)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [ETL](#etl)
7. [Machine Learning](#machine-learning)
8. [Model Deployment](#model-deployment)
9. [Conclusion](#conclusion)
---

## **File Structure**

The file structure for this project is as follows:
1. .gitignore: File containing files and directories to be ignored by git. Not relevant to this project.
2. README.md: This file, containing documentation for the project.
3. data.csv: The dataset used for this project (downloaded from Kaggle).
4. ibm_eda.ipynb: Jupyter notebook containing the exploratory data analysis of the dataset.
5. etl.py: Python script containing the ETL process for the dataset.
6. ibm_ml.ipynb: Jupyter notebook containing the machine learning model building, comparison, and deployment.
7. predict.py: Python script containing the deployed machine learning model.

## **Requirements**

- **Python Version:** 3.11.2

The following Python packages are required to run the code in this project:
- **Pandas** (version 1.5.2)
- **Numpy** (version 1.23.5)
- **Matplotlib** (version 3.6.2)
- **Seaborn** (version 0.12.2)
- **Scikit-Learn** (version 1.2.1)
- **Joblib** (version 1.2.0)

These were the versions used for this project, but later versions should work as well.

## **Data**

This dataset is a collection of employee data from IBM. The dataset contains 1470 rows and 35 columns, with each row representing an employee and each column representing an attribute of the employee.

The dataset contains the following columns:
- **Age**: Age of the employee.
- **Attrition**: Whether the employee left the company or not.
- **BusinessTravel**: Whether the employee travels frequently or not.
- **DailyRate**: Daily rate of pay for the employee.
- **Department**: Department the employee works in.
- **DistanceFromHome**: Distance from home to work in miles.
- **Education**: Education level of the employee (1 = 'Below College', 2 = 'College', 3 = 'Bachelor', 4 = 'Master', 5 = 'Doctor').
- **EducationField**: Field of education of the employee (Life Sciences, Medical, Marketing, Technical Degree, Human Resources, Other).
- **EmployeeCount**: Number of employees in the company. Constant value of 1 for all observations, so this column will be dropped.
- **EmployeeNumber**: Unique identifier for each employee.
- **EnvironmentSatisfaction**: Employee's satisfaction with the work environment (1 = 'Low', 2 = 'Medium', 3 = 'High', 4 = 'Very High').
- **Gender**: Employee's gender (Male or Female, in this case).
- **HourlyRate**: Hourly rate of pay for the employee.
- **JobInvolvement**: Employee's level of involvement in their job (1 = 'Low', 2 = 'Medium', 3 = 'High', 4 = 'Very High').
- **JobLevel**: Employee's job level (1 = 'Entry Level', 2 = 'Mid Level', 3 = 'Senior Level', 4 = 'Manager').
- **JobRole**: Employee's job role (Sales Executive, Research Scientist, Laboratory Technician, Manufacturing Director, Healthcare Representative, Manager, Sales Representative, Research Director, Human Resources).
- **JobSatisfaction**: Employee's satisfaction with their job (1 = 'Low', 2 = 'Medium', 3 = 'High', 4 = 'Very High').
- **MaritalStatus**: Employee's marital status (Single, Married, Divorced).
- **MonthlyIncome**: Employee's monthly income, including base pay, bonus, and other benefits.
- **MonthlyRate**: Monthly rate of pay for the employee.
- **NumCompaniesWorked**: Number of companies the employee has worked for.
- **Over18**: Whether the employee is over 18 years old. Constant value of 'Y' for all observations, so this column will be dropped.
- **OverTime**: Whether the employee works overtime or not.
- **PercentSalaryHike**: Percent increase in salary from previous year.
- **PerformanceRating**: Employee's performance rating (1 = 'Low', 2 = 'Good', 3 = 'Excellent', 4 = 'Outstanding').
- **RelationshipSatisfaction**: Employee's satisfaction with their relationships at work (1 = 'Low', 2 = 'Medium', 3 = 'High', 4 = 'Very High').
- **StandardHours**: Number of standard work hours per week. Constant value of 80 for all observations, so this column will be dropped.
- **StockOptionLevel**: Employee's stock option level (0 = 'No Stock Options', 1 = 'Low', 2 = 'Medium', 3 = 'High').
- **TotalWorkingYears**: Total number of years the employee has worked. This includes time spent at other companies.
- **TrainingTimesLastYear**: Number of times the employee has received training in the past year.
- **WorkLifeBalance**: Employee's work-life balance (1 = 'Bad', 2 = 'Good', 3 = 'Better', 4 = 'Best').
- **YearsAtCompany**: Number of years the employee has worked at IBM.
- **YearsInCurrentRole**: Number of years the employee has worked in their current role.
- **YearsSinceLastPromotion**: Number of years since the employee's last promotion.
- **YearsWithCurrManager**: Number of years the employee has worked with their current manager.

## **Exploratory Data Analysis**

Before diving into the machine learning process, it's important to understand the data through exploratory data analysis (EDA). EDA involves analyzing the data to discover patterns, spot anomalies and check assumptions. This is an important step in the data science process, as it allows us to understand the data and make informed decisions about the machine learning process.

The EDA process for this project is as follows:
1. **Import the dataset.** We will use the Pandas library to import the dataset into a Pandas DataFrame.
2. **Preliminary data exploration.** We will use Pandas to do a quick exploration of the dataset, including checking the shape of the dataset, the data types of each column, the number of missing values in each column and the number of unique values in each column. Conclusions from this step are as follows:
    - The dataset contains 1470 rows and 35 columns.
    - The dataset contains 8 numerical columns and 27 categorical columns.
    - The dataset contains no missing values.
    - The dataset contains 2 columns with constant values (EmployeeCount and Over18), which will be dropped.
    - The EmployeeNumber column will also be dropped, as it is a unique identifier for each employee and will not be useful for the machine learning or the EDA processes.
3. **Further Analysis.** Using Pandas, Matplotlib and Seaborn, we did further analysis of the financial, personal and work-related attributes of the employees. Here are some of the most significant findings:
    - **Age**: We found that younger employees were more likely to leave the company, with a particularly high attrition rate for employees aged 18-25 years old. 
    - **Job Role**: Sales representatives were the most likely to leave the company, with a 40% attrition rate. On the other hand, managers and research directors were the least likely to leave the company on average.
    - **Distance from Home**: Employees who lived further away from work were more likely to leave the company, with a particularly high attrition rate for employees who lived more than 25 miles away from work. However, it is important to note that the majority of employees lived within 10 miles of work.
    - **Department**: The sales department had the highest attrition rate, with a 22.62% attrition rate. The research and development department had the lowest rate, with an overall attrition rate of 13.83%.
    - **Business Travel**: Employees who traveled frequently were more likely to leave the company, with a 25% attrition rate. Employees who rarely traveled were the least likely to leave the company, with an attrition rate of 10.5%.
    - **Over Time**: Employees who worked overtime were more likely to leave the company than those who didn't, with an attrition rate of almost 30%.
4. **Conclusion.** Based on the findings from the EDA process, we can conclude that the following factors are important in predicting employee attrition:
    - Age
    - Job Role
    - Distance from Home
    - Department
    - Business Travel
    - Over Time

## **ETL**

ETL stands for Extract, Transform, and Load. It is the process of collecting data from one or more sources (in this case, the survey that IBM would have conducted on their employees), cleaning and transforming it, and loading it into a target database or application. In this project, we will use the ETL process to prepare the data for the machine learning process.

1. **Extract.** We used the Pandas library to import the dataset (*data.csv*) into a Pandas DataFrame.
2. **Data Cleaning.** We dropped the columns that we previously determined were not useful for the machine learning process (EmployeeCount, Over18, EmployeeNumber and StandardHours).
3. **Feature Engineering.** We encoded the categorical columns with only two unique values using a simple function. This includes the *Attrition* column, which would also be our target. We also one-hot encoded the categorical columns with more than two unique values using the Pandas get_dummies() function. 
4. **Data Loading.** We used the Pandas to_csv() function to save the cleaned and transformed dataset to a new file (*data_processed.csv*). This file will be used later in the machine learning process.

## **Machine Learning**

During the machine learning process, we used the sci-kit learn library to build a machine learning model that predicts whether an employee will leave the company or not. We used the following machine learning algorithms:
- Logistic Regression
- Decision Tree
- Gradient Boosting
- Support Vector Machine
- K-Nearest Neighbors
First, we loaded the previously processed dataset into a Pandas DataFrame. We then split the dataset into the features (X) and the target (y). We then split the dataset again into training and testing sets. We used 80% of the dataset for training and 20% for testing.

We then fitted and tuned the machine learning models using the training set. We used the following metrics to evaluate the performance of the models:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score

Each one of these metrics are explained in detail in the notebook.

For tuning, we used the RandomizedSearchCV function from the sci-kit learn library. This function performs a randomized search over the hyperparameters of the model. We opted for a randomized search over a grid search because the number of hyperparameters for each model was quite large, and we didn't noticably lose any accuracy by using a randomized search.

After we tuned, fitted and tested all of the models, we found that the Logistic Regression model performed the best in the metrics that we cared most about (recall and F1 score). The Logistic Regression model had a recall of 0.46 and an F1 score of 0.59. However, these metrics are still not very good. This could be improved by balancing the dataset, which we will attempt in the future.

## **Model Deployment**

In this section, we will deploy the machine learning model using a simple python script. In a real production environment, we would use a more robust deployment method, such as a web application or a REST API. However, for the purposes of this project, we will use a simple python script.

The python script will import the previously trained model and scaler, and then ask the user to input the values for each feature. The script will then scale the features using the scaler, and then use the model to predict whether the employee will leave the company or not.

We included a random data generator in the script, which will generate random values for each feature. This is useful for testing the script, however, the data generated by the random data generator may not be realistic.

## **Conclusion**

Overall, this project was a great learning experience. We learned how to perform exploratory data analysis, data cleaning, feature engineering, machine learning and model deployment. 
