
# Part 1: Decision Trees with Categorical Attributes

import pandas as pd
from sklearn import preprocessing
from sklearn import tree

# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.
def read_csv_1(data_file):
    df = pd.read_csv(data_file)
    df = df.drop(['fnlwgt'], axis = 1)
    return df

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
    return len(df)


# Return a list with the column names in the pandas dataframe df.
def column_names(df):
    att = df.columns.tolist()
    return att


# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
    return df.isnull().sum().sum()


# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
    return df.columns[df.isnull().any()].tolist()
	

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
    num_higher = df['education'].value_counts()[['Bachelors' ,'Masters']].sum()
    percentage = (num_higher/len(df['education']))*100
    return round(percentage, 1)


# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
    df_copy = df.copy()
    df_copy = df_copy.dropna(axis=0)
    return df_copy


# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
    df_copy = df.copy()
    df_copy = df_copy.drop(['class'], axis = 1)
    att = df_copy.columns.tolist()
    att.remove('education-num')
    one_hot = pd.get_dummies(df_copy, columns=att)
    return one_hot

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
    df_copy = df.copy()
    att = df.columns.tolist()
    att.remove('class')
    df_copy = df_copy.drop(att, axis = 1)
    label_encoder = preprocessing.LabelEncoder()
    df_copy['class'] = label_encoder.fit_transform(df_copy['class'])
    return df_copy


# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
    dt = tree.DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    predicted = dt.predict(X_train)
    return pd.Series(predicted)

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
    count = 0.0
    for i in range(len(y_true)):
        if (y_pred[i] == y_true[i]):
            count += 1

    score = (count/len(y_true)) 
    error = 1 - score
    return round(error,3)










