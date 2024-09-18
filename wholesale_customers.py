# Part 2: Cluster Analysis

import pandas as pd
from sklearn import preprocessing
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    df = pd.read_csv(data_file)
    df = df.drop(['Channel', 'Region'], axis = 1)
    return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    mu = round(df[df.columns.to_list()].mean())
    sigma = round(df[df.columns.to_list()].std())
    mi = df[df.columns.to_list()].min()
    ma = df[df.columns.to_list()].max()
    summary_stats = pd.concat([mu,sigma,mi,ma], axis = 1)
    summary_stats.columns = ['mean', 'std', 'min', 'max']
    summary_stats.index = df.columns.to_list()
    return summary_stats


# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
    df_copy = df.copy()
    scaler = preprocessing.StandardScaler()
    standardized_df = scaler.fit_transform(df_copy) 
    standardized_df = pd.DataFrame(standardized_df)
    standardized_df.columns = df.columns.to_list()
    return standardized_df



# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
    km = cluster.KMeans(n_clusters=k, init='random', n_init = 1)
    km.fit(df)
    assign = km.labels_
    return pd.Series(assign)


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    km = cluster.KMeans(n_clusters=k, init='k-means++', n_init = 1)
    km.fit(df)
    assign = km.labels_
    return pd.Series(assign)



# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    ac = cluster.AgglomerativeClustering(k, linkage ='average', metric ='euclidean')
    ac.fit(df)
    assign = ac.labels_
    return pd.Series(assign)


# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X,y):
    s_score = metrics.silhouette_score(X, y , metric = 'euclidean')
    return s_score


# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
#When we have finalised our parameters for the below, use our above functions

def cluster_evaluation(df):
     k_values = {3,5,10}
     results = []
     
     for k in k_values:
         for i in range(10):
             km_labels = kmeans(df, k)
             km_s_score = metrics.silhouette_score(df, km_labels , metric = 'euclidean')
             results.append({'Algorithm' : 'Kmeans', 'data' : 'Original', 'k':k, 'Silhouette Score':km_s_score})
       
     for k in k_values:
         ac_labels = agglomerative(df, k)
         ac_s_score = metrics.silhouette_score(df, ac_labels, metric = 'euclidean')
         results.append({'Algorithm' : 'Agglomerative', 'data' : 'Original', 'k':k, 'Silhouette Score':ac_s_score})
         
     std_df = standardize(df)
        
     for k in k_values:
         for i in range(10):
             km_labels = kmeans(std_df, k)
             km_s_score = metrics.silhouette_score(std_df, km_labels , metric = 'euclidean')
             results.append({'Algorithm' : 'Kmeans', 'data' : 'Standardized', 'k':k, 'Silhouette Score':km_s_score})
         
     for k in k_values:   
         ac_labels = agglomerative(std_df, k)
         ac_s_score = metrics.silhouette_score(std_df, ac_labels, metric = 'euclidean')
         results.append({'Algorithm' : 'Agglomerative', 'data' : 'Standardized', 'k':k, 'Silhouette Score':ac_s_score})
         
     results = pd.DataFrame(results)
     return results
	

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
     max_ss = rdf['Silhouette Score'].max()
     return max_ss
	

# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
    std_df = standardize(df)
    labels = kmeans(std_df, 3)
 
    u_labels = np.unique(labels)
    col = ['red', 'green', 'blue']

    x = 1
    for k in range(len(std_df.columns)-1):
        for j in range(k+1, len(std_df.columns)):
            for i in u_labels:
                plt.scatter(std_df.iloc[:,k][labels == i],std_df.iloc[:,j][labels == i],label = i, color = col[i])
            plt.legend()
            plt.savefig('Plot{x}.png'.format(x = x))
            plt.show()
            x += 1






