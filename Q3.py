#!/usr/bin/env python
# coding: utf-8

# # Question&nbsp;3&nbsp;:&nbsp;Loan&nbsp;Application&nbsp;Modeling

# Import libraries

# In[ ]:


#Data Processing
import seaborn as sns
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
#Confusion Matrix 
from sklearn.metrics import confusion_matrix

#Use in Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

#Use in Decision Tree
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.cluster import KMeans 
import io

st.set_option('deprecation.showPyplotGlobalUse', False)
# About the Dataset

# Read Dataset

# In[ ]:


df = pd.read_csv('Bank_CreditScoring.csv') # Bank_CreditScoring.csv
st.write("""# Read Dataset""")
st.write(df.head())


# Dataset Info/ Dimensions

# In[ ]:


#Check dataset's info
st.write("""# Check dataset info""")
buffer = io.StringIO()
df.info(buf=buffer)
info = buffer.getvalue()

st.text(info)

st.write()

st.write("""# Find the dimension of dataframe""")
shape = df.shape
st.write("Size of dataset :",shape)


# Find Unique Values for each columns

# In[ ]:

st.write("""# Find Unique Values for each columns""")
for col_names,col in zip(df.columns,df):
    st.write(f'{col_names}: {df[col].unique()}\n')


# Show Quantitative Data

# In[ ]:

st.write("""# Show Quantitative Data""")
st.dataframe(df.describe())


# Show Categorical Data

# In[ ]:

st.write("""# Show Categorical Data""")
cate = df.describe(include=['object']) #describe categorical columns
st.dataframe(cate.astype(str))

# Find Correlation relationship between all columns 

# In[ ]:

st.write("""# Find Correlation Relationship""")
fig1 = sns.set(rc = {'figure.figsize':(20,10)})
sns.heatmap(df.corr().round(3), annot = True)
st.pyplot(fig1)

#categorical columns not include


# Data Preprocessing

# In[ ]:

st.write("""# Find whether missing value exists""")
#find whether missing value exists
st.dataframe(df.isnull().sum() )


# Clean Up Categorical Data to Numerical Data

# In[ ]:


# change categorical data to numerical data
df["Decision"] = df.Decision.map(dict(Accept = 1, Reject = 0))

LabelEncoder = LabelEncoder()
clean_df = df.copy()
clean_df = clean_df.apply(LabelEncoder.fit_transform)
st.write("""# Change categorical data to numerical data""")
st.dataframe(clean_df.head())


# Confirmation Datatypes

# In[ ]:

st.write("""# Check data types""")
st.dataframe(clean_df.dtypes.astype(str)) #checking data types


# Correlations Heatmap after Preprocessing Data

# In[ ]:

st.write("""# Correlations Heatmap after Preprocessing Data
""")
fig2 = sns.set(rc = {'figure.figsize':(20,15)})
sns.heatmap(clean_df.corr().round(3), annot = True)
st.pyplot(fig2)

# Correlation value against Desicion

# In[ ]:

st.write("""# Correlation value against Desicion""")
cleancorr = clean_df.corr().round(3)["Decision"]
st.dataframe(cleancorr.astype(str))

# Spilt dataset

# In[ ]:


# For this example, we use the mass, width, and height features of each fruit instance
X = clean_df.drop("Decision", axis = 1)    # big X represents independent variable
y = clean_df["Decision"]                   # small y represents dependent variable

# Prepare X_train, X_test, y_train, y_test
# Train Set = 70%
# Test Set = 30%
# Random State = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# Feature Selection for cluster process on K-Means
StandardScaler = StandardScaler()
X_train = StandardScaler.fit_transform(X_train)
X_test = StandardScaler.fit_transform(X_test)


# # Classifications&nbsp;1&nbsp;: Naive Bayes

# In[ ]:
st.write("""# Classification 1: Naive Bayes""")

#Create a Gaussian Classifier
# Train the model using the training sets
nb = GaussianNB()
nb.fit(X_train, y_train)


# Prediction

# In[ ]:


#Predict Output
y_pred = nb.predict(X_test)


# Accuracy for Naive Bayes

# In[ ]:


# accuracy
st.write("Accuracy")
nb.score(X_test, y_test)
st.write("Classifications 1 : Naive Bayes -> ", nb.score(X_test, y_test))


# Confusion Matrix

# In[ ]:




nb_matrix = confusion_matrix(y_test, y_pred)

fig3 = plt.figure(figsize = (10,5))

ax = sns.heatmap(nb_matrix, annot=True, cmap='Blues', cbar = False,fmt='g')

ax.set_title('Naive Bayes Confusion Matrix');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1'])
ax.yaxis.set_ticklabels(['0','1'])

## Display the visualization of the Confusion Matrix.
st.write("""# Visualization of the Confusion Matrix""")
plt.show()
st.pyplot(fig3)

# # Classification&nbsp;2&nbsp;:&nbsp;Decision&nbsp;Tree
st.write("""# Classification 2: Decision Tree""")
# In[ ]:


# Create Decision Tree classifer object
dt = DecisionTreeClassifier()

# Train Decision Tree Classifer
dt = dt.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dt.predict(X_test)
st.write("Accuracy before optimize:",metrics.accuracy_score(y_test, y_pred))


# Optimizing Decision Tree Performance

# In[ ]:


# Create Decision Tree classifer object
dt = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
dt = dt.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dt.predict(X_test)

# Prediction after optimize
print("Classifications 2 : Decision Tree ")
st.write("Accuracy after optimize:",metrics.accuracy_score(y_test, y_pred))


# Confusion Matrix

# In[ ]:
st.write("""# Confusion Matrix""")

dt_matrix = confusion_matrix(y_test, y_pred)

fig4 = plt.figure(figsize = (10,5))

ax = sns.heatmap(dt_matrix, annot=True, cmap='Blues', cbar = False,fmt='g')

ax.set_title('Decision Tree Confusion Matrix');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1'])
ax.yaxis.set_ticklabels(['0','1'])

## Display the visualization of the Confusion Matrix.
plt.show()
st.pyplot(fig4)

# # Clustering&nbsp;:&nbsp;K-Means

# In[ ]:

st.write("""# Clustering: K-Means""")
#normalize data
cluster_data = preprocessing.normalize(clean_df)
cluster_data = pd.DataFrame(cluster_data, columns = clean_df.columns)
st.dataframe(cluster_data.head())


# In[ ]:

st.write("""# Columns to be Visualised""")
# selects columns target to be visualised
km_data = cluster_data[[ "Monthly_Salary","Years_to_Financial_Freedom"]]
st.dataframe(km_data.head())


# In[ ]:


#Create plots to plotting the relationship
fig5 = plt.figure()
st.write("""# Plot the Relationship""")
sns.scatterplot(x="Years_to_Financial_Freedom", y="Monthly_Salary", hue=y, data=cluster_data)
st.pyplot(fig5)

# In[ ]:


# loop from 1 to 10 to find the best k value
st.write(""" #Find the best K value""")
distortions = []
for i in range (1, 11):
    km = KMeans(
        n_clusters = i, init = "random",
        n_init = 10, max_iter = 300,
        tol = 1e-04, random_state = 1
    )
    km.fit(km_data)
    distortions.append(km.inertia_)
    
# plot
fig6 = plt.figure()
plt.plot(range(1, 11), distortions, marker = "x")
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion")
plt.show()
st.pyplot(fig6)

# In[ ]:


# 5 is select as the number of clusters
km = KMeans(n_clusters = 5, random_state = 1)
st.write(km.fit(km_data))


# In[ ]:


# merging data with labeling for visualize
st.write("""# Merge Data""")
distortions = []
merge_data = cluster_data.copy()
merge_data = merge_data[km_data.columns]
merge_data["Clusters"] = km.labels_
st.dataframe(merge_data.head())


# In[ ]:


# compare the differences before and after
st.write("""# Compare the Differences Before and After""")
fig, axes = plt.subplots(1, 2, figsize = (15, 10))

sns.scatterplot(x="Years_to_Financial_Freedom", y="Monthly_Salary", hue = y, data = cluster_data, ax = axes[0])
sns.scatterplot(x="Years_to_Financial_Freedom", y="Monthly_Salary", hue = "Clusters", data = merge_data, ax = axes[1])

st.pyplot(fig)
# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=158f0e5f-0c09-4da4-a8e2-16b96f03a497' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
