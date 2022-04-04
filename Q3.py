# Import libraries
import seaborn as sns
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.cluster import KMeans 
from sklearn.metrics import confusion_matrix

st.title("Question 3: Loan Application Modelling")

st.sidebar.header("User Input")
train_ratio = st.sidebar.slider("Training Set Ratio", min_value = 0.1, max_value = 1.0, value = 0.7, step = 0.1)
k_cluster = st.sidebar.slider("Number of Clusters (k)", min_value = 1, max_value = 10, value = 5, step = 1)
data = {"train_ratio" : train_ratio,
         "clus_k" : k_cluster}
        

test_ratio = round((1 - train_ratio), 1)
k_cluster = int(k_cluster)


# About the Dataset

# Read Dataset
df = pd.read_csv('Bank_CreditScoring.csv')
st.header("Dataset")
st.write(df)

# Dataset Info/ Dimensions
#Check dataset's info
st.subheader("Check dataset info")
buffer = io.StringIO()
df.info(buf=buffer)
info = buffer.getvalue()

st.text(info)

st.write()

st.subheader("Find the dimension of dataframe")
shape = df.shape
st.write("Size of dataset :",shape)

st.markdown("The original data set have 21 classes and 2350 samples.")
st.markdown("There are 16 classes are numerical data whereas 5 classes are categorical data.")

# Find Unique Values for each columns

st.subheader("Find Unique Values for each columns")
for col_names,col in zip(df.columns,df):
    st.write(f'{col_names}: {df[col].unique()}\n')

# Show Quantitative Data

st.subheader("Analyze Quantitative Data")
st.markdown("Numerical data")
st.dataframe(df.describe())


# Show Categorical Data

st.subheader("Analyze Categorical Data")
st.markdown("Categorial data")
cate = df.describe(include=['object']) #describe categorical columns
st.dataframe(cate.astype(str))

# Find Correlation relationship between all columns 

st.subheader("Correlation Heatmap")
fig1 = plt.figure()
sns.set(rc = {'figure.figsize':(20,10)})
sns.heatmap(df.corr().round(3), annot = True)
st.pyplot(fig1)

st.markdown("While the function only can process numerical data, the categorial data must be convert to numerical data for further data processing")
#categorical columns not include

# Data Preprocessing

st.subheader("Checking whether missing value exists")
#find whether missing value exists
st.dataframe(df.isnull().sum() )


# Clean Up Categorical Data to Numerical Data
# change categorical data to numerical data
st.subheader("Change categorical data to numerical data")
st.markdown("After checking no missing values detected, we encode the categorical data.")

df["Decision"] = df.Decision.map(dict(Accept = 1, Reject = 0))

LabelEncoder = LabelEncoder()
clean_df = df.copy()
clean_df = clean_df.apply(LabelEncoder.fit_transform)
st.dataframe(clean_df.head())


# Confirmation Datatypes
st.subheader("Checking data types")
st.markdown("To check whether all categorical data have changed to numerical data")

st.dataframe(clean_df.dtypes.astype(str)) #checking data types

# Correlations Heatmap after Preprocessing Data

st.subheader(" Correlations Heatmap after Preprocessing Data")
fig2 = plt.figure()
sns.set(rc = {'figure.figsize':(20,15)})
sns.heatmap(clean_df.corr().round(3), annot = True)
st.pyplot(fig2)
st.markdown("Now, all classes have been have been ready for analyze")

# Correlation value against Desicion

st.subheader("Correlation value against Desicion")
st.markdown("Discover the correlation between Decision and others classes")

cleancorr = clean_df.corr().round(3)["Decision"]
st.dataframe(cleancorr.astype(str))


# Spilt dataset

# For this example, we use the mass, width, and height features of each fruit instance
X = clean_df.drop("Decision", axis = 1)    # big X represents independent variable
y = clean_df["Decision"]                   # small y represents dependent variable
st.markdown("By default Train Set = 70%, Test Set = 30%")

st.markdown("**Independent Variables:**")
st.write(X)
st.markdown("**Dependent Variable:**")
st.write(y)

# Prepare X_train, X_test, y_train, y_test
# Random State = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = train_ratio, random_state = 1)

# Feature Selection for cluster process on K-Means
StandardScaler = StandardScaler()
X_train = StandardScaler.fit_transform(X_train)
X_test = StandardScaler.fit_transform(X_test)


# # Classifications&nbsp;1&nbsp;: Naive Bayes
st.markdown("""# Classification 1: Naive Bayes""")
st.markdown("Naive Bayes Classifier is based on Bayes’ Theorem.")

#Create a Gaussian Classifier
# Train the model using the training sets
nb = GaussianNB()
nb.fit(X_train, y_train)

# Prediction

#Predict Output
y_pred = nb.predict(X_test)

# Accuracy for Naive Bayes

# accuracy
nb.score(X_test, y_test)
st.write("Accuracy for Classifications 1 : Naive Bayes -> ", nb.score(X_test, y_test))


# Confusion Matrix

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
st.subheader("Visualization of the Confusion Matrix")
plt.show()
st.pyplot(fig3)

# # Classification&nbsp;2&nbsp;:&nbsp;Decision&nbsp;Tree
st.write("""# Classification 2: Decision Tree""")
st.markdown("A Decision Tree is a supervised Machine learning algorithm. It is used in both classification and regression algorithms. ")
# Create Decision Tree classifer object
dt = DecisionTreeClassifier()

# Train Decision Tree Classifer
dt = dt.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dt.predict(X_test)
st.write("Accuracy before optimize:",metrics.accuracy_score(y_test, y_pred))

st.subheader("Optimizing Decision Tree Performance")

# Create Decision Tree classifer object
dt = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
dt = dt.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dt.predict(X_test)

# Prediction after optimize
st.write("Accuracy after optimize:",metrics.accuracy_score(y_test, y_pred))

st.markdown("From the result above shows that after we optimize the decision tree by tuning the performance is faster than before we optimze the model. ")

# Confusion Matrix

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

st.write("""# Clustering: K-Means""")
st.markdown("We decide to use *\"Monthly_Salary\"* and *\"Years_to_Financial_Freedom\"* to perform K-Means clustering.")

#normalize data
st.markdown("Before we clustering, the model need to be normalize to make sure the data are equally.")

cluster_data = preprocessing.normalize(clean_df)
cluster_data = pd.DataFrame(cluster_data, columns = clean_df.columns)
st.dataframe(cluster_data.head())

st.write("""# Columns to be Visualised""")
st.markdown("We have selected columns  *\"Monthly_Salary\"* and *\"Years_to_Financial_Freedom\"* and check for the value.")

# selects columns target to be visualised
km_data = cluster_data[[ "Monthly_Salary","Years_to_Financial_Freedom"]]
st.dataframe(km_data.head())

#Create plots to plotting the relationship
fig5 = plt.figure()
st.write("""# Plot the Relationship""")
sns.scatterplot(x="Years_to_Financial_Freedom", y="Monthly_Salary", hue=y, data=cluster_data)
st.pyplot(fig5)
st.markdown("From the result above, it is hard to determine the number of clusters. We need to use Elbow method to find the K value.")

# loop from 1 to 10 to find the best k value
st.write("""# Find the best K value""")
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

st.markdown("From the line graph above, we can see that the elbow point is at 5. Number of clusters = 5.")
st.write("**Adjust the **Number of Clusters (k)** in the sidebar and investigate the scatter plot again.**")


# 5 is select as the number of clusters
km = KMeans(n_clusters = k_cluster, random_state = 1)
st.write(km.fit(km_data))

# merging data with labeling for visualize
distortions = []
merge_data = cluster_data.copy()
merge_data = merge_data[km_data.columns]
merge_data["Clusters"] = km.labels_

# compare the differences before and after
ax = plt.figure()
sns.scatterplot(x="Years_to_Financial_Freedom", y="Monthly_Salary", hue = "Clusters", data = merge_data)

st.pyplot(ax)
