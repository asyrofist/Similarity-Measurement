import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn import neighbors
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Classification
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.sidebar.subheader('Classification Parameter')
randomforest = st.sidebar.checkbox('RandomForest')
knearest     = st.sidebar.checkbox('knearest')
gaussian     = st.sidebar.checkbox('gaussian')
decission    = st.sidebar.checkbox('decission')
svm          = st.sidebar.checkbox('svm')

if randomforest:
  # random forest classifier
  clf = RandomForestClassifier()
  clf.fit(X, Y)
  prediction = clf.predict(df)
  prediction_proba = clf.predict_proba(df)
  st.subheader('Prediction RF')
  st.write(iris.target_names[prediction])

elif knearest:
  # K-Nearset Neighbor
  kNN = neighbors.KNeighborsClassifier(n_neighbors = 10, weights='distance')
  kNN.fit(X, Y)
  prediction_knn = kNN.predict(df)
  st.subheader('Prediction KNN')
  st.write(iris.target_names[prediction_knn])

elif gaussian:
  # Gaussian Naive Bias
  gnb = GaussianNB()
  nbc = gnb.fit(X, Y) 
  prediction_gnb = nbc.predict(df)
  st.subheader('Prediction GNB')
  st.write(iris.target_names[prediction_gnb])

elif decission:
  # Decission Tree
  DT = tree.DecisionTreeClassifier()
  DT = DT.fit(X, Y)
  prediction_DT = DT.predict(df)
  st.subheader('Prediction Decission Tree')
  st.write(iris.target_names[prediction_DT])

elif svm:
  dSVM = SVC(decision_function_shape='ovo') # one versus one SVM
  dSVM.fit(X, Y)
  prediction_SVM = dSVM.predict(df)
  st.subheader('Prediction RF Probability')
  st.write(iris.target_names[prediction_SVM])