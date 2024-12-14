import pandas as pd
# to make the frontend 
import streamlit as st
# to load the dataset of the iris
from sklearn.datasets import load_iris
# to load the model 
from sklearn.ensemble import RandomForestClassifier


@st.cache_data


# function to load the data and separate the target and features
def load_data():
    iris = load_iris()
    df= pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species']=iris.target
    return df,iris.target_names


# load the data and the model
df,target_name=load_data()
model=RandomForestClassifier()


# train the model (features , target)
model.fit(df.iloc[:,:-1],df['species'])


st.sidebar.title("Input Features")
sepal_length= st.sidebar.slider("Sepal length :",float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max()))
sepal_width= st.sidebar.slider("Sepal width :",float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()))
petal_length= st.sidebar.slider("petal length :",float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))
petal_width= st.sidebar.slider("petal width :",float(df['petal width (cm)'].min()),float(df['petal width (cm)'].max()))



input_data=[[sepal_length,sepal_width,petal_length,petal_width]]


# predicting the data and give the output
prediction = model .predict(input_data)
predicted_species=target_name[prediction[0]]



# showing the output
st.write("Prediction")
st.write("The predicted value is :",predicted_species)