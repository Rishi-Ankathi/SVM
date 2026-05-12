import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('loan.csv')

st.title('SVC Classifier for Loan Approval Prediction')
st.subheader('Dataset Overview')
st.write(df.head())
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

x = df.drop('loan_status', axis=1)
y = df['loan_status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
svc = SVC(kernel = 'rbf',
            C=1,
            gamma='scale')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)

st.subheader('Model Evaluation')
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')