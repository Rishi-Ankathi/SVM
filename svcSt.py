import streamlit as st
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# LOAD DATA
df = pd.read_csv('loan.csv')

st.title('SVC Classifier for Loan Approval Prediction')

st.subheader('Dataset Overview')
st.write(df.head())

# REMOVE SPACES FROM COLUMN NAMES
df.columns = df.columns.str.strip()

# STORE ENCODERS
label_encoders = {}

# ENCODE ALL OBJECT COLUMNS
for column in df.columns:

    if df[column].dtype == 'object':

        # convert everything to string
        df[column] = df[column].astype(str)

        le = LabelEncoder()

        df[column] = le.fit_transform(df[column])

        label_encoders[column] = le

# FEATURES AND TARGET
x = df.drop('loan_status', axis=1)
y = df['loan_status']

# TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# MODEL
svc = SVC(
    kernel='rbf',
    C=1,
    gamma='scale'
)

# TRAIN MODEL
svc.fit(x_train, y_train)

# PREDICTIONS
y_pred = svc.predict(x_test)

# ACCURACY
accuracy = accuracy_score(y_test, y_pred)

st.subheader('Model Evaluation')
st.write(f'Accuracy: {accuracy:.2f}')