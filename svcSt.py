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

# STORE ENCODERS
label_encoders = {}

# ENCODE CATEGORICAL COLUMNS
for column in df.columns:

    if df[column].dtype == 'object':

        le = LabelEncoder()

        df[column] = le.fit_transform(df[column])

        # SAVE ENCODER
        label_encoders[column] = le

# FEATURES AND TARGET
x = df.drop('loan_status', axis=1)
y = df['loan_status']

# SPLIT DATA
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# TRAIN MODEL
svc = SVC(
    kernel='rbf',
    C=1,
    gamma='scale'
)

svc.fit(x_train, y_train)

# PREDICTION
y_pred = svc.predict(x_test)

# ACCURACY
accuracy = accuracy_score(y_test, y_pred)

st.subheader('Model Evaluation')
st.write(f'Accuracy: {accuracy:.2f}')