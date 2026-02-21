import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
st.title("Iris flower classification project")

data = load_iris()
y = data['target']
X = pd.DataFrame(data['data'], columns = data['feature_names'])
target_class = data['target_names']

st.sidebar.title("Select Iris Feature")

user_input = []

for i in X:
  min_value = X[i].min()
  max_value = X[i].max()

  ans = st.sidebar.slider(f'Select value of {i}', min_value, max_value)
  user_input.append(ans)

final_input = [user_input]

import pickle

with open('iris_model.pkl', 'rb') as f:
  chatgpt = pickle.load(f)

final_ans = chatgpt.predict(final_input)[0]

flower_name = target_class[final_ans]

prob = chatgpt.predict_proba(final_input).ravel()


for i, j in enumerate(prob):
  flower = target_class[i]
  st.write(f'Probability of {flower} is : {round(j*100, 2)}')

st.success(f'The final predicted flower is {flower_name}')
