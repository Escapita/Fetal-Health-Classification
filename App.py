import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

def predicts(data):
    clf = joblib.load("svc_model.pkl")
  #  full_pipeline = fetch_pipeline()
   # data_prepared = full_pipeline.transform(data)
    return clf.predict(data)

def fetch_pipeline():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])

    housing = pd.read_csv("datasets/fetal_health.csv")
    housing = housing.drop("median_house_value", axis=1)
    
    housing_num = housing.drop("ocean_proximity", axis=1)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])

    full_pipeline.fit(housing)

    return full_pipeline

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
  def __init__(self, add_bedrooms_per_room=True):
    self.add_bedrooms_per_room=add_bedrooms_per_room
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, X):
    rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
    population_per_household = X[:, population_ix] / X[:, households_ix]
    if self.add_bedrooms_per_room:
      bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
      return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
    else:
      return np.c_[X, rooms_per_household, population_per_household]

st.title("Fetal Health Classification Model")

baseline_value = st.number_input("Baseline Value", max_value = 160.0)
accelerations = st.number_input("Accelerations", max_value = 0.019)
fetal_movement = st.number_input("Fetal Movement", max_value = 0.481)
uterine_contractions = st.number_input("Uterine Contractions", max_value = 0.015)
light_decelerations = st.number_input("Light Decelerations", max_value = 0.015)
severe_decelerations = st.number_input("Severe Decelerations", max_value = 0.001)
prolongued_decelerations = st.number_input("Prolongued Decelerations", max_value = 0.005)
abnormal_short_term_variability = st.number_input("Abnormal Short Term Variability", min_value = 12.0, max_value = 87.0)
mean_value_of_short_term_variability = st.number_input("Mean Value of Short Term Variability", max_value = 7.0)
percentage_of_time_with_abnormal_long_term_variability = st.number_input("Percentage of Time with Abnormal Long Term Variability", max_value = 91.0)
mean_value_of_long_term_variability = st.number_input("Mean Value of Long Term Variability", max_value = 50.7)
histogram_width = st.number_input("Histogram Width", min_value = 3.0)
histogram_min = st.number_input("Histogram Min", min_value = 50.0)
histogram_max = st.number_input("Histogram Max", min_value = 122.0)
histogram_number_of_peaks = st.number_input("Histogram Number of Peaks", max_value = 18.0)
histogram_number_of_zeroes = st.number_input("Histogram Number of Zeroes", max_value = 10.0)
histogram_mode = st.number_input("Histogram Mode", min_value = 60.0)
histogram_mean = st.number_input("Histogram Mean", min_value = 73.0)
histogram_median = st.number_input("Histogram Median", min_value = 77.0)
histogram_variance = st.number_input("Histogram Variance", max_value = 269.0)
histogram_tendency = st.number_input("Histogram Tendency", min_value = -1.0, max_value=1.0)



if st.button("Classify Fetal Health"):  
    data = pd.DataFrame({
            'baseline value': [baseline_value],
            'accelerations': [accelerations],
            'fetal_movement': [fetal_movement],
            'uterine_contractions': [uterine_contractions],
            'light_decelerations': [light_decelerations],
            'severe_decelerations': [severe_decelerations],
            'prolongued_decelerations': [prolongued_decelerations],
            'abnormal_short_term_variability': [abnormal_short_term_variability],
            'mean_value_of_short_term_variability': [mean_value_of_short_term_variability],
            'percentage_of_time_with_abnormal_long_term_variability': [percentage_of_time_with_abnormal_long_term_variability],
            'mean_value_of_long_term_variability': [mean_value_of_long_term_variability],
            'histogram_width': [histogram_width],
            'histogram_min': [histogram_min],
            'histogram_max': [histogram_max],
            'histogram_number_of_peaks': [histogram_number_of_peaks],
            'histogram_number_of_zeroes': [histogram_number_of_zeroes],
            'histogram_mode': [histogram_mode],
            'histogram_mean': [histogram_mean],
            'histogram_median': [histogram_median],
            'histogram_variance': [histogram_variance],
            'histogram_tendency': [histogram_tendency]
        }) 
    result = predicts(data)
    
    def health(argument):
        switcher = {
            1: "Normal",
            2: "Suspect",
            3: "Pathological",
        }
        return switcher.get(int(argument), 'ERROR')

    st.text(health(result[0])) 
    
