#run with streamlit run file.py

#Description : This program detecs if someone has diabetes using machine learning and python !

#Import the librarys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image 
import streamlit as st

#create a title and subtitle
st.write("""
Breast Cancer Detection 
using machine learning and python Detect if someone has Breast cancer
""")
#open and display an image
image = Image.open('C:/Users/Yogesh/Desktop/ML Algorithm/Ai production/Breast cancer/streamlite app/Indian AI Hospital.png')
st.image(image, caption='ML', use_column_width=True)

#Get the data
df = pd.read_csv('C:/Users/Yogesh/Desktop/ML Algorithm/Ai production/Breast cancer/streamlite app/bs.csv')
#Set a subheader
st.subheader('Data Information:')
#Show the data as a table
st.dataframe(df)
#Show statistics on the data
st.write(df.describe())
#show the data as a chart
chart = st.bar_chart(df)

#split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:30].values
Y = df.iloc[:,-1].values
#split the data set into 75% Training and 25% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#Get the features input from the user
def get_user_input():
    mean_radius = st.sidebar.slider('meanradius', 6.981, 28.11, 14.127)
    mean_texture = st.sidebar.slider('meantexture', 9.71, 39.28, 19.289)
    mean_perimeter = st.sidebar.slider('meanperimeter', 43.79, 188.5, 91.969)
    mean_area = st.sidebar.slider('meanarea', 143.5, 2501.0, 654.889)
    mean_smoothness = st.sidebar.slider('meansmoothness', 0.05263, 0.1634, 0.09636)
    mean_compactness = st.sidebar.slider('meancompactness', 0.01938, 0.3454, 0.104341)
    mean_concavity = st.sidebar.slider('meanconcavity', 0.0, 0.4268, 0.088799)
    mean_concave_points = st.sidebar.slider('meanconcavepoints', 0.0, 0.2012, 0.048919)
    mean_symmetry = st.sidebar.slider('meansymmetry', 0.106, 0.304, 0.181162)
    mean_fractal_dimension = st.sidebar.slider('meanfractaldimension', 0.04996, 0.09744, 0.062798)
    radius_error = st.sidebar.slider('radiuserror', 0.1115, 2.873, 0.405172)
    texture_error = st.sidebar.slider('textureerror', 0.3602, 4.885, 1.216853)
    perimeter_error = st.sidebar.slider('perimetererror', 0.757, 21.98, 2.866059)
    area_error = st.sidebar.slider('areaerror', 6.802, 542.2, 40.33708)
    smoothness_error = st.sidebar.slider('smoothnesserror', 0.001713, 0.03113, 0.007041)
    compactness_error = st.sidebar.slider('compactnesserror', 0.002252, 0.1354, 0.025478)
    concavity_error = st.sidebar.slider('concavityerror', 0.0, 0.396, 0.031894)
    concave_points_error = st.sidebar.slider('concavepointserror', 0.0, 0.05279, 0.011796)
    symmetry_error = st.sidebar.slider('symmetryerror', 0.007882, 0.07895, 0.020542)
    fractal_dimension_error = st.sidebar.slider('fractaldimensionerror', 0.000895, 0.02984, 0.003795)
    worst_radius = st.sidebar.slider('worstradius', 7.93, 36.04, 16.26919)
    worst_texture = st.sidebar.slider('worsttexture', 12.02, 49.54, 25.67722)
    worst_perimeter = st.sidebar.slider('worstperimeter', 50.41, 251.2, 107.2612)
    worst_area = st.sidebar.slider('worstarea', 185.2, 4254.0, 880.5831)
    worst_smoothness = st.sidebar.slider('worstsmoothness', 0.07117, 0.2226, 0.132369)
    worst_compactness = st.sidebar.slider('worstcompactness', 0.02729, 1.058, 0.254265)
    worst_concavity = st.sidebar.slider('worstconcavity', 0.0, 1.252, 0.272188)
    worst_concave_points = st.sidebar.slider('worstconcave points', 0.0, 0.291, 0.114606)
    worst_symmetry = st.sidebar.slider('worstsymmetry', 0.1565, 0.6638, 0.290076)
    worst_fractal_dimension = st.sidebar.slider('worstfractaldimension', 0.05504, 0.2075, 0.083946)
    
    
    #store a dictonary into a variables
    user_data = {'mean_radius': mean_radius,
                 'mean_texture': mean_texture,
                 'mean_perimeter': mean_perimeter,
                 'mean_area': mean_area,
                 'mean_smoothness': mean_smoothness,
                 'mean_compactness': mean_compactness,
                 'mean_concavity': mean_concavity,
                 'mean_concave_points': mean_concave_points,
                 'mean_symmetry': mean_symmetry,
                 'mean_fractal_dimension': mean_fractal_dimension,
                 'radius_error': radius_error,
                 'texture_error': texture_error,
                 'perimeter_error': perimeter_error,
                 'area_error': area_error,
                 'smoothness_error': smoothness_error,
                 'compactness_error': compactness_error,
                 'concavity_error': concavity_error,
                 'concave_points_error': concave_points_error,
                 'symmetry_error': symmetry_error,
                 'fractal_dimension_error': fractal_dimension_error,
                 'worst_radius': worst_radius,
                 'worst_texture': worst_texture,
                 'worst_perimeter': worst_perimeter,
                 'worst_area': worst_area,
                 'worst_smoothness': worst_smoothness,
                 'worst_compactness': worst_compactness,
                 'worst_concavity': worst_concavity,
                 'worst_concave_points': worst_concave_points,
                 'worst_symmetry': worst_symmetry,
                 'worst_fractal_dimension': worst_fractal_dimension,
                }
    #Transform the data into a data frame
    features = pd.DataFrame(user_data, index = [0])
    return features
    
#store the user input into a variables
user_input = get_user_input()

#set a subheader and display the users input
st.subheader('User Input:')
st.write(user_input)

#Create and train the model_selection
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#show the models matrics
st.subheader('Model Test Accuracy Score:')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))* 100)+'%' )

#Store the model prediction in a variable
prediction = RandomForestClassifier.predict(user_input)

#set a subheader and display the classifiocation
st.subheader('Classification: ')
st.write(prediction)

