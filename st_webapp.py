# Importing libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error
import os
import sys

# Title
st.title('Regression WebApp')
st.warning('Please note that it is only a simple regression application')
st.write('It only considers numerical columns, hence you have to encode the categorical values mannually and then enter the encoded data here')


def main():
    st.info(__doc__)
    st.markdown(STYLE, unsafe_allow_html = True)
    file = st.file_uploader("upload csv file", type = ["csv"])
    show_file = st.empty()
    if not file:
        show_file.info("Please upload a file: {}" .format(["csv"]))
        return
    content = file.getvalue()
    
    df = pd.read_csv(file)
    st.write('Data:')
    st.dataframe(df)
    file.close()
    # Name of the data file
    #df = st.text_input('Enter the .csv file with extension')
    #st.write('Enter a valid file name')

    # Regression models and web app design
    try:
    
        # Read data
        #df = pd.read_csv(df)
        

        s2 = st.sidebar.selectbox('Select Y',df.columns)
        s3 = st.sidebar.multiselect('Select X',df.columns)
        st.write('Final data')
        st.write(df[s3])
        s4 = st.sidebar.slider('Enter test size', min_value = 0.1, value = 0.25, step = 0.05)
        x_train, x_test, y_train, y_test = train_test_split(df[s3],df[s2],test_size = s4,random_state = 0)
        metrics = st.sidebar.selectbox('select evaluation metrics',['r2 score','mse'])
        s5 = st.sidebar.selectbox('Choose regression model',['None','Linear','Random Forest','Decision Tree'])

        # Clean Data
        x_train = x_train.apply(pd.to_numeric, errors='coerce')
        y_train = y_train.apply(pd.to_numeric, errors='coerce')
        x_test = x_test.apply(pd.to_numeric, errors='coerce')
        x_train.fillna(0, inplace=True)
        y_train.fillna(0, inplace=True)
        x_test.fillna(0, inplace=True)
        st.write('Plotting Output (Y)')
        st.write(st.line_chart(y_train))
    
        # Models
        if s5 == 'Linear':
            reg = LinearRegression()
            reg.fit(x_train,y_train)
            y_pred = reg.predict(x_test)
            st.write('Predicted Values')
            st.write(pd.DataFrame({'Actual':y_test,'Predicted':y_pred}))
        if s5 == 'Random Forest':
            reg = RandomForestRegressor()
            reg.fit(x_train,y_train)
            y_pred = reg.predict(x_test)
            st.write('Predicted Values')
            st.write(pd.DataFrame({'Actual':y_test,'Predicted':y_pred}))
        if s5 == 'Decision Tree':
            reg = DecisionTreeRegressor()
            reg.fit(x_train,y_train)
            y_pred = reg.predict(x_test)
            st.write('Predicted Values')
            st.write(pd.DataFrame({'Actual':y_test,'Predicted':y_pred}))
        # Evaluation Metrics
        if metrics == 'r2 score':
            st.write('r2 score: ')
            st.write(r2_score(y_test,y_pred))
        if metrics == 'mse':
            st.write('Mean squared error: ')
            st.write(mean_squared_error(y_test,y_pred))
                 
    except:
        st.write('ERROR')
        #pass

main()







                          
