#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

#load the model from disk
import joblib
model = joblib.load(r"C:\Users\selas\OneDrive\Desktop\capstone project\model.sav")

#Import python scripts
from sklearn import preprocessing


def main():
    #Setting Application title
    st.title('Customer Churn Prediction App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a telecommunication use case.
    The application is functional for both online prediction and batch data prediction. n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('app.jpg')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        
        st.subheader("Data")

        TENURE = st.slider('duration in the network', min_value=0, max_value=7, value=0)
        MONTANT = st.number_input('top-up amount', min_value=20, max_value=470000, value=20 )
        FREQUENCE_RECH = st.number_input('A number of times the customer refilled', min_value=1, max_value=131, value=1 )
        REVENUE = st.number_input('monthly income of each client', min_value=1, max_value=532177, value=1)
        ARPU_SEGMENT = st.number_input('income over 90 days / 3',min_value=0, max_value=177392, value=0)

        FREQUENCE = st.number_input("number of times the client has made an income", min_value=1, max_value=91, value=1 )
        DATA_VOLUME = st.number_input('number of connections', min_value=0, max_value=1702309, value=0 )
        ON_NET = st.number_input("inter expresso call",min_value=0, max_value=50809, value=0 )
        ORANGE = st.number_input("call to orange", min_value=0, max_value=12040, value=0 )
        TIGO = st.number_input("call to Tigo",min_value=0, max_value=4174, value=0 )
        ZONE1 = st.number_input("call to zones1", min_value=0, max_value=2507, value=0 )
        ZONE2 = st.number_input("call to zones2", min_value=0, max_value=3697, value=0 )
        #MRG = st.number_input('a client who is going', min_value=)
        REGULARITY = st.number_input('number of times the client is active for 90 days', min_value=1, max_value=62, value=1 )
        TOP_PACK = st.number_input('the most active packs', min_value=1, max_value=624, value=1)
        FREQ_TOP_PACK = st.number_input('number of times the client has activated the top pack packages', min_value=1, max_value=624, value=1 )



        data = {
                'MONTANT': MONTANT,
                'FREQUENCE_RECH': FREQUENCE_RECH,
                'TENURE':TENURE,
                'REVENUE': REVENUE,
                'ARPU_SEGMENT': ARPU_SEGMENT,
                'FREQUENCE': FREQUENCE,
                'DATA_VOLUME': DATA_VOLUME,
                'ON_NET': ON_NET,
                'ORANGE': ORANGE,
                'TIGO': TIGO,
                'ZONE1': ZONE1,
                'ZONE2': ZONE2,
                #'MRG': MRG,
                'REGULARITY':REGULARITY,
                'TOP_PACK': TOP_PACK,
                'FREQ_TOP_PACK ': FREQ_TOP_PACK  
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        #Preprocess inputs
        preprocess_df = preprocessing(features_df, 'Online')

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer will terminate the service.')
            else:
                st.success('No, the customer is happy with the Services.')


    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocessing(data, "Batch")
            if st.button('Predict'):
                #Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the customer will terminate the service.',
                                                    0:'No, the customer is happy with the Services.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)

if __name__ == '__main__':
        main()