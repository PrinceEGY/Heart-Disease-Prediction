import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import joblib

DATASET_PATH = 'Dataset/Cleaned.csv'

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="♥",
    # layout='wide'
)


@st.cache(persist=True)
def load_dataset() -> pd.DataFrame:
    heart_df = pd.read_csv(DATASET_PATH, index_col=0)
    return heart_df


def user_input_features() -> pd.DataFrame:
    df = load_dataset()
    st.markdown("""
                <style>
                div {
                    font-size:1.01em !important;
                }
                label, input, textarea {
                    font-size:1.2em !important;
                }
                button{
                    font-size:1.8em !important;
                    font-weight: bold !important;
                }
                </style>
        """, unsafe_allow_html=True)
    race = st.sidebar.selectbox("Race", options=(
        race for race in df.Race.unique()))
    sex = st.sidebar.selectbox("Gender", options=(
        sex for sex in df.Sex.unique()))
    ages = [age_cat for age_cat in df.AgeCategory.unique()]
    ages.sort()
    age_cat = st.sidebar.selectbox("Age category",
                                   options=(ages))
    bmi_cat = st.sidebar.selectbox("BMI category",
                                   options=(bmi_cat for bmi_cat in df.BMICategory.unique()))
    sleep_time = st.sidebar.number_input(
        "Average Sleep hours", 0, 24, 7)
    gen_health = st.sidebar.selectbox("General health?",
                                      options=(gen_health for gen_health in df.GenHealth.unique()))
    phys_health = st.sidebar.number_input("For how many days during the past 30 days was"
                                          " your physical health not good?", 0, 30, 0)
    ment_health = st.sidebar.number_input("For how many days during the past 30 days was"
                                          " your mental health not good?", 0, 30, 0)
    phys_act = st.sidebar.selectbox("Have you played any sports (running, biking, etc.)"
                                    " in the past month?", options=("No", "Yes"))
    smoking = st.sidebar.selectbox("Have you smoked at least 100 cigarettes in"
                                   " your entire life (approx. 5 packs)?)",
                                   options=("No", "Yes"))
    alcohol_drink = st.sidebar.selectbox("Do you have more than 14 drinks of alcohol (men)"
                                         " or more than 7 (women) in a week?", options=("No", "Yes"))
    stroke = st.sidebar.selectbox(
        "Did you have a stroke?", options=("No", "Yes"))
    diff_walk = st.sidebar.selectbox("Do you have serious difficulty walking"
                                     " or climbing stairs?", options=("No", "Yes"))
    diabetic = st.sidebar.selectbox("Have you ever had diabetes?",
                                    options=(diabetic for diabetic in df.Diabetic.unique()))
    asthma = st.sidebar.selectbox("Do you have asthma?", options=("No", "Yes"))
    kid_dis = st.sidebar.selectbox(
        "Do you have kidney disease?", options=("No", "Yes"))
    skin_canc = st.sidebar.selectbox(
        "Do you have skin cancer?", options=("No", "Yes"))

    features = pd.DataFrame({
        "PhysicalHealth": [phys_health],
        "MentalHealth": [ment_health],
        "SleepTime": [sleep_time],
        "BMICategory": [bmi_cat],
        "Smoking": [smoking],
        "AlcoholDrinking": [alcohol_drink],
        "Stroke": [stroke],
        "DiffWalking": [diff_walk],
        "Sex": [sex],
        "AgeCategory": [age_cat],
        "Race": [race],
        "Diabetic": [diabetic],
        "PhysicalActivity": [phys_act],
        "GenHealth": [gen_health],
        "Asthma": [asthma],
        "KidneyDisease": [kid_dis],
        "SkinCancer": [skin_canc]
    })

    return features


def get_result(input_df):
    model = joblib.load('Preprocessing/XGB.pkl')
    df = load_dataset()
    df = pd.concat([df, input_df], axis=0)
    # encoding
    order_cols = ["BMICategory", "AgeCategory"]
    no_order_cols = ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                     "Sex", "Race", "Diabetic", "PhysicalActivity",
                     "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
    # Label encoding
    for col in order_cols:
        df[col] = preprocessing.LabelEncoder().fit_transform(df[col])

    # One-hot encoding
    for col in no_order_cols:
        dummy_col = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy_col], axis=1)
        del df[col]
    df.drop('HeartDisease', axis=1, inplace=True)
    return model.predict_proba(df[-1:])


with st.sidebar:
    with st.expander('Tool authors'):
        st.markdown("""
        **Authors: [Ahmed Mohsen](https://www.linkedin.com/in/AhmedMohsen-), [Hossam Galal](https://www.linkedin.com/in/hossam-galal-b817bb197/), [Yomna Ramdan]()**

        You can see the steps of building the model, evaluating it, and cleaning the data itself on GitHub repo [here](https://github.com/PrinceEGY/LinkedIn-Job-Scraper).
        """)
st.title('Heart Disease Prediction')
st.subheader(
    'Are you wondering about the condition of your heart? This app will help you to diagnose it!')
cols = st.columns([1, 3])
with cols[0]:
    st.image("images/doctor.png")
with cols[1]:
    st.markdown('''
    This application will help you to know the probability of having heart disease with accuracy about 99%

    *Keep in mind that this results is not equivalent to a medical diagnosis!
    Doctors or patients CANNOT fully rely on it, but it can be used as an aid to confirm the diagnosis, so if you have any problems, consult a human doctor.*

    To predict your heart disease status, simply follow the steps bellow:

    1. Enter the parameters that best describe you on the left side bar
    2. Press the button below and wait for the result.
    ''')
col = st.columns([2, 5, 1])
submit = False
with col[1]:
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        width:450px;
        height:75px;
        font-size:1.5em !important;
    }
    </style>""", unsafe_allow_html=True)
    submit = st.button('Check my condition')
gradient_color = ['#FF0000', '#FF1100', '#FF2300', '#FF3400', '#FF4600',
                  '#FF5700', '#FF6900', '#FF7B00', '#FF8C00', '#FF9E00', '#FFAF00', '#FFC100',
                  '#FFD300', '#FFE400', '#FFF600', '#F7FF00', '#C2FF00', '#58FF00', '#12FF00', '#00FF00'
                  ]


st.sidebar.title("Personal Key Indicators")
st.sidebar.image("images/heartbeat.png", width=300)
input_df = user_input_features()


if submit:
    pred = round(get_result(input_df)[0][1]*100, 2)
    st.markdown("""
    <style>
    strong {
        font-size:1.6em !important;
        color:%s;
        text-indent: 50px;
    }
    em{
        font-size:1.5em !important;
        font-style:normal;
        font-weight: bold;
        word-spacing: 2px;
    }
    </style>
    """ % (gradient_color[19-int(pred//5.1)]), unsafe_allow_html=True)
    st.markdown(
        "_The probability that you will have heart disease is_&nbsp;&nbsp; **{0}%**".format(pred))
