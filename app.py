import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
dec_tree=DecisionTreeClassifier()
#from sklearn.ensemble import  GradientBoostingRegressor
#gr=GradientBoostingRegressor( loss='huber')
df=pd.read_csv("diabetes_data.csv",delimiter=';')
df = pd.get_dummies(df)
x=df[["age","polyuria","polydipsia","sudden_weight_loss","weakness","polyphagia","genital_thrush","visual_blurring","itching","delayed_healing","partial_paresis","muscle_stiffness","alopecia","obesity","gender_Female","gender_Male"]]
y=df["class"]
dec_tree.fit(x,y)
import joblib
joblib.dump(dec_tree,"dec1.pkl")

# Title
st.header("Diabetes Machine Learning App")

age=st.number_input("Enter Age")
polyuria = st.number_input("Enter Polyuria")
polydipsia = st.number_input("Enter polydipsia")
sudden_weight_loss = st.number_input("Enter sudden_weight_loss")
weekness = st.number_input("Enter weekness")
polyphagia = st.number_input("Enter polyphagia")
genital_thrush = st.number_input("Enter genital_thrush")
visual_blurring = st.number_input("Enter visual_blurring")
itching = st.number_input("Enter itching")
delayed_healing = st.number_input("Enter delayed_healing")
partial_paresis = st.number_input("Enter partial_paresis")
muscle_stiffness = st.number_input("Enter muscle_stiffness")
alopecia = st.number_input("Enter alopecia")
obesity = st.number_input("Enter obesity")
gender_Female = st.number_input("Enter gender_female")
gender_Male = st.number_input("Enter Gender_male")


# If button is pressed
if st.button("Submit"):
    # Unpickle classifier
    clf = joblib.load("dec1.pkl")

    # Store inputs into dataframe
    X = pd.DataFrame([[age,polyuria,polydipsia,sudden_weight_loss,weekness,polyphagia,genital_thrush,visual_blurring,itching,delayed_healing,partial_paresis,muscle_stiffness,alopecia,obesity,gender_Female,gender_Male]],
                     columns=["age","polyuria","polydipsia","sudden_weight_loss","weekness","polyphagia","genital_thrush","visual_blurring","itching","delayed_healing","partial_paresis","muscle_stiffness","alopecia","obesity","gender_Female","gender_Male"])

    # Get prediction
    prediction = clf.predict(X)[0]

    # Output prediction
    st.text(f"This instance is a {prediction}")
