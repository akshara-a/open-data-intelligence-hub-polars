import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="BlinkIT Recommendation System",
    layout="wide"
)

st.title("🛒 BlinkIT Recommendation System")

st.write("AI/ML Internship Project")

df = pd.read_csv("data/final_dataset.csv")

classification_model = joblib.load("models/final_classification_model.pkl")

regression_model = joblib.load("models/regression_model.pkl")

kmeans = joblib.load("models/kmeans_model.pkl")

scaler = joblib.load("models/scaler.pkl")

page = st.sidebar.selectbox(

    "Select Page",

    [

        "Home",

        "Dataset",

        "Visualizations",

        "Sales Prediction",

        "Classification",

        "Product Segmentation",

        "Recommendation",

        "Model Comparison"

    ]

)

if page=="Home":

    st.title("BlinkIT Recommendation System")

    st.write("""
This application predicts product sales,
classifies products,
segments products,
and recommends similar products.
""")
    
if page=="Dataset":

    st.title("Dataset")

    st.dataframe(df)

    st.write("Rows:",df.shape[0])

    st.write("Columns:",df.shape[1])

if page=="Visualizations":

    fig,ax=plt.subplots()

    sns.histplot(df["Sales"],kde=True)

    st.pyplot(fig)

rating = st.number_input("Rating")

visibility = st.number_input("Item Visibility")

weight = st.number_input("Item Weight")

if st.button("Predict Sales"):

    input_data=np.array([[

        visibility,

        weight,

        rating

    ]])

    prediction=regression_model.predict(input_data)

    st.success(f"Predicted Sales : {prediction[0]:.2f}")

if page=="Classification":

    rating=st.number_input("Rating")

    visibility=st.number_input("Visibility")

    weight=st.number_input("Weight")

    if st.button("Predict"):

        prediction=classification_model.predict(

            np.array([[

                visibility,

                weight,

                rating

            ]])

        )

        if prediction[0]==1:

            st.success("High Sales Product")

        else:

            st.error("Low Sales Product")

if page=="Product Segmentation":

    rating=st.number_input("Rating ")

    visibility=st.number_input("Visibility ")

    weight=st.number_input("Weight ")

    if st.button("Find Cluster"):

        cluster=kmeans.predict(

            np.array([[

                visibility,

                weight,

                rating

            ]])

        )

        st.success(f"Cluster : {cluster[0]}")

similarity_matrix=joblib.load(
    "models/similarity_matrix.pkl"
)

product_index=joblib.load(
    "models/product_index.pkl"
)

def recommend_products(product_name, top_n=5):
    if product_name not in product_index:
        return pd.DataFrame(columns=["Item Identifier", "Score"])
    idx = product_index[product_name]
    scores = similarity_matrix[idx]
    similar_indices = np.argsort(scores)[::-1]
    similar_indices = similar_indices[similar_indices != idx][:top_n]
    recommendations = df.loc[similar_indices, ["Item Identifier"]].copy()
    recommendations["Score"] = scores[similar_indices]
    return recommendations

selected_product=st.selectbox(

    "Select Product",

    df["Item Identifier"]

)

if st.button("Recommend"):

    recommendations=recommend_products(selected_product)

    st.dataframe(recommendations)

regression=pd.read_csv(
    "data/regression_results.csv"
)

classification=pd.read_csv(
    "data/classification_results.csv"
)

regression=pd.read_csv(
    "data/regression_results.csv"
)

classification=pd.read_csv(
    "data/classification_results.csv"
)

st.markdown("---")

st.write("Developed for AI/ML Internship")

