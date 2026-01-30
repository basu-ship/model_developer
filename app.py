import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = "iris_model.pkl"

    if not os.path.exists(model_path):
        st.error("❌ Model file 'iris_model.pkl' not found.")
        return None

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error("❌ Failed to load model.")
        st.exception(e)
        return None


model = load_model()

# ---------------- UI ----------------
st.title("🌸 Iris Species Predictor")

st.markdown("""
This application uses a **Logistic Regression** model  
to predict the **species of an Iris flower** based on its measurements.
""")

st.sidebar.header("🌿 Input Floral Features")

def get_user_input():
    sepal_length = st.sidebar.slider(
        "Sepal Length (cm)", 4.0, 8.0, 5.8
    )
    sepal_width = st.sidebar.slider(
        "Sepal Width (cm)", 2.0, 4.5, 3.0
    )
    petal_length = st.sidebar.slider(
        "Petal Length (cm)", 1.0, 7.0, 4.3
    )
    petal_width = st.sidebar.slider(
        "Petal Width (cm)", 0.1, 2.5, 1.3
    )

    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

input_data = get_user_input()

# ---------------- PREDICTION ----------------
if model is not None:

    target_names = ["Setosa", "Versicolor", "Virginica"]

    if st.button("🔍 Predict Species"):
        try:
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)

            species = target_names[prediction[0]]

            st.success(f"### 🌼 Predicted Species: {species}")

            st.subheader("📊 Prediction Probabilities")
            prob_df = pd.DataFrame(
                probability,
                columns=target_names
            )
            st.bar_chart(prob_df.T)

        except Exception as e:
            st.error("❌ Prediction failed.")
            st.exception(e)

else:
    st.warning("⚠️ Upload `iris_model.pkl` to enable predictions.")
