import streamlit as st
import tensorflow as tf
import numpy as np

import streamlit as st
import tensorflow as tf
import numpy as np

# Configure the Streamlit app
st.set_page_config(
    page_title="Sales Prediction",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Set the theme for the app
st.beta_set_theme(
    "dark",
    bgColor="black",
    textColor="orange",
)

st.title("Sales Prediction")

st.title("Sales Prediction")


# Load the saved ANN model
model = tf.keras.models.load_model("1.h5")

# Define the input fields
country = st.number_input("Enter the country code:", value=0)
store = st.number_input("Enter the store code:", value=0)
product = st.number_input("Enter the product code:", value=0)

# Define a function to preprocess the input values and make a prediction
def predict_sales(country, store, product):
    # Preprocess the input values
    inputs = np.array([country, store, product]).reshape(1, -1)

    # Make a prediction using the loaded model
    prediction = model.predict(inputs)

    return prediction[0][0]

# Call the predict_sales function when the user clicks the "Predict" button
if st.button("Predict"):
    prediction = predict_sales(country, store, product)
    st.write("### Prediction Result")
    st.write(f"The predicted sales for this product is {prediction:.2f}.")

# Display a sample prediction if the user clicks the "Use Sample Data" button
if st.button("Use Sample Data"):
    country, store, product = 0, 0, 0  # Replace with your own sample inputs
    prediction = predict_sales(country, store, product)
    st.write("### Prediction Result")
    st.write(f"The predicted sales for this product is {prediction:.2f}.")
