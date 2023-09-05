import streamlit as st
import tensorflow as tf
import numpy as np

st.title("Drowsiness detection")
st.write("Predict whether a person is feeling drowsy or not")

model = tf.keras.models.load_model("drowsiness.h5")
labels = ['Closed', 'No yawn', 'Open', 'Yawn']

# Define input fields for the three numerical inputs
input1 = st.number_input("Enter value 1", value=0)
input2 = st.number_input("Enter value 2", value=0)
input3 = st.number_input("Enter value 3", value=0)

# Define a function to preprocess the input values and make a prediction
def predict(input1, input2, input3):
    # Preprocess the input values
    inputs = np.array([input1, input2, input3]).reshape(1, -1)

    # Make a prediction using the loaded model
    prediction = model.predict(inputs)
    predicted_label_index = np.argmax(prediction)
    label = labels[predicted_label_index]

    return label

# Call the predict function when the user clicks the "Predict" button
if st.button("Predict"):
    label = predict(input1, input2, input3)
    st.write("### Prediction Result")
    st.write(f"The person is feeling {label}.")

# Display a sample image and prediction if the user clicks the "Use Sample Image" button
if st.button("Use Sample Image"):
    input1, input2, input3 = 1, 2, 3  # Replace with your own sample inputs
    label = predict(input1, input2, input3)
    st.write("### Prediction Result")
    st.write(f"The person is feeling {label}.")
