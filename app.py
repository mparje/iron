import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('iron_price_model.pkl')

# Define the Streamlit app
def main():
    # Set the title and description
    st.title("Iron Price Prediction")
    st.write("This app predicts the price of iron based on input features.")

    # Create input fields for the features
    ri = st.number_input("RI (Refractive Index)")
    depth = st.number_input("Depth Percentage")
    width = st.number_input("Width of Iron Top Width")
    axis_x = st.number_input("Axis X")
    axis_y = st.number_input("Axis Y")
    axis_z = st.number_input("Axis Z")

    # Create a dataframe with the input features
    input_data = pd.DataFrame({
        'RI': [ri],
        'Depth-percentage': [depth],
        'Width-of-iron-top-width': [width],
        'Axis-x': [axis_x],
        'Axis-y': [axis_y],
        'Axis-z': [axis_z]
    })

    # Make predictions using the trained model
    price_prediction = model.predict(input_data)

    # Display the predicted price
    st.subheader("Price Prediction")
    st.write("The predicted price of iron is $", price_prediction[0])

# Run the app
if __name__ == '__main__':
    main()
