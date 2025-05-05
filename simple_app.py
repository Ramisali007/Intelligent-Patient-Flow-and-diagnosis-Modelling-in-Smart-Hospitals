import streamlit as st
import pandas as pd
import numpy as np

st.title("Simple Streamlit App")

st.write("This is a simple Streamlit app to test if Streamlit is working correctly.")

# Create a simple dataframe
df = pd.DataFrame({
    'A': np.random.randn(10),
    'B': np.random.randn(10),
    'C': np.random.randn(10)
})

st.write("Here's a simple dataframe:")
st.dataframe(df)

st.write("And here's a simple chart:")
st.line_chart(df)

st.write("If you can see this, Streamlit is working correctly!")
