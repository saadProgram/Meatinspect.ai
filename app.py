import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import img_to_array
from PIL import Image
import os

# -----------------------------#
#        Custom CSS Styling    #
# -----------------------------#
st.markdown(
    """
    <style>
    /* Background color */
    .stApp {
        background-color: #f9f9f9;  /* Light background color */
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f9f9f9;  /* Same as main content background */
        color: #AA3C3B;  /* Text color in sidebar */
    }

    /* Sidebar button styling */
    .stButton>button {
        background-color: #008080;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        cursor: pointer;
    }

    /* Sidebar button on hover */
    .stButton>button:hover {
        background-color: #006666;
    }

    /* Main title */
    .main-title {
        font-family: 'Arial Black', sans-serif;
        font-size: 36px;
        color: #AA3C3B;
        text-align: left;
        margin-left: 20px;
        margin-top: 30px;
    }

    /* General text */
    .custom-text {
        color: #AA3C3B;  /* Custom text color */
    }

    /* Logo */
    .logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 150px;
    }

    /* Image container */
    .uploaded-image {
        border: 2px solid #AA3C3B;
        padding: 10px;
        margin-top: 20px;
        border-radius: 10px;
        width: 50%;  /* Reduce the size of the uploaded image */
    }

    /* Prediction text */
    .prediction {
        font-size: 24px;
        color: #FF6347;
        text-align: center;
        margin-top: 20px;
    }

    /* Adjusting the padding for the top container */
    .top-container {
        width: 100%;
        display: flex;
        align-items: center;
        padding: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------#
#         Sidebar Navigation    #
# -----------------------------#
# Display the logo on the sidebar
logo_path = os.path.join('image', "main.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_column_width=True, width=150, caption="", output_format="PNG")
else:
    st.sidebar.warning("Logo image not found. Please check the path.")

st.sidebar.title("Options")
page = st.sidebar.radio("Go to", ["Home", "About", "Contact Us"])

# -----------------------------#
#           Home Page           #
# -----------------------------#
if page == "Home":
    # Create two columns for logo and title
    col1, col2 = st.columns([1, 3])  # Adjust the ratio as needed

    with col1:
        # Display the logo
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width=True, width=150, caption="", output_format="PNG")
        else:
            st.warning("Logo image not found. Please check the path.")

    with col2:
        # Display the main title
        st.markdown('<h1 class="main-title">Meat Quality Analyzer</h1>', unsafe_allow_html=True)

    # Add a brief description below the header
    st.markdown(
        '<p class="custom-text" style="font-size:18px;">Upload an image of meat, and the model will predict whether it is Fresh, Half Fresh, or Spoiled.</p>',
        unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="fileUploader")

    if uploaded_file is not None:
        # Display the uploaded image with custom styling
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=False, width=300, output_format="PNG")
        st.markdown(
            '<p class="custom-text" style="text-align: center;">Uploaded Image.</p>',
            unsafe_allow_html=True)

        # Preprocess the image
        img = image.resize((128, 128))
        x = img_to_array(img)
        x /= 255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        # -----------------------------#
        #       Model Prediction        #
        # -----------------------------#
        # Load the trained DenseNet model
        @st.cache_resource
        def load_model():
            model = tf.keras.models.load_model('meat_quality_analyzer_model.h5')
            return model
        
        model = load_model()

        # Define class names
        class_names = ['Fresh', 'Half Fresh', 'Spoiled']

        classes = model.predict(images, batch_size=10)
        print(classes[0])
        predicted_class = class_names[np.argmax(classes[0])]

        # Display prediction
        st.markdown(f'<p class="prediction">Prediction: <strong>{predicted_class}</strong></p>', unsafe_allow_html=True)

# -----------------------------#
#           About Page          #
# -----------------------------#
elif page == "About":
    st.markdown('<h1 class="main-title">About Us</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="custom-text">
        The **Meat Quality Analyzer** is a machine learning-based application designed to help users assess the freshness of meat through image analysis. 
        By uploading a photo of the meat, the model predicts whether the meat is **Fresh**, **Half Fresh**, or **Spoiled**.
        </p>
        """,
        unsafe_allow_html=True)

# -----------------------------#
#         Contact Us Page       #
# -----------------------------#
elif page == "Contact Us":
    st.markdown('<h1 class="main-title">Contact Us</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="custom-text">
        We'd love to hear from you! Whether you have a question about the project, need assistance, or just want to say hi, feel free to reach out.
        </p>
        <p class="custom-text">
        ### Contact Information:
        - **Email**: [team@example.com](mailto:team@example.com)
        - **Phone**: +1 (234) 567-8901
        - **Address**:
            ```
            1234 Meat Street
            Food City, FC 56789
            ```
        ### Follow Us:
        - [LinkedIn](#)
        - [Twitter](#)
        - [GitHub](#)
        </p>
        """,
        unsafe_allow_html=True)
