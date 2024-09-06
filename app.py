import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import img_to_array
from PIL import Image
import os
from fpdf import FPDF
from io import BytesIO
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from groq import Groq

# -----------------------------#
#         BLIP-2 Initialization
# -----------------------------#
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

# -----------------------------#
#        Custom CSS Styling    #
# -----------------------------#
st.markdown(
    """
    <style>
    /* Background color */
    .stApp {
        background-color: #f9f9f9;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f9f9f9;
        color: #AA3C3B;
    }

    /* Sidebar button styling */
    .stButton>button {
        background-color: #AA3C3B;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        cursor: pointer;
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
        color: #AA3C3B;
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
page = st.sidebar.radio("Go to", ["Home", "About Us", "Contact Us"])

# -----------------------------#
#         LLM work             #
# -----------------------------#
client = Groq(api_key="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")

# Function to generate image caption using BLIP-2
def generate_caption(image):
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return None

# Function to generate the inspection report in PDF format
def generate_inspection_report(predicted_class, report_text):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Meat Inspection Report", ln=True, align="C")

    # Classification Result
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Classification: {predicted_class}", ln=True)

    # Report Details
    pdf.multi_cell(0, 10, report_text)

    # Save the PDF to a string buffer instead of a file
    pdf_output = pdf.output(dest='S').encode('latin1')  # 'S' returns a string, encoded to match PDF format
    pdf_buffer = BytesIO(pdf_output)  # Convert the string to BytesIO object for download
    
    return pdf_buffer

# Function to create the report content using BLIP-2 for image captioning
def create_llm_report(predicted_class, image_caption):
    report_text = (
        f"The meat is classified as {predicted_class}.\n"
        f"Image Caption: {image_caption}\n"
        "Generate a detailed inspection report including:\n"
        "1. Recommended actions\n"
        "2. Possible shelf-life\n"
        "3. Guidelines on handling spoiled meat (if applicable)\n"
        "4. Whether the meat is eatable or not."
    )
    return report_text

# -----------------------------#
#           Home Page           #
# -----------------------------#
if page == "Home":
    # Create two columns for logo and title
    col1, col2 = st.columns([1, 3])  # Adjust the ratio as needed

    with col1:
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width=True, width=150, caption="", output_format="PNG")
        else:
            st.warning("Logo image not found. Please check the path.")

    with col2:
        st.markdown('<h1 class="main-title">Meat Quality Analyzer</h1>', unsafe_allow_html=True)

    st.markdown(
        '<p class="custom-text" style="font-size:18px;">Upload an image of meat, and the model will predict whether it is Fresh, Half Fresh, or Spoiled.</p>',
        unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="fileUploader")

    if uploaded_file is not None:
        # Display the uploaded image with custom styling
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=False, width=300, output_format="PNG")

        # Preprocess the image for the model
        img = image.resize((128, 128))
        x = img_to_array(img)
        x /= 255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])

        @st.cache_resource
        def load_model():
            model = tf.keras.models.load_model('meat_quality_analyzer_model.h5')
            return model
        
        model = load_model()

        # Define class names
        class_names = ['Fresh', 'Half Fresh', 'Spoiled']

        classes = model.predict(images, batch_size=10)
        predicted_class = class_names[np.argmax(classes[0])]

        # Display prediction
        st.markdown(f'<p class="prediction">Prediction: <strong>{predicted_class}</strong></p>', unsafe_allow_html=True)

        # Generate caption using BLIP-2
        image_caption = generate_caption(image)
        if image_caption:
            st.markdown(f"**Image Caption:** {image_caption}")

            if st.button("Create Inspection Report"):
                report_text = create_llm_report(predicted_class, image_caption)
                report_buffer = generate_inspection_report(predicted_class, report_text)

                # Provide download button with the in-memory PDF buffer
                st.download_button(
                    label="Download Report",
                    data=report_buffer,
                    file_name="inspection_report.pdf",
                    mime="application/pdf"
                )

# -----------------------------#
#           About Us Page       #
# -----------------------------#
elif page == "About Us":
    st.markdown('<h1 class="main-title">About Us</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="custom-text" style="font-size:18px;">
        Welcome to the Meat Quality Analyzer. Our mission is to help users analyze and assess the quality of meat using advanced machine learning models.
        Upload an image, and let our model predict the freshness of your meat and generate a report.
        </p>
        """
    )
    st.image('image/team.JPG', use_column_width=True)

# -----------------------------#
#        Contact Us Page        #
# -----------------------------#
elif page == "Contact Us":
    st.markdown('<h1 class="main-title">Contact Us</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="custom-text" style="font-size:18px;">
        We'd love to hear from you! Whether you have a question about the project, need assistance, or just want to say hi, feel free to reach out at: <strong>info@meatqualityanalyzer.com</strong>
        </p>
        """
    )
