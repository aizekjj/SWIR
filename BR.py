import streamlit as st
import rasterio
from rasterio.io import MemoryFile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Set a dark theme for matplotlib
plt.style.use('dark_background')

# Add logo at the top left of the page
logo_path = "C:/Users/1/OneDrive/Desktop/Python/VS code/Xplorelink/Xplorelink_logo.webp"
logo = Image.open(logo_path)
st.image(logo, width=200)

# Title of the app
st.title("Iron Detection Using Band Ratio Technique")

# Step 1: Upload Hyperspectral Image for Iron Detection
uploaded_file = st.file_uploader("Upload a Hyperspectral Image (GeoTIFF format) for Iron Detection", type=["tif"])

# Confirm if file is uploaded
if uploaded_file is not None:
    st.write("File uploaded successfully. Processing the file...")

    # Handle in-memory file using MemoryFile
    with MemoryFile(uploaded_file) as memfile:
        with memfile.open() as dataset:
            hyperspectral_image = dataset.read()  # Load all bands
            num_bands = hyperspectral_image.shape[0]  # Get the number of bands

    st.write(f"Image loaded successfully with shape {hyperspectral_image.shape}")
    st.write(f"Number of bands in the image: {num_bands}")

    # Select specific bands for iron detection
    # For this example, we use bands 6 and 7 (assuming they correspond to Red and NIR/SWIR sensitive to iron).
    if num_bands > 7:
        iron_band_1 = hyperspectral_image[7, :, :].astype(np.float64)  # Example: band 8
        iron_band_2 = hyperspectral_image[6, :, :].astype(np.float64)  # Example: band 7
    else:
        st.error("The image doesn't have enough bands. Using the first available bands.")
        iron_band_1 = hyperspectral_image[0, :, :].astype(np.float64)  # Fallback: first band
        iron_band_2 = hyperspectral_image[1, :, :].astype(np.float64)  # Fallback: second band

    # Step 2: Calculate Band Ratio to highlight iron deposits, handling division by zero
    st.write("Performing Band Ratio Analysis for Iron Detection...")
    epsilon = 1e-10
    iron_band_2 = np.where(iron_band_2 == 0, epsilon, iron_band_2)  # Replace 0 in iron_band_2 with epsilon
    iron_ratio = iron_band_1 / iron_band_2

    # Step 3: Display the Band Ratio Result
    st.write("Band Ratio Result (Iron Detection):")
    fig, ax = plt.subplots()
    ax.imshow(iron_ratio, cmap='inferno')  # Use a darker colormap like 'inferno' or 'plasma'
    ax.set_title("Iron Detection (Band Ratio)", color='white')
    st.pyplot(fig)
else:
    st.write("Please upload a hyperspectral image to start the analysis.")
