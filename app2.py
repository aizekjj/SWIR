import streamlit as st
import rasterio
from rasterio.io import MemoryFile
import numpy as np
import matplotlib.pyplot as plt
import spectral
from spectral import msam
import folium

# Title of the app
st.title("Hyperspectral Image Analysis for Iron Deposit Detection")

# Step 1: Upload the Hyperspectral Image
uploaded_file = st.file_uploader("Upload a Hyperspectral Image (GeoTIFF format)", type=["tif"])

# Step 2: Handle in-memory file using MemoryFile
if uploaded_file is not None:
    # Read the hyperspectral image from the in-memory file
    # Use MemoryFile to handle the uploaded file correctly
    with MemoryFile(uploaded_file) as memfile:
        with memfile.open() as dataset:
            hyperspectral_image = dataset.read()  # Load all bands
            bounds = dataset.bounds  # Get the geographic bounds of the image

    st.write(f"Image loaded successfully with shape {hyperspectral_image.shape}")
    
    # Example of identifying bands related to iron deposits (replace with actual bands)
    iron_band_1 = hyperspectral_image[30, :, :]  # Example band for iron feature
    iron_band_2 = hyperspectral_image[45, :, :]  # Another example band

    # Step 3: Band ratio to detect iron deposits
    st.write("Performing band ratio analysis...")
    iron_ratio = iron_band_1 / iron_band_2
    
    # Step 4: Display the Band Ratio Result
    st.write("Band Ratio Result (Iron detection):")
    fig, ax = plt.subplots()
    ax.imshow(iron_ratio, cmap='gray')
    ax.set_title("Iron Detection (Band Ratio)")
    st.pyplot(fig)

    # Step 5: SAM Classification using `msam`
    st.write("Performing Spectral Angle Mapper (SAM) analysis...")

    # Define a reference spectrum (replace with an actual reference for iron minerals)
    reference_spectrum = np.random.random(hyperspectral_image.shape[0])  # Replace with the actual spectrum
    
    # Perform Multiple Spectral Angle Mapper (msam)
    sam_result = msam(hyperspectral_image, np.array([reference_spectrum]))

    # Threshold the SAM result to highlight potential iron-rich areas
    sam_thresholded = np.where(sam_result < 0.1, 1, 0)

    # Step 6: Display SAM Result
    st.write("SAM Classification Result:")
    fig, ax = plt.subplots()
    ax.imshow(sam_thresholded, cmap='Reds')
    ax.set_title("Iron Deposits (SAM Classification)")
    st.pyplot(fig)

    # Step 7: Overlay results on a map using Folium
    st.write("Visualizing on an interactive map...")

    # Create a folium map centered on the image bounds
    m = folium.Map(location=[(bounds.top + bounds.bottom) / 2, (bounds.left + bounds.right) / 2], zoom_start=10)

    # Overlay the SAM classification result on the map
    folium.raster_layers.ImageOverlay(
        image=sam_thresholded,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.6
    ).add_to(m)

    # Display the map
    st_data = st._legacy_folium_static(m)

else:
    st.write("Please upload a hyperspectral image to start the analysis.")
