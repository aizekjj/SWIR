import streamlit as st
import rasterio
from rasterio.io import MemoryFile
import numpy as np
import matplotlib.pyplot as plt
import spectral
from spectral import msam
import folium
from streamlit_folium import st_folium

# Add logo at the top left of the page
st.image("/mnt/data/Xplorelink (500 x 100 px) (1).webp", width=200)

# Title of the app
st.title("Hyperspectral Image Analysis")

# Step 1: Choose Analysis Type
analysis_type = st.selectbox("Choose Analysis Type", ["Mineral Exploration", "Deforestation", "Hurricane Damage"])

if analysis_type == "Mineral Exploration":
    # Step 2: Choose Mineral Type
    mineral_type = st.selectbox("Choose Mineral Type", ["Gold", "Copper", "Kymberlites"])

    # Step 3: Upload Hyperspectral Image
    uploaded_file = st.file_uploader("Upload a Hyperspectral Image (GeoTIFF format)", type=["tif"])

    if uploaded_file is not None:
        # Handle in-memory file using MemoryFile
        with MemoryFile(uploaded_file) as memfile:
            with memfile.open() as dataset:
                hyperspectral_image = dataset.read()  # Load all bands
                num_bands = hyperspectral_image.shape[0]  # Get the number of bands
                bounds = dataset.bounds  # Get the geographic bounds of the image

        st.write(f"Image loaded successfully with shape {hyperspectral_image.shape}")
        st.write(f"Number of bands in the image: {num_bands}")

        # Band selection based on mineral type (for demonstration purposes, using arbitrary bands)
        if mineral_type == "Gold":
            band_1 = hyperspectral_image[7, :, :]
            band_2 = hyperspectral_image[6, :, :]
        elif mineral_type == "Copper":
            band_1 = hyperspectral_image[5, :, :]
            band_2 = hyperspectral_image[4, :, :]
        elif mineral_type == "Kymberlites":
            band_1 = hyperspectral_image[3, :, :]
            band_2 = hyperspectral_image[2, :, :]

        # Step 4: Band ratio to detect deposits
        st.write("Performing band ratio analysis...")
        mineral_ratio = band_1 / band_2

        # Display the Band Ratio Result
        st.write("Band Ratio Result (Deposit Detection):")
        fig, ax = plt.subplots()
        ax.imshow(mineral_ratio, cmap='gray')
        ax.set_title(f"{mineral_type} Detection (Band Ratio)")
        st.pyplot(fig)

        # Step 5: SAM Classification using `msam`
        st.write("Performing Spectral Angle Mapper (SAM) analysis...")

        # Define a reference spectrum that matches the number of bands in the image
        reference_spectrum = np.random.random(num_bands)  # Replace with actual spectrum for specific mineral
        
        # Reshape the reference_spectrum into a 2D array: (n_spectra, n_bands)
        reference_spectrum = reference_spectrum.reshape(1, -1)  # Single spectrum, same number of bands
        
        # Perform Multiple Spectral Angle Mapper (msam)
        sam_result = msam(hyperspectral_image, reference_spectrum)

        # Threshold the SAM result to highlight potential deposit areas
        sam_thresholded = np.where(sam_result < 0.1, 1, 0)

        # Display SAM Result
        st.write("SAM Classification Result:")
        fig, ax = plt.subplots()
        ax.imshow(sam_thresholded, cmap='Reds')
        ax.set_title(f"{mineral_type} Deposits (SAM Classification)")
        st.pyplot(fig)

        # Step 6: Overlay results on a map using Folium
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
        st_folium(m, width=700, height=500)

elif analysis_type == "Deforestation":
    st.write("Deforestation analysis page under construction.")

elif analysis_type == "Hurricane Damage":
    st.write("Hurricane damage analysis page under construction.")
