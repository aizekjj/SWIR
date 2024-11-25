import streamlit as st
import rasterio
from rasterio.io import MemoryFile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import folium
from streamlit_folium import st_folium

# Set a dark theme for matplotlib
plt.style.use('dark_background')

# Add logo at the top left of the page
logo_path = "C:/Users/1/OneDrive/Desktop/Python/VS code/Xplorelink/Xplorelink_logo.webp"
logo = Image.open(logo_path)
st.image(logo, width=200)

# Title of the app
st.title("AI Minerals")

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
            bounds = dataset.bounds  # Get the geographic bounds of the image

    st.write(f"Image loaded successfully with shape {hyperspectral_image.shape}")
    st.write(f"Number of bands in the image: {num_bands}")

    # Select specific bands for iron detection
    if num_bands > 7:
        iron_band_1 = hyperspectral_image[7, :, :].astype(np.float64)  # Example: band 8
        iron_band_2 = hyperspectral_image[6, :, :].astype(np.float64)  # Example: band 7
    else:
        st.error("The image doesn't have enough bands. Using the first available bands.")
        iron_band_1 = hyperspectral_image[0, :, :].astype(np.float64)  # Fallback: first band
        iron_band_2 = hyperspectral_image[1, :, :].astype(np.float64)  # Fallback: second band

    # Step 2: Calculate Band Ratio to highlight iron deposits using np.divide
    st.write("Performing Band Ratio Analysis for Iron Detection...")
    iron_ratio = np.divide(iron_band_1, iron_band_2, out=np.zeros_like(iron_band_1), where=iron_band_2 != 0)

    # Replace any remaining NaN or Inf values with 0
    iron_ratio = np.nan_to_num(iron_ratio, nan=0, posinf=0, neginf=0)

    # Filter out NaN and Inf values in iron_ratio for accurate min, max, and mean calculations
    valid_ratio_values = iron_ratio[np.isfinite(iron_ratio)]

    # Display statistics for the iron_ratio values before showing the result
    min_val = valid_ratio_values.min()
    max_val = valid_ratio_values.max()
    mean_val = valid_ratio_values.mean()
    st.write(f"**Iron Ratio Statistics**:")
    st.write(f"Minimum: {min_val}")
    st.write(f"Maximum: {max_val}")
    st.write(f"Mean: {mean_val}")

    # Plot histogram of iron_ratio values
    st.write("**Distribution of Iron Ratio Values**:")
    fig, ax = plt.subplots()
    ax.hist(valid_ratio_values, bins=5, color='gray', edgecolor='black')
    ax.set_title("Iron Ratio Values")
    ax.set_xlabel("Iron Ratio Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Display a specific band from the original image as chosen by the user
    band_choice = st.selectbox("Select a band to view the original image", range(num_bands))
    st.write(f"Original Image (Band {band_choice + 1})")
    fig, ax = plt.subplots()
    ax.imshow(hyperspectral_image[band_choice, :, :], cmap='gray')
    ax.set_title(f"Original Image (Band {band_choice + 1})")
    st.pyplot(fig)

    # Allow user to adjust threshold for highlighting iron-rich areas
    threshold_value = st.slider("Set Threshold for Iron-Rich Detection", min_value=0.0, max_value=float(max_val), value=1.2, step=0.1)

    # Step 3: Display the Band Ratio Result with highlighted points for values above the threshold
    st.write(f"BR Result (Iron Detection) for Values > {threshold_value}:")
    fig, ax = plt.subplots()
    ax.imshow(iron_ratio, cmap='cividis')
    ax.set_title("Iron Deposit Detection", color='white')

    # Identify coordinates where iron_ratio exceeds threshold and overlay red dots
    y_coords, x_coords = np.where(iron_ratio > threshold_value)
    ax.scatter(x_coords, y_coords, color='white', s=1, label=f"Iron-rich (> {threshold_value})", alpha=0.7)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    st.pyplot(fig)

    # Step 4: Display the Folium map with highlighted regions
    st.write("Iron Detection Map")

    # Create a Folium map centered on the image bounds
    m = folium.Map(location=[(bounds.top + bounds.bottom) / 2, (bounds.left + bounds.right) / 2], zoom_start=10)

    # Convert thresholded_image to 8-bit for compatibility with Folium
    thresholded_image = np.where(iron_ratio > threshold_value, 255, 0).astype(np.uint8)

    # Overlay the binary mask on the map
    folium.raster_layers.ImageOverlay(
        image=thresholded_image,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.6,
        colormap=lambda x: (1, 0, 0, x),  # Use red for highlighted regions
    ).add_to(m)

    # Display the Folium map in Streamlit
    st_folium(m, width=700, height=500)

else:
    st.write("Please upload a hyperspectral image to start the analysis.")
