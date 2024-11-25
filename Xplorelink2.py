import streamlit as st
import rasterio
from rasterio.io import MemoryFile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Set a dark theme for matplotlib
plt.style.use('dark_background')

# Function to reset the session state
def reset_app():
    for key in st.session_state.keys():
        del st.session_state[key]

# Add "Start New" button at the top
if st.button("Start New"):
    reset_app()
    st.experimental_rerun()  # Refresh the app to go back to the main page

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
    if num_bands > 8:
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
    ax.hist(valid_ratio_values, bins=50, color='orange', edgecolor='black')
    ax.set_title("Histogram of Iron Ratio Values")
    ax.set_xlabel("Iron Ratio Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Allow user to adjust threshold for highlighting iron-rich areas
    threshold_value = st.slider("Set Threshold for Iron-Rich Detection", min_value=0.0, max_value=float(max_val), value=1.2, step=0.1)

    # Step 3: Display the Band Ratio Result with highlighted points for values above the threshold
    st.write(f"Band Ratio Result (Iron Detection) with Bold Red Dots for Values > {threshold_value}:")
    fig, ax = plt.subplots()
    ax.imshow(iron_ratio, cmap='Pastel1')
    ax.set_title("Iron Detection (Band Ratio with Highlighted Points)", color='white')

    # Identify coordinates where iron_ratio exceeds threshold and overlay red dots
    y_coords, x_coords = np.where(iron_ratio > threshold_value)
    ax.scatter(x_coords, y_coords, color='black', s=1, label=f"Iron-rich (> {threshold_value})", alpha=0.7)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    st.pyplot(fig)

else:
    st.write("Please upload a hyperspectral image to start the analysis.")
