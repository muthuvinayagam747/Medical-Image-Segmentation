
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Pancreas CT Segmentation",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Image Segmentation using Pancreas-CT")
st.write(
    "Upload a CT image to visualize the predicted segmentation mask."
)

st.warning(
    "Educational demonstration only. This model was trained on "
    "synthetic sample data, not real Pancreas-CT scans. "
    "Do not use the results for medical diagnosis."
)

# Build the same U-Net architecture used in Colab
@st.cache_resource
def build_model():
    from tensorflow.keras import layers, models

    H, W = 64, 64
    inputs = layers.Input((H, W, 1))

    c1 = layers.Conv2D(16, 3, activation="relu",
                       padding="same")(inputs)
    c1 = layers.Conv2D(16, 3, activation="relu",
                       padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation="relu",
                       padding="same")(p1)
    c2 = layers.Conv2D(32, 3, activation="relu",
                       padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    b = layers.Conv2D(64, 3, activation="relu",
                     padding="same")(p2)

    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(32, 3, activation="relu",
                       padding="same")(u1)

    u2 = layers.UpSampling2D()(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(16, 3, activation="relu",
                       padding="same")(u2)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c4)

    return models.Model(inputs, outputs)


# Load the model trained and saved in Google Colab
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "pancreas_segmentation_demo.keras"
    )


uploaded_file = st.file_uploader(
    "Upload a CT image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    original = np.array(image)

    # Prepare image for the demonstration model
    resized = image.resize((64, 64))
    array = np.array(resized, dtype=np.float32) / 255.0
    array = array[np.newaxis, :, :, np.newaxis]

    try:
        model = load_model()
        prediction = model.predict(array, verbose=0)[0, :, :, 0]
        mask = (prediction > 0.5).astype(np.uint8)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Input Image")
            st.image(original, use_container_width=True)

        with col2:
            st.subheader("Predicted Mask")
            st.image(mask * 255, clamp=True,
                     use_container_width=True)

        with col3:
            st.subheader("Segmentation Overlay")
            fig, ax = plt.subplots()
            ax.imshow(original, cmap="gray")
            ax.imshow(
                mask,
                cmap="Reds",
                alpha=0.5,
                extent=(0, original.shape[1],
                        original.shape[0], 0),
                interpolation="nearest"
            )
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        st.success("Image processing completed.")

    except Exception as e:
        st.error(
            "Could not load the trained model. Ensure that "
            "'pancreas_segmentation_demo.keras' is present "
            "in the application folder."
        )
        st.caption(str(e))
else:
    st.info("Upload an image to view the demonstration output.")
