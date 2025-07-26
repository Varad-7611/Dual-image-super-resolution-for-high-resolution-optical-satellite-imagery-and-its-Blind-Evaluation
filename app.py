import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_squared_error as mse
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from io import BytesIO
from PIL import Image

st.set_page_config(layout="wide")
st.title("Dual Image Super Resolution with Evaluation Metrics")

# Upload two LR images
col1, col2 = st.columns(2)
with col1:
    lr1_file = st.file_uploader("Upload Low-Resolution Image 1", type=["png", "jpg"], key="lr1")
with col2:
    lr2_file = st.file_uploader("Upload Low-Resolution Image 2", type=["png", "jpg"], key="lr2")

def read_and_resize(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (256, 256))

def upsample(img):
    return cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0

def non_uniform_fusion(img1, img2):
    grad1 = np.abs(cv2.Laplacian(cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F))
    grad2 = np.abs(cv2.Laplacian(cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_64F))
    weight1 = grad1 / (grad1 + grad2 + 1e-8)
    weight2 = 1.0 - weight1
    fused = img1 * weight1[..., None] + img2 * weight2[..., None]
    return np.clip(fused, 0, 1)

def deep_feature_distance(img1, img2):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
    model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    f1 = model.predict(np.expand_dims(img1, axis=0), verbose=0)[0]
    f2 = model.predict(np.expand_dims(img2, axis=0), verbose=0)[0]
    return np.mean((f1 - f2) ** 2)

if lr1_file and lr2_file:
    lr1 = read_and_resize(lr1_file)
    lr2 = read_and_resize(lr2_file)

    lr1_up = upsample(lr1)
    lr2_up = upsample(lr2)

    sr = non_uniform_fusion(lr1_up, lr2_up)  # Simulated SR
    sr_uint8 = (sr * 255).astype(np.uint8)

    # Full-reference metrics (simulated since no real GT)
    mse_val = mse(lr1_up, sr)
    rmse_val = np.sqrt(mse_val)
    psnr_val = psnr(lr1_up, sr, data_range=1.0)
    ssim_val = ssim(lr1_up, sr, channel_axis=-1, data_range=1.0)

    # Blind metric: Deep Feature Distance
    vgg_score = deep_feature_distance(lr1_up, sr)

    # Plot 3 panels
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(lr1)
    ax[0].set_title("LR1")
    ax[0].axis('off')
    ax[1].imshow(lr2)
    ax[1].set_title("LR2")
    ax[1].axis('off')
    ax[2].imshow(sr_uint8)
    ax[2].set_title("Super-resolved")
    ax[2].axis('off')
    st.pyplot(fig)

    st.markdown("### Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse_val:.4f}")
    col1.metric("RMSE", f"{rmse_val:.4f}")
    col2.metric("PSNR", f"{psnr_val:.2f} dB")
    col2.metric("SSIM", f"{ssim_val:.4f}")
    col3.metric("Deep VGG Distance", f"{vgg_score:.6f}")

    # Download button
    sr_pil = Image.fromarray(sr_uint8)
    buf = BytesIO()
    sr_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download High-Resolution Image",
        data=byte_im,
        file_name="super_resolved_image.png",
        mime="image/png"
    )
