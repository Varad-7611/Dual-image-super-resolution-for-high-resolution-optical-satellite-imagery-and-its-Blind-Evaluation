# 🛰️ Dual Image Super-Resolution with Blind Evaluation

This project focuses on enhancing the resolution of satellite imagery by using two low-resolution (LR) optical satellite images of the same region. By combining classical image fusion techniques and deep learning (SRCNN), we generate a high-resolution (HR) image and evaluate its quality using both traditional and perceptual (blind) metrics.

---

## 🚀 Key Features

- ✅ Upload two LR satellite images of the same area
- 🧠 Automatic image registration and alignment
- 🔁 Image fusion to combine complementary details
- 🧬 Super-resolution using a trained SRCNN model
- 📊 Evaluation metrics: PSNR, SSIM, MSE, RMSE, and Deep VGG Perceptual Distance
- 🌐 Streamlit Web UI (can also be deployed on Android/Web using Flask API)
- 📷 Outputs a high-resolution image and comparison with ground truth (if available)

---

## 🔧 Technologies Used & Benefits

| Technology     | Purpose                                       | Benefit                                                |
|----------------|-----------------------------------------------|---------------------------------------------------------|
| **Python**     | Core programming language                     | Fast prototyping and model integration                 |
| **OpenCV**     | Image processing & registration               | Efficient image alignment and preprocessing            |
| **SRCNN (Keras/TensorFlow)** | Deep learning model for SR      | Lightweight, accurate super-resolution                 |
| **Streamlit**  | Frontend UI                                   | Real-time web interface for user interaction           |
| **Flask**      | Backend API (optional)                        | Enables integration with Android or external systems   |
| **scikit-image**| Evaluation metrics computation               | Easily calculate PSNR, SSIM, MSE, etc.                 |
| **VGG16**      | Deep feature extractor for blind metrics      | Provides perceptual similarity score (no HR needed)    |

---

## 📦 Deployment


### 💻 Local Run

```bash
# Clone the repo
git clone https://github.com/yourusername/dual-image-super-resolution.git
cd dual-image-super-resolution

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

```


