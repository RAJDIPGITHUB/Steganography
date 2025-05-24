# Steganography
Research publication

# Robust QR Code-Based Image Steganography 📦🔍  
**Using ML-Guided Safe Zones + Multi-Copy Embedding with Error Correction**

This project implements a novel and robust QR code-based image steganography framework designed to survive lossy compression (JPEG/PNG) while maintaining high imperceptibility. It uses **machine learning and edge detection** to identify optimal embedding zones and leverages **multi-copy block-based QR encoding** with a voting-based decoder for resilient message recovery.

---

## 📌 Features

- ✅ Safe zone detection using **CNN classifier** + **Sobel/Canny edge detectors**
- 🔁 **8×8 block-based rotation** to enhance embedding variation
- 📦 Embeds **QR code** data in blue-channel LSBs of safe zones
- 🧠 **Multi-copy embedding** + **voting-based decoding** for robust QR recovery
- 🧱 Structured Python modules with modular design for easy extension
- 📊 Supports PSNR, SSIM, and QR recovery metrics after compression
- 🧪 Works even with **JPEG quality as low as 50%**

---

## 🧰 Requirements

- Python 3.8+
- OpenCV (`cv2`)
- TensorFlow (`keras`)
- NumPy
- `qrcode`
- `matplotlib` *(optional for visual outputs)*
- `scikit-image` *(for SSIM computation)*

```bash
pip install opencv-python numpy tensorflow qrcode scikit-image matplotlib
