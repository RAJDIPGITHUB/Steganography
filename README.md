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
## 🚀 How It Works
🧵 Embedding Flow (main_flow.py)
Load a cover image (preferably with rich edges).

Detect safe zones using Sobel/Canny and a trained CNN.

Generate a QR code from the message.

Convert QR to bitstream and replicate n times for redundancy.

Embed bits into 8×8 rotated safe blocks (only blue channel LSB).

Save stego_output.png and secret_key.json.

🔓 Decoder Flow (decoder_flow.py)
Load compressed stego_output.jpg or .png.

Use secret_key.json to identify block positions and rotation angles.

Extract LSBs and apply vote-based reconstruction.

Save and decode the QR to recover the hidden message.

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

