# Data Card: Face Recognition Dataset

## 1. Sources and Licenses
- **Dataset:** Celebrity Face Image Dataset (Kaggle)
- **License:** CC0: Public Domain (Free for commercial and academic use)
- **Link:** https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset

## 2. Counts per Class
- Total Classes (Students): 17 Celebrities (e.g., Tom Cruise, Will Smith, etc.)
- Images per class: 1 image per class (used for 1-shot learning via embedding extraction).
- *Note: In a real-world scenario, this is populated via the "Live Registration" GUI feature.*

## 3. Image Sizes & Preprocessing
- Images are loaded via OpenCV and resized dynamically to match the VGG-Face model requirements (224x224).
- Color space is converted from BGR to RGB before embedding extraction.

## 4. Known Gaps
- The dataset contains celebrity faces which are high-quality, well-lit, and centered. It does not represent real-world classroom variations (poor lighting, occlusion, angled faces).

## 5. Split Strategy (Validation)
- **Test/Validation:** Because this is a real-time application, the "test split" is conducted live via webcam. We test the system by having a person stand in front of the camera and verifying if the bounding box and CSV log match the ground truth.
