 CT-Scan Kidney Disease Classification (Nephropathy Classification)

A deep learning-based medical imaging system designed to classify kidney CT scans into four clinically significant categories: Normal, Cyst, Tumor, and Stone.
Built with TensorFlow, utilizing transfer learning (VGG16) and a custom CNN model, and deployed as an interactive Streamlit web application for real-time clinical use.
<img width="1506" height="673" alt="image" src="https://github.com/user-attachments/assets/c3427a92-2dc3-45e3-bd78-314c0009a69e" />
<img width="558" height="661" alt="image" src="https://github.com/user-attachments/assets/7495fdb6-3b64-4bba-92c2-b9f5437ce8b3" />



Features
 
Multi-Class Classification: Distinguishes between four kidney conditions using 2D CT scan slices.
Deep Learning Models:
Custom CNN: Lightweight architecture achieving 81.2% test accuracy.
VGG16 Transfer Learning: Pre-trained on ImageNet, fine-tuned for medical imaging.
Real-Time Web App: Interactive Streamlit interface for instant predictions and visual feedback.
Data Augmentation: Comprehensive augmentation pipeline (rotation, flipping, zoom, brightness) to combat overfitting.
=Model Interpretability: Confidence scores and prediction probabilities displayed for clinical transparency.



Model Performance

| Model        | Training Accuracy | Test Accuracy | Note                          |
|--------------|-------------------|---------------|-------------------------------|
| VGG16     | 99.1%          | N/A       | High training accuracy, used with augmentation |
| Custom CNN| ~95%           | 81.2%     | Generalizes well to unseen data |

> Note: The custom CNN's 81.2% test accuracy represents a robust, deployable performance on a limited medical imaging dataset.



🛠️ Tech Stack

Deep Learning: TensorFlow 2.x, Keras
Computer Vision: OpenCV, Pillow
Web Framework: Streamlit
Data Processing: Pandas, NumPy, Scikit-learn
Visualization: Matplotlib, Seaborn
Environment: Python 3.8+

