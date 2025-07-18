
# 🐶🐱 Dog vs Cat Image Classifier

This project is a deep learning-based image classification model that distinguishes between images of dogs and cats using Convolutional Neural Networks (CNNs) in TensorFlow/Keras.

## 📁 Dataset
The dataset used is the **Dogs vs. Cats** dataset available on Kaggle:  
https://www.kaggle.com/datasets/salader/dogs-vs-cats

It contains over 25,000 images of dogs and cats, with a roughly equal distribution.

## 📌 Project Objectives
- Preprocess and explore image data
- Build a CNN model for binary classification
- Train and evaluate the model on labeled data
- Visualize training progress (accuracy, loss)
- Predict and test model performance

## 🧠 Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Matplotlib
- NumPy

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/dog-vs-cat-classifier.git
   cd dog-vs-cat-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook tutorial.ipynb
   ```

## 📊 Model Architecture

- Input Layer (128x128x3)
- Conv2D + ReLU
- MaxPooling2D
- Conv2D + ReLU
- MaxPooling2D
- Flatten
- Dense (Fully Connected)
- Output (Sigmoid for binary classification)

## 📈 Results
- Training accuracy reached ~XX% (fill this based on your actual result)
- Loss and accuracy plots are included for visual evaluation

## 📌 Sample Predictions
Include example output images if possible showing correct/incorrect predictions.

## 📂 File Structure
.
├── tutorial.ipynb
├── README.md
├── train/               # Training images (dog/cat)
├── test/                # Testing images
└── ...

## 📜 License
This project is for educational purposes. Dataset is provided by Kaggle under their respective license.

---

### 🔗 Links
- [Kaggle Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats?select=train)
- [Project Notebook](tutorial.ipynb)

---

### 🙌 Acknowledgments
Thanks to the Kaggle community and TensorFlow/Keras documentation.
