# 😊 Real-Time Facial Emotion Detector
### Deep Learning · Computer Vision · Transfer Learning

A real-time facial emotion detection system that classifies **7 human emotions** from live webcam feed using a Convolutional Neural Network trained on the FER-2013 dataset. Built with TensorFlow/Keras and OpenCV.

---

## What It Does

- Detects faces in real time using **OpenCV Haar Cascade**
- Classifies each detected face into one of **7 emotions**:
  `Angry` · `Disgust` · `Fear` · `Happy` · `Sad` · `Surprise` · `Neutral`
- Displays predicted emotion + confidence score as overlay on the video frame
- Shows a **probability bar chart** for all 7 classes per prediction

---

## Results

| Model | Test Accuracy |
|---|---|
| Custom CNN (3-block) | ~61–63% |
| VGG16 Transfer Learning ✓ | ~64–67% |

> FER-2013 is a genuinely hard benchmark — published research achieves 65–72%. Human accuracy on this dataset is ~65%.

---

## Dataset

**FER-2013** — Facial Expression Recognition 2013  
- 35,887 grayscale images at 48×48 pixels  
- 7 emotion classes  
- Severe class imbalance (Disgust: ~436 images vs Happy: ~7,215)  
- Source: [Kaggle — msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

---

## Model Architecture

### Model A — Custom CNN
```
Input (48×48×1)
  → Conv Block 1: Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → MaxPool → Dropout(0.25)
  → Conv Block 2: Conv2D(128) → BatchNorm → ReLU → Conv2D(128) → MaxPool → Dropout(0.25)
  → Conv Block 3: Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
  → Dense(512) → BatchNorm → ReLU → Dropout(0.5)
  → Dense(256) → ReLU → Dropout(0.3)
  → Dense(7, softmax)
```

### Model B — VGG16 Transfer Learning ✓ (Best)
```
Input (48×48×1)
  → Lambda: Grayscale → RGB (repeat channel ×3)
  → VGG16 base (ImageNet weights, top 4 layers fine-tuned)
  → GlobalAveragePooling2D
  → Dense(512) → BatchNorm → ReLU → Dropout(0.5)
  → Dense(256) → ReLU → Dropout(0.3)
  → Dense(7, softmax)
```

---

## Key Techniques

| Technique | Why Used |
|---|---|
| **Transfer Learning (VGG16)** | Leverages ImageNet features — edges, textures, shapes — already learned on 1.2M images |
| **Class Weights** | Fixes severe dataset imbalance — Disgust class (436 samples) gets weight of ~9.3× |
| **Data Augmentation** | Random rotation ±15°, zoom ±15%, horizontal flip — reduces overfitting |
| **BatchNormalization** | Stabilises and speeds up training across all layers |
| **Dropout (0.25–0.5)** | Prevents co-adaptation of neurons, improves generalisation |
| **ReduceLROnPlateau** | Halves learning rate when validation accuracy plateaus |
| **EarlyStopping** | Stops training when val_accuracy stops improving, restores best weights |
| **Haar Cascade (OpenCV)** | Fast real-time face detection before emotion classification |

---

## Project Structure

```
facial-emotion-detector/
│
├── Emotion_Detector_Complete_Code.ipynb   ← Full notebook (training + webcam)
│
├── outputs/
│   ├── sample_emotions.png                ← Sample images per emotion class
│   ├── class_distribution.png             ← Class imbalance bar chart
│   ├── training_curves.png                ← Accuracy & loss curves (both models)
│   ├── confusion_matrix.png               ← Raw + normalised confusion matrix
│   ├── predictions_sample.png             ← 14 test predictions (green/red)
│   └── webcam_prediction.png              ← Live webcam emotion detection result
│
└── README.md
```

---

## How to Run

### Option 1 — Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Emotion_Detector_Complete_Code.ipynb`
3. Go to **Runtime → Change runtime type → T4 GPU**
4. Set up Kaggle API and download dataset (instructions in notebook Step 2)
5. Run cells one by one — total time ~25–35 minutes

### Option 2 — Local Setup
```bash
# Install dependencies
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn pillow kaggle

# Download dataset
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d fer2013/

# Run notebook
jupyter notebook Emotion_Detector_Complete_Code.ipynb
```

---

## Visualisations

### Class Distribution (Training Set)
> Disgust has ~436 samples vs Happy's ~7,215 — handled using `compute_class_weight`

### Confusion Matrix Observations
- **Best classified:** Happy (~90%+), Neutral (~75%+)
- **Most confused:** Fear ↔ Sad, Angry ↔ Disgust
- **Reason:** Visually similar facial muscle patterns between these pairs

### Training Curves
- VGG16 converges faster than Custom CNN
- Both models show close train/val curves → no significant overfitting

---

## What I Learned

- How transfer learning improves performance on small, domain-specific datasets
- How class imbalance affects model training and how to correct it with class weights
- How Haar Cascade face detection works as a preprocessing step before deep learning inference
- Why Fear and Sad are confused — and how a larger dataset or attention mechanisms could help
- The gap between benchmark accuracy and human accuracy on emotion recognition tasks

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Transfer_Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![NumPy](https://img.shields.io/badge/NumPy-1.x-lightblue)

---

## Author

**Yuvraj Agrawal**  
B.Tech Computer Science, OP Jindal University  
[GitHub](https://github.com/yuvrajagrawal29) · [LinkedIn](www.linkedin.com/in/yuvraj-agrawal-95b111333)
