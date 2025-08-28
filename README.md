# Pneumonia Detection from Chest X-ray Images 

This project applies deep learning techniques to classify chest X-ray images as Pneumonia or Normal, using a Convolutional Neural Network (CNN). It was developed as part of an academic assignment in a Deep Learning course.

## Dataset

The dataset provided for this task includes chest X-ray images categorized into:
- `NORMAL` (healthy lungs)
- `PNEUMONIA` (infected lungs)

The dataset used is Chest X-Ray Images (Pneumonia) from Kaggle.

## Models and Approach

We implemented and compared multiple models:

1. **SVM (Support Vector Machine)**:
   - Features were extracted from images using histogram techniques.
   - Used as a baseline classical ML model.

2. **CNN (Convolutional Neural Network)**:
   - Built from scratch using Keras.
   - Included convolution, pooling, and dense layers.
   - Served as our first deep learning baseline.

3. **Transfer Learning with EfficientNet**:
   - Used pre-trained EfficientNetB0 from `keras.applications`.
   - Added custom dense layers for classification.
   - Achieved the best performance.

## Results

| Model         | Accuracy | Notes                        |
|---------------|----------|------------------------------|
| SVM           | ~70%     | Baseline, classical approach |
| CNN (scratch) | ~85%     | Good performance             |
| EfficientNet  | ~91%     | Best accuracy achieved       |

## Key Highlights

- Applied extensive data preprocessing, including resizing, grayscale conversion, and normalization.
- Visualized sample X-rays and predictions.
- Compared learning curves and confusion matrices.
- Identified areas for further model improvement (e.g., class imbalance handling, data augmentation).
