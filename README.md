# Facial Landmark Detection with Dense SIFT + Ridge Regression

Predicts the (x, y) coordinates of 5 facial landmarks per image using classical computer
vision features and a regularised linear model. No deep learning is used.

## Pipeline
1. **Preprocessing:** resize 256→128 px, grayscale, histogram equalization (lighting),
   unsharp masking (edges), normalization. Landmark coordinates are rescaled to match.
2. **Features:** Dense SIFT on a 12×12 grid, giving 144 descriptors × 128 dims = an
   18,432-dimensional vector per image
3. **Model:** Ridge Regression (alpha = 0.5), predicting all 10 coordinates at once.
   L2 regularisation handles the highly correlated SIFT features.
4. **Evaluation:** 80/20 split, per-landmark Euclidean error and cumulative error curves

## Results
| Metric | Value |
| Mean landmark error | 3.13 px |
| Median error | 2.69 px |
| MSE | 7.19 |
| Points within 3 px | 55.7% |
| Points within 5 px | 84.0% |
| Points within 10 px | 99.0% |

The model is most accurate on clear, front-facing images. The largest errors (up to
21.6 px) come from side profiles and occluded faces.

## Why this approach
- Dense SIFT captures gradient texture around the eyes, nose and mouth with no training needed
- Ridge is fast, interpretable and handles thousands of correlated features without
  overfitting. SVR was considered but is slow at this dimensionality.

## Dataset
2,811 training and 554 test images (256×256 RGB), each with 5 annotated landmarks.

## Tools
Python, OpenCV, NumPy, scikit-learn, matplotlib (Google Colab)
