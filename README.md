# Gesture Recognition System

A hand gesture recognition system using MediaPipe and CNN, with support for single-hand gestures and two-hand combinations.

## Project Structure

### Datasets
- `dataset/` - Raw image dataset for CNN training
- `dataset_grayscale/` - Grayscale converted images for CNN
- `dataset_mask/` - Failed attempt at masked dataset
- `dataset_distance/` - Distance-based features for ANN (incomplete, no training code)

### Main Files
- `mediapipe_detection.py` - Real-time two-hand gesture detection with combos
- `train_cnn.py` - Train CNN model on grayscale dataset
- `test_cnn.py` - Test CNN model with single hand
- `test_hybrid.py` - Hybrid CNN + rule-based detection

### Dataset Tools
- `dataset_create.py` - Capture image dataset manually
- `capture_distance_dataset.py` - Capture distance-based sequences (for ANN)
- `convert_to_grayscale.py` - Convert dataset images to grayscale
- `dataset_mask.py` - Attempt to create masked dataset (failed)

## Supported Gestures

### Single-Hand Gestures
- `up` - Index finger pointing up
- `down` - Index finger pointing down
- `left` - Index finger pointing left
- `right` - Index finger pointing right
- `peace` - Index and middle fingers extended
- `palm` - Open hand, all fingers extended
- `fist` - Closed fist
- `thumbs_up` - Thumb extended upward

### Two-Hand Combinations
- `UNLOCK` - Left: PEACE + Right: OK
- `SELECT` - Left: PALM + Right: FIST
- `ZOOM_IN` - Left: PEACE + Right: PALM
- `ZOOM_OUT` - Left: PALM + Right: PEACE
- `THUMBS_COMBO` - Both hands: THUMBS_UP
- `PEACE_COMBO` - Both hands: PEACE
- `HIGH_FIVE` - Both hands: PALM

## Usage

### 1. Create Dataset

Run the dataset creator and press number keys (1-8) to select gesture:
```bash
python dataset_create.py
```
Controls:
- `1-8` - Select gesture type
- `c` - Capture image
- `q` - Quit

### 2. Convert to Grayscale

```bash
python convert_to_grayscale.py
```

### 3. Train CNN Model

```bash
python train_cnn.py
```
Outputs: `gesture_cnn.h5` and `classes.txt`

### 4. Run Detection

Two-hand detection with combos:
```bash
python mediapipe_detection.py
```
Controls:
- `q` - Quit
- `r` - Refresh/reset sequences
- `c` - Clear all buffers

CNN-only detection:
```bash
python test_cnn.py
```

Hybrid detection:
```bash
python test_hybrid.py
```

## Requirements

```
opencv-python
mediapipe
numpy
tensorflow
keras
```

Install MediaPipe hand landmarker model:
Download `hand_landmarker.task` from MediaPipe and place in project root.

## Notes

- CNN approach uses grayscale images (128x128)
- Two-hand detection requires both hands visible simultaneously
- Dataset distance approach (ANN) is incomplete
- Dataset mask approach was unsuccessful