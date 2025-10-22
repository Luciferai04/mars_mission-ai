# Hazard Model Training Guide

## Overview

This guide covers training machine learning models to identify terrain hazards from Mars DEM data. The models complement GPT-4V analysis by providing fast, lightweight hazard classification.

---

## Dataset Creation

### 1. Build Training Dataset from DEMs

```bash
# Extract features from DEM
python scripts/build_hazard_dataset.py \
  --dem data/dem/jezero_demo.tif \
  --output data/cache/hazard_dataset.json \
  --samples 10000 \
  --lat 18.4
```

**Process:**
1. Load DEM GeoTIFF
2. Compute slope at each pixel
3. Calculate local terrain features (curvature, roughness)
4. Label pixels: SAFE (0°-15°), CAUTION (15°-30°), HAZARD (>30°)
5. Extract random samples with balanced classes
6. Save to JSON format

### 2. Dataset Schema

```json
{
  "samples": [
    {
      "features": {
        "slope_deg": 12.5,
        "elevation_m": 100.0,
        "curvature": 0.02,
        "roughness": 5.3,
        "local_slope_variance": 3.2
      },
      "label": 0,
      "label_name": "SAFE",
      "coordinates": [18.445, 77.451]
    }
  ],
  "metadata": {
    "dem_source": "jezero_demo.tif",
    "total_samples": 10000,
    "class_distribution": {
      "SAFE": 7500,
      "CAUTION": 2000,
      "HAZARD": 500
    }
  }
}
```

---

## Feature Engineering

### Terrain Features

**1. Slope (Primary)**
- Calculated from elevation gradient
- Most important hazard indicator
- Range: 0°-90°

**2. Curvature**
- Second derivative of elevation
- Detects ridges and valleys
- Positive = convex, Negative = concave

**3. Roughness**
- Standard deviation of elevation in local window
- Indicates rocky/uneven terrain
- Window size: 3x3 to 5x5 pixels

**4. Local Variance**
- Variance of slope in neighborhood
- Captures terrain complexity
- Higher = more unpredictable

**5. Aspect**
- Direction of slope (N, E, S, W)
- Affects solar exposure and visibility
- Range: 0°-360°

### Feature Extraction Code

```python
import numpy as np
from scipy.ndimage import generic_filter

def compute_features(elevation, pixel_size):
    """Extract terrain features from elevation data."""
    
    # Slope
    dx = np.gradient(elevation, pixel_size[0], axis=1)
    dy = np.gradient(elevation, pixel_size[1], axis=0)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # Curvature (2nd derivative)
    dxx = np.gradient(dx, pixel_size[0], axis=1)
    dyy = np.gradient(dy, pixel_size[1], axis=0)
    curvature = dxx + dyy
    
    # Roughness (std dev in 3x3 window)
    def local_std(values):
        return np.std(values)
    roughness = generic_filter(elevation, local_std, size=3)
    
    # Slope variance
    def slope_var(values):
        return np.var(values)
    slope_variance = generic_filter(slope, slope_var, size=3)
    
    # Aspect
    aspect = np.degrees(np.arctan2(dy, dx)) % 360
    
    return {
        'slope': slope,
        'curvature': curvature,
        'roughness': roughness,
        'slope_variance': slope_variance,
        'aspect': aspect
    }
```

---

## Model Training

### Option 1: Simple Threshold Model

**Best for:** Fast deployment, interpretable decisions

```python
class ThresholdHazardModel:
    """Rule-based hazard classifier."""
    
    def __init__(self):
        self.slope_safe_max = 15.0
        self.slope_caution_max = 30.0
        self.roughness_threshold = 10.0
    
    def predict(self, features):
        slope = features['slope_deg']
        roughness = features['roughness']
        
        # High roughness increases hazard level
        if roughness > self.roughness_threshold:
            slope -= 5.0  # Effective slope penalty
        
        if slope <= self.slope_safe_max:
            return 0  # SAFE
        elif slope <= self.slope_caution_max:
            return 1  # CAUTION
        else:
            return 2  # HAZARD
```

### Option 2: Random Forest Classifier

**Best for:** Better accuracy, handles complex patterns

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json

# Load dataset
with open('data/cache/hazard_dataset.json', 'r') as f:
    dataset = json.load(f)

# Prepare features and labels
X = []
y = []
for sample in dataset['samples']:
    features = sample['features']
    X.append([
        features['slope_deg'],
        features['curvature'],
        features['roughness'],
        features['local_slope_variance']
    ])
    y.append(sample['label'])

X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=50,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

### Option 3: Neural Network (Advanced)

**Best for:** Maximum accuracy, large datasets

```python
import torch
import torch.nn as nn

class HazardNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# Training loop
model = HazardNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
```

---

## Evaluation Metrics

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
    target_names=['SAFE', 'CAUTION', 'HAZARD']))
```

### Key Metrics

**1. Precision**
- What % of predicted hazards are actual hazards?
- Critical for avoiding false alarms

**2. Recall**
- What % of actual hazards are detected?
- Critical for safety (minimize missed hazards)

**3. F1-Score**
- Harmonic mean of precision and recall
- Balance between false alarms and missed hazards

**4. Safety Score**
- Custom metric prioritizing recall on HAZARD class
- Formula: `0.6 * recall_hazard + 0.4 * overall_accuracy`

### Acceptable Performance

| Metric | Minimum | Target |
|--------|---------|--------|
| Overall Accuracy | 85% | 90%+ |
| Hazard Recall | 90% | 95%+ |
| Safe Precision | 90% | 95%+ |
| Safety Score | 0.85 | 0.90+ |

---

## Model Deployment

### 1. Save Model

```python
import joblib

# Save scikit-learn model
joblib.dump(model, 'models/hazard_classifier.pkl')

# Save PyTorch model
torch.save(model.state_dict(), 'models/hazard_net.pth')

# Save metadata
metadata = {
    'model_type': 'RandomForestClassifier',
    'features': ['slope_deg', 'curvature', 'roughness', 'slope_variance'],
    'classes': ['SAFE', 'CAUTION', 'HAZARD'],
    'accuracy': float(accuracy),
    'trained_on': datetime.now().isoformat()
}
with open('models/hazard_classifier_meta.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### 2. Load and Use

```python
# Load model
model = joblib.load('models/hazard_classifier.pkl')

# Predict on new data
features = np.array([[12.5, 0.02, 5.3, 3.2]])
prediction = model.predict(features)[0]
confidence = model.predict_proba(features)[0]

print(f"Prediction: {['SAFE', 'CAUTION', 'HAZARD'][prediction]}")
print(f"Confidence: {confidence[prediction]:.2%}")
```

### 3. Integration with DEM Processor

```python
# In src/data_pipeline/dem_processor.py

class DEMProcessor:
    def __init__(self, hazard_model_path=None):
        self.hazard_model = None
        if hazard_model_path:
            self.hazard_model = joblib.load(hazard_model_path)
    
    def identify_hazards_ml(self, elevation, pixel_size):
        """Use ML model for hazard detection."""
        features = self._compute_features(elevation, pixel_size)
        
        # Flatten for prediction
        h, w = elevation.shape
        X = np.column_stack([
            features['slope'].flatten(),
            features['curvature'].flatten(),
            features['roughness'].flatten(),
            features['slope_variance'].flatten()
        ])
        
        # Predict
        predictions = self.hazard_model.predict(X)
        hazard_map = predictions.reshape((h, w))
        
        return {
            'hazard_map': hazard_map,
            'safe_percentage': (hazard_map == 0).sum() / hazard_map.size * 100,
            'caution_percentage': (hazard_map == 1).sum() / hazard_map.size * 100,
            'hazard_percentage': (hazard_map == 2).sum() / hazard_map.size * 100
        }
```

---

## Training Scripts

### Build Dataset
```bash
python scripts/build_hazard_dataset.py \
  --dem data/dem/jezero.tif \
  --output data/cache/hazard_dataset.json \
  --samples 10000
```

### Train Model
```bash
python scripts/train_hazard_model.py \
  --dataset data/cache/hazard_dataset.json \
  --model-type random_forest \
  --output models/hazard_classifier.pkl
```

### Evaluate Model
```bash
python scripts/eval_hazard_model.py \
  --dataset data/cache/hazard_dataset.json \
  --model models/hazard_classifier.pkl
```

---

## Advanced Topics

### Transfer Learning
- Pre-train on Earth terrain data
- Fine-tune on Mars DEMs
- Requires labeled Earth DEM datasets

### Ensemble Methods
- Combine threshold + ML + GPT-4V
- Vote or weighted average
- Improves robustness

### Active Learning
- Model identifies uncertain predictions
- Human labels ambiguous cases
- Iteratively improve model

### Real-time Inference
- Optimize model size for edge deployment
- Use quantization (int8) for faster inference
- Deploy on rover compute hardware

---

## Best Practices

1. **Balance Dataset**: Equal samples per class
2. **Feature Scaling**: Normalize features before training
3. **Cross-validation**: Use k-fold CV for robust evaluation
4. **Regularization**: Prevent overfitting with dropout/pruning
5. **Version Control**: Track model versions with metadata
6. **Safety First**: Prefer false alarms over missed hazards
7. **Human Validation**: Always verify predictions on critical missions

---

## Troubleshooting

### Low Accuracy
- Add more training samples
- Engineer better features
- Try different model architectures
- Check for data quality issues

### Overfitting
- Reduce model complexity
- Add regularization
- Use more training data
- Increase dropout rate

### Class Imbalance
- Use class weights in loss function
- Oversample minority class (HAZARD)
- Use SMOTE for synthetic samples
- Adjust decision threshold

---

## Resources

- **NASA PDS**: https://pds.nasa.gov/
- **Scikit-learn**: https://scikit-learn.org/
- **PyTorch**: https://pytorch.org/
- **GDAL**: https://gdal.org/

---

## Next Steps

1. Build initial dataset from available DEMs
2. Train baseline threshold model
3. Train ML model and evaluate
4. Deploy best model to production
5. Monitor performance and retrain as needed
