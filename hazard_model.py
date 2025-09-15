from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import os


@dataclass
class HazardModel:
    """Optional local hazard classifier wrapper.

    Supports both PyTorch models (.pt) and trained JSON threshold models.
    Falls back to no-op predictions otherwise. This is used to augment LLM terrain analysis.
    """

    model_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/hazard_classifier.pt"))
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path:
            self.model_path = os.path.abspath(model_path)
        else:
            self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/hazard_classifier.pt"))
        
        # Load model config if it's a JSON threshold model
        self.model_config = None
        self.torch_model = None
        self.torch_model_info = None
        
        if self.model_path.endswith('.json') and os.path.exists(self.model_path):
            with open(self.model_path) as f:
                self.model_config = json.load(f)
        elif self.model_path.endswith('.pt') and os.path.exists(self.model_path):
            self._load_torch_model()

    def _load_torch(self):  # pragma: no cover - optional dependency
        try:
            import torch  # type: ignore
            return torch
        except Exception:
            return None
    
    def _load_torch_model(self):  # pragma: no cover - optional dependency
        """Load PyTorch CNN model."""
        torch = self._load_torch()
        if torch is None:
            return
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # New format with metadata
                self.torch_model_info = checkpoint
                model_state = checkpoint['model_state_dict']
                
                # Recreate CNN architecture
                model = self._create_terrain_cnn(
                    checkpoint.get('input_channels', 3),
                    checkpoint.get('num_classes', 1)
                )
                model.load_state_dict(model_state)
                model.eval()
                self.torch_model = model
            else:
                # Legacy format - direct state dict
                print("Warning: Loading legacy model format")
                
        except Exception as e:
            print(f"Failed to load PyTorch model: {e}")
    
    def _create_terrain_cnn(self, input_channels=3, num_classes=1):
        """Create TerrainCNN architecture matching training script."""
        torch = self._load_torch()
        if torch is None:
            return None
        
        import torch.nn as nn  # type: ignore
        
        class TerrainCNN(nn.Module):
            def __init__(self, input_channels=3, num_classes=1):
                super().__init__()
                self.features = nn.Sequential(
                    # First conv block
                    nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Second conv block
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    # Third conv block
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(64, num_classes),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return TerrainCNN(input_channels, num_classes)

    def is_available(self) -> bool:
        # Available if we have a JSON threshold model or loaded PyTorch model
        return (self.model_config is not None) or (self.torch_model is not None)

    def predict(self, image_bgr) -> Dict[str, Any]:  # image as numpy array (H,W,3)
        if not self.is_available():
            return {"hazard_score": 0.0, "notes": "model unavailable"}
        torch = self._load_torch()
        import numpy as np  # lazy import fine

        try:
            # Minimal example: downsample and compute a dummy mean intensity feature
            x = image_bgr.astype("float32") / 255.0
            feat = float(x.mean())
            # Fake mapping: darker -> potentially hazardous (placeholder)
            hazard = max(0.0, min(1.0, 1.0 - feat))
            return {"hazard_score": hazard, "notes": "heuristic placeholder"}
        except Exception as e:
            return {"hazard_score": 0.0, "notes": f"prediction failed: {e}"}
    
    def assess_hazard(self, lat: float, lon: float) -> float:
        """Assess hazard for a specific coordinate using trained models.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            Hazard score between 0.0 (safe) and 1.0 (hazardous)
        """
        # Try CNN model first
        if self.torch_model is not None:
            return self._assess_with_cnn(lat, lon)
        
        # Fall back to threshold model
        if self.model_config is not None:
            model_type = self.model_config.get('type', 'threshold')
            if model_type == 'threshold':
                threshold = self.model_config.get('value', 0.5)
                # Simple threshold-based assessment
                # For demo: use coordinate-based features to simulate terrain roughness
                roughness = abs(lat * 0.01 + lon * 0.005) % 1.0  # Synthetic roughness
                return 1.0 if roughness > threshold else 0.0
        
        # Fallback to basic coordinate-based heuristic
        # Higher latitudes and certain longitude ranges might be more hazardous
        base_hazard = abs(lat) / 90.0  # Higher latitudes = more hazardous
        return min(1.0, base_hazard * 0.3)  # Scale down to reasonable range
    
    def _assess_with_cnn(self, lat: float, lon: float) -> float:
        """Use CNN model to assess terrain hazard."""
        torch = self._load_torch()
        if torch is None or self.torch_model is None:
            return 0.0
        
        try:
            import numpy as np
            
            # Generate synthetic terrain patch (same as training)
            patch = self._generate_terrain_patch(lat, lon)
            
            # Convert to tensor and add batch dimension
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                prediction = self.torch_model(patch_tensor)
                hazard_score = float(prediction.squeeze().item())
            
            return hazard_score
        
        except Exception as e:
            print(f"CNN inference failed: {e}")
            return 0.0
    
    def _generate_terrain_patch(self, lat: float, lon: float, patch_size: int = 64) -> np.ndarray:
        """Generate synthetic terrain patch (matches training data generation)."""
        import numpy as np
        
        # Use hazard fraction from model or estimate from coordinates
        if self.torch_model_info and 'threshold' in self.torch_model_info:
            threshold = self.torch_model_info['threshold']
            # Estimate hazard level based on coordinate features
            roughness_estimate = abs(lat * 0.01 + lon * 0.005) % 1.0
            hazard_frac = 1.0 if roughness_estimate > threshold else 0.3
        else:
            # Default hazard estimation
            hazard_frac = min(1.0, abs(lat * 0.01 + lon * 0.005) % 1.0)
        
        # Create base terrain heightmap
        x = np.linspace(0, 10, patch_size)
        y = np.linspace(0, 10, patch_size)
        X, Y = np.meshgrid(x, y)
        
        # Base elevation with coordinate-dependent features
        base_elevation = np.sin(X * 0.3 + lat * 0.1) * np.cos(Y * 0.2 + lon * 0.05)
        
        # Add hazard-dependent roughness
        noise_scale = hazard_frac * 2.0 + 0.1
        roughness = np.random.normal(0, noise_scale, (patch_size, patch_size))
        
        # Combine elevation and roughness
        elevation = base_elevation + roughness
        
        # Create slope magnitude (gradient)
        grad_y, grad_x = np.gradient(elevation)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        
        # Create aspect (direction of slope)
        aspect = np.arctan2(grad_y, grad_x)
        
        # Stack as 3-channel image (elevation, slope, aspect)
        patch = np.stack([elevation, slope, aspect], axis=0)
        
        # Normalize channels
        for i in range(3):
            patch[i] = (patch[i] - patch[i].mean()) / (patch[i].std() + 1e-8)
        
        return patch
