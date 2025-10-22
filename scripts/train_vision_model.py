#!/usr/bin/env python3
"""
Train Vision-Based Hazard Detection from Mastcam Images

Uses MSL/Perseverance Mastcam raw images to train a vision model
for terrain hazard classification.

Approaches:
1. GPT-4V labeling + traditional CNN training
2. CLIP-based zero-shot classification
3. Vision Transformer fine-tuning
"""

import sys
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - install with: pip install torch torchvision")

# Optional: timm for state-of-the-art backbones
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not available - install with: pip install timm")

# Optional: open-clip for local zero-shot labeling
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    logger.warning("open-clip-torch not available - install with: pip install open-clip-torch")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available - install with: pip install openai")


class MastcamDataset(Dataset):
    """Dataset for Mastcam terrain images."""
    
    def __init__(self, image_dir: Path, labels_file: Path = None, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Find all image files
        self.image_paths = list(self.image_dir.glob("*.jpg")) + \
                          list(self.image_dir.glob("*.png")) + \
                          list(self.image_dir.glob("*.JPG"))
        
        # Load labels if available
        self.labels = {}
        if labels_file and labels_file.exists():
            with open(labels_file, 'r') as f:
                raw = json.load(f)
                # Support both flat mapping and {"labels": {...}}
                if isinstance(raw, dict) and 'labels' in raw:
                    lbls = raw['labels']
                else:
                    lbls = raw
                # Normalize to filename -> int label
                self.labels = {}
                for k, v in lbls.items():
                    if isinstance(v, dict) and 'hazard_level' in v:
                        self.labels[k] = int(v['hazard_level'])
                    else:
                        try:
                            self.labels[k] = int(v)
                        except Exception:
                            self.labels[k] = -1
        
        logger.info(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get label if available
        label = self.labels.get(img_path.name, -1)  # -1 = unlabeled
        
        return image, label, str(img_path)


class CLIPLabeler:
    """Use OpenCLIP to pseudo-label images locally (no API)."""
    def __init__(self, model_name: str = 'ViT-B-32', pretrained: str = 'laion2b_s34b_b79k', device: str = None):
        if not OPENCLIP_AVAILABLE:
            raise ImportError("open-clip-torch required: pip install open-clip-torch")
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        # Define class prompts
        self.classes = [
            (0, 'SAFE terrain: flat, smooth, sandy, no large rocks, slope less than 15 degrees, safe for rover driving'),
            (1, 'CAUTION terrain: some rocks or moderate slopes between 15 and 30 degrees, driveable with care'),
            (2, 'HAZARD terrain: large rocks, very rough ground, or steep slopes greater than 30 degrees, unsafe for rover')
        ]
        self.text_tokens = self.tokenizer([c[1] for c in self.classes]).to(self.device)

    @torch.no_grad()
    def label_image(self, image_path: Path):
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(self.text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        label = int(self.classes[idx][0])
        conf = float(probs[idx])
        return {
            'hazard_level': label,
            'confidence': conf,
            'reasoning': 'CLIP zero-shot classification using terrain safety prompts',
            'features': {}
        }

    def label_dataset(self, image_dir: Path, output_file: Path, max_images: int = None):
        image_dir = Path(image_dir)
        images = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')) + list(image_dir.glob('*.JPG'))
        if max_images:
            images = images[:max_images]
        labels = {}
        stats = {'safe': 0, 'caution': 0, 'hazard': 0, 'failed': 0}
        for i, img_path in enumerate(images, 1):
            try:
                res = self.label_image(img_path)
                labels[img_path.name] = res
                lvl = res['hazard_level']
                if lvl == 0: stats['safe'] += 1
                elif lvl == 1: stats['caution'] += 1
                else: stats['hazard'] += 1
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"Label failed for {img_path.name}: {e}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({'labels': labels, 'stats': stats, 'timestamp': datetime.utcnow().isoformat()}, f, indent=2)
        logger.info(f"\n CLIP labeled {len(labels)} images; stats: {stats} -> {output_file}")
        return labels


class GPT4VLabeler:
    """Use GPT-4V to label Mastcam images with terrain hazards."""
    
    def __init__(self, api_key: str = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package required: pip install openai")
        
        # Use explicit key if provided, else fall back to environment variable
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
    
    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def label_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Label a Mastcam image with GPT-4V.
        
        Returns:
            {
                'hazard_level': 0/1/2,  # SAFE/CAUTION/HAZARD
                'confidence': float,
                'reasoning': str,
                'features': {
                    'rocks': bool,
                    'steep_slope': bool,
                    'sandy': bool,
                    'rough_terrain': bool
                }
            }
        """
        base64_image = self.encode_image(image_path)
        
        prompt = """Analyze this Mars terrain image from NASA's Mastcam.

Classify the terrain safety for a rover:
- SAFE (0): Flat, smooth, sandy terrain. Slopes <15°. Safe for driving.
- CAUTION (1): Moderate rocks or slopes 15-30°. Driveable with care.
- HAZARD (2): Large rocks, steep slopes >30°, or very rough terrain. Unsafe.

Respond in JSON format:
{
  "hazard_level": 0/1/2,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation",
  "features": {
    "rocks": true/false,
    "steep_slope": true/false,
    "sandy": true/false,
    "rough_terrain": true/false
  }
}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"GPT-4V labeling failed: {e}")
            return None
    
    def label_dataset(self, image_dir: Path, output_file: Path, 
                     max_images: int = None) -> Dict[str, Any]:
        """Label all images in a directory."""
        
        image_dir = Path(image_dir)
        images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if max_images:
            images = images[:max_images]
        
        logger.info(f"Labeling {len(images)} images with GPT-4V...")
        
        labels = {}
        stats = {'safe': 0, 'caution': 0, 'hazard': 0, 'failed': 0}
        
        for i, img_path in enumerate(images, 1):
            logger.info(f"  [{i}/{len(images)}] {img_path.name}")
            
            result = self.label_image(img_path)
            
            if result:
                labels[img_path.name] = result
                level = result['hazard_level']
                if level == 0:
                    stats['safe'] += 1
                elif level == 1:
                    stats['caution'] += 1
                else:
                    stats['hazard'] += 1
            else:
                stats['failed'] += 1
        
        # Save labels
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'labels': labels,
                'stats': stats,
                'timestamp': datetime.utcnow().isoformat()
            }, f, indent=2)
        
        logger.info(f"\n Labeled {len(labels)} images")
        logger.info(f"  SAFE: {stats['safe']}")
        logger.info(f"  CAUTION: {stats['caution']}")
        logger.info(f"  HAZARD: {stats['hazard']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"\n Saved to: {output_file}")
        
        return labels


class TerrainCNN(nn.Module):
    """Simple CNN for terrain classification."""
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(arch: str, num_classes: int = 3):
    """Build best-practice image classifier.
    arch options: 'vit_b16' (default), 'convnext_tiny', 'resnet50', 'cnn_simple'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if arch == 'cnn_simple' or not TIMM_AVAILABLE:
        model = TerrainCNN(num_classes=num_classes)
        input_size = 224
    else:
        if arch == 'vit_b16':
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
            input_size = 224
        elif arch == 'convnext_tiny':
            model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
            input_size = 224
        elif arch == 'resnet50':
            model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
            input_size = 224
        else:
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
            input_size = 224
    return model.to(device), input_size


def train_model(train_dir: Path, labels_file: Path, output_model: Path, epochs: int = 20, arch: str = 'vit_b16'):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required: pip install torch torchvision")

    # Build model
    model, input_size = build_model(arch, num_classes=3)

    logger.info(f"Training {arch} on labeled images...")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = MastcamDataset(train_dir, labels_file, transform=transform)

    labeled_indices = [i for i in range(len(dataset)) if dataset[i][1] != -1]
    if len(labeled_indices) == 0:
        raise ValueError("No labeled images found!")

    logger.info(f"Training on {len(labeled_indices)} labeled images")

    # Split train/val
    split = int(0.8 * len(labeled_indices)) or 1
    train_indices = labeled_indices[:split]
    val_indices = labeled_indices[split:] or labeled_indices[-1:]

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=8, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_indices), batch_size=8)

    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_correct = train_total = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, pred = outputs.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()
        train_acc = 100.0 * train_correct / max(1, train_total)

        # Val
        model.eval(); val_correct = val_total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
        val_acc = 100.0 * val_correct / max(1, val_total)

        logger.info(f"Epoch {epoch+1}/{epochs} - Train {train_acc:.2f}% | Val {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_model.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_model)
            logger.info(f"   Saved best model ({val_acc:.2f}% val acc)")

    logger.info(f"\n Training complete. Best val acc: {best_val_acc:.2f}%")
    # Save simple metrics JSON
    metrics_path = output_model.with_suffix('.json')
    try:
        import json, time
        with open(metrics_path, 'w') as f:
            json.dump({
                'best_val_acc': best_val_acc,
                'epochs': epochs,
                'arch': arch,
                'timestamp': time.time()
            }, f, indent=2)
        logger.info(f" Metrics saved to {metrics_path}")
    except Exception as e:
        logger.warning(f"Could not save metrics: {e}")
    return model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train vision model on Mastcam images"
    )
    parser.add_argument('--mode', choices=['label', 'train'], required=True,
                       help='Label images with GPT-4V or train CNN')
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing Mastcam images')
    parser.add_argument('--labels', type=str, default='data/cache/mastcam_labels.json',
                       help='Path to labels file')
    parser.add_argument('--output', type=str, default='models/terrain_vision_model.pth',
                       help='Output model path')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Max images to label (for testing)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Training epochs')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--label-mode', choices=['gpt4v','clip'], default='clip',
                       help='Pseudo-labeling method')
    parser.add_argument('--arch', choices=['vit_b16', 'convnext_tiny', 'resnet50', 'cnn_simple'], default='vit_b16',
                       help='Backbone architecture for training')
    
    args = parser.parse_args()
    
    if args.mode == 'label':
        # Label images
        if args.label_mode == 'gpt4v':
            labeler = GPT4VLabeler(api_key=args.api_key)
        else:
            labeler = CLIPLabeler()
        labeler.label_dataset(Path(args.images), Path(args.labels), max_images=args.max_images)

    elif args.mode == 'train':
        # Train with selected backbone
        train_model(Path(args.images), Path(args.labels), Path(args.output), epochs=args.epochs, arch=args.arch)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
