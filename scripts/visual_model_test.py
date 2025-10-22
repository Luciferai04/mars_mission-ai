#!/usr/bin/env python3
"""
Interactive Visual Model Testing Tool

Creates interactive visualizations to test the trained hazard detection model
with different terrain scenarios and view predictions in real-time.
"""

import sys
import numpy as np
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.ndimage import uniform_filter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class InteractiveModelTester:
    """Interactive visual tester for hazard detection model."""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        
        # Load model
        print(f"Loading model from {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(" Model loaded")
        
        # Generate test terrain
        self.size = 200
        self.terrain = self.generate_terrain()
        
        # Setup interactive plot
        self.setup_interactive_plot()
    
    def generate_terrain(self, crater_height=50, rock_density=2.0):
        """Generate synthetic Mars terrain."""
        
        terrain = np.zeros((self.size, self.size))
        terrain += -2500  # Base elevation
        
        x, y = np.meshgrid(np.linspace(0, 1, self.size), np.linspace(0, 1, self.size))
        
        # Add crater
        crater = crater_height * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.02)
        terrain += crater
        
        # Add rocks
        np.random.seed(42)
        rocks = np.random.normal(0, rock_density, (self.size, self.size))
        terrain += rocks
        
        return terrain
    
    def compute_features(self, terrain):
        """Compute terrain features for model prediction."""
        
        # Gradients
        grad_y, grad_x = np.gradient(terrain)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        slope_deg = np.degrees(np.arctan(slope))
        
        # Curvature
        grad2_y, grad2_x = np.gradient(slope)
        curvature = np.sqrt(grad2_x**2 + grad2_y**2)
        
        # Roughness
        local_mean = uniform_filter(terrain, size=5)
        local_var = uniform_filter((terrain - local_mean)**2, size=5)
        roughness = np.sqrt(local_var)
        
        return {
            'elevation': terrain,
            'slope_deg': slope_deg,
            'curvature': curvature,
            'roughness': roughness
        }
    
    def predict_terrain(self, terrain):
        """Predict hazards for entire terrain."""
        
        metrics = self.compute_features(terrain)
        
        height, width = terrain.shape
        predictions = np.zeros((height, width))
        confidence = np.zeros((height, width))
        
        # Predict for each location
        for i in range(5, height - 5):
            for j in range(5, width - 5):
                feature_vec = [
                    terrain[i, j],
                    metrics['slope_deg'][i, j],
                    metrics['curvature'][i, j],
                    metrics['roughness'][i, j],
                    metrics['slope_deg'][max(0, i-2):i+3, max(0, j-2):j+3].mean(),
                    metrics['slope_deg'][max(0, i-2):i+3, max(0, j-2):j+3].std(),
                    metrics['roughness'][max(0, i-2):i+3, max(0, j-2):j+3].mean(),
                ]
                
                pred = self.model.predict([feature_vec])[0]
                prob = self.model.predict_proba([feature_vec])[0]
                
                predictions[i, j] = pred
                confidence[i, j] = prob[int(pred)]
        
        return predictions, confidence, metrics
    
    def setup_interactive_plot(self):
        """Setup interactive matplotlib visualization."""
        
        # Create figure
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle(' Interactive Mars Hazard Detection Model Testing', 
                         fontsize=16, fontweight='bold')
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        self.ax_terrain = self.fig.add_subplot(gs[0:2, 0])
        self.ax_hazards = self.fig.add_subplot(gs[0:2, 1])
        self.ax_confidence = self.fig.add_subplot(gs[0:2, 2])
        
        self.ax_slope = self.fig.add_subplot(gs[2, 0])
        self.ax_roughness = self.fig.add_subplot(gs[2, 1])
        self.ax_stats = self.fig.add_subplot(gs[2, 2])
        
        # Initial prediction
        self.update_visualization()
        
        # Add sliders
        slider_ax1 = plt.axes([0.15, 0.02, 0.25, 0.02])
        slider_ax2 = plt.axes([0.55, 0.02, 0.25, 0.02])
        
        self.slider_crater = Slider(slider_ax1, 'Crater Height', 0, 100, valinit=50, valstep=5)
        self.slider_rocks = Slider(slider_ax2, 'Rock Density', 0, 5, valinit=2.0, valstep=0.1)
        
        self.slider_crater.on_changed(self.on_slider_change)
        self.slider_rocks.on_changed(self.on_slider_change)
        
        # Add buttons
        button_ax1 = plt.axes([0.85, 0.02, 0.1, 0.03])
        button_ax2 = plt.axes([0.85, 0.06, 0.1, 0.03])
        
        self.btn_random = Button(button_ax1, 'Random Terrain')
        self.btn_save = Button(button_ax2, 'Save Plot')
        
        self.btn_random.on_clicked(self.on_random_click)
        self.btn_save.on_clicked(self.on_save_click)
        
        print("\n" + "="*70)
        print(" INTERACTIVE MODEL TESTING")
        print("="*70)
        print("\nControls:")
        print("  • Crater Height Slider - Adjust terrain steepness")
        print("  • Rock Density Slider - Adjust surface roughness")
        print("  • Random Terrain Button - Generate new random terrain")
        print("  • Save Plot Button - Export current visualization")
        print("\nColor Legend:")
        print("  🟢 GREEN = SAFE (slopes <15°)")
        print("  🟡 YELLOW = CAUTION (slopes 15-30°)")
        print("   RED = HAZARD (slopes >30°)")
        print("\nClose the window to exit.")
        print("="*70 + "\n")
    
    def update_visualization(self):
        """Update all visualizations with current terrain."""
        
        # Predict
        predictions, confidence, metrics = self.predict_terrain(self.terrain)
        
        # Clear axes
        for ax in [self.ax_terrain, self.ax_hazards, self.ax_confidence,
                   self.ax_slope, self.ax_roughness, self.ax_stats]:
            ax.clear()
        
        # Plot 1: Terrain DEM
        im1 = self.ax_terrain.imshow(self.terrain, cmap='terrain', origin='lower')
        self.ax_terrain.set_title('Mars Terrain DEM', fontweight='bold')
        self.ax_terrain.set_xlabel('X (pixels)')
        self.ax_terrain.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=self.ax_terrain, label='Elevation (m)', fraction=0.046)
        
        # Plot 2: Hazard Predictions
        colors = ['green', 'yellow', 'red']
        cmap = matplotlib.colors.ListedColormap(colors)
        im2 = self.ax_hazards.imshow(predictions, cmap=cmap, origin='lower', vmin=0, vmax=2)
        self.ax_hazards.set_title('Hazard Predictions', fontweight='bold')
        self.ax_hazards.set_xlabel('X (pixels)')
        self.ax_hazards.set_ylabel('Y (pixels)')
        
        # Add legend
        import matplotlib.patches as mpatches
        patches = [
            mpatches.Patch(color='green', label='SAFE'),
            mpatches.Patch(color='yellow', label='CAUTION'),
            mpatches.Patch(color='red', label='HAZARD')
        ]
        self.ax_hazards.legend(handles=patches, loc='upper right')
        
        # Plot 3: Confidence
        im3 = self.ax_confidence.imshow(confidence, cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
        self.ax_confidence.set_title('Prediction Confidence', fontweight='bold')
        self.ax_confidence.set_xlabel('X (pixels)')
        self.ax_confidence.set_ylabel('Y (pixels)')
        plt.colorbar(im3, ax=self.ax_confidence, label='Confidence', fraction=0.046)
        
        # Plot 4: Slope
        im4 = self.ax_slope.imshow(metrics['slope_deg'], cmap='hot', origin='lower')
        self.ax_slope.set_title('Slope (degrees)', fontweight='bold')
        plt.colorbar(im4, ax=self.ax_slope, label='°', fraction=0.046)
        
        # Plot 5: Roughness
        im5 = self.ax_roughness.imshow(metrics['roughness'], cmap='plasma', origin='lower')
        self.ax_roughness.set_title('Surface Roughness', fontweight='bold')
        plt.colorbar(im5, ax=self.ax_roughness, label='m', fraction=0.046)
        
        # Plot 6: Statistics
        self.ax_stats.axis('off')
        
        # Calculate stats
        total_pixels = predictions.size
        safe = np.sum(predictions == 0)
        caution = np.sum(predictions == 1)
        hazard = np.sum(predictions == 2)
        
        avg_slope = np.mean(metrics['slope_deg'])
        max_slope = np.max(metrics['slope_deg'])
        avg_confidence = np.mean(confidence[confidence > 0])
        
        stats_text = f"""
        TERRAIN STATISTICS
        ==================
        
        Total Area: {total_pixels:,} locations
        
        Safety Classification:
          🟢 SAFE: {safe:,} ({safe/total_pixels*100:.1f}%)
          🟡 CAUTION: {caution:,} ({caution/total_pixels*100:.1f}%)
           HAZARD: {hazard:,} ({hazard/total_pixels*100:.1f}%)
        
        Terrain Metrics:
          Avg Slope: {avg_slope:.1f}°
          Max Slope: {max_slope:.1f}°
          Avg Confidence: {avg_confidence*100:.1f}%
        
        NASA Traversability:
          {' SAFE TO TRAVERSE' if hazard/total_pixels < 0.3 else ' CAUTION REQUIRED' if hazard/total_pixels < 0.7 else ' HIGH RISK TERRAIN'}
        """
        
        self.ax_stats.text(0.1, 0.5, stats_text, 
                          fontfamily='monospace', fontsize=9,
                          verticalalignment='center')
        
        self.fig.canvas.draw_idle()
    
    def on_slider_change(self, val):
        """Handle slider changes."""
        crater_height = self.slider_crater.val
        rock_density = self.slider_rocks.val
        
        self.terrain = self.generate_terrain(crater_height, rock_density)
        self.update_visualization()
    
    def on_random_click(self, event):
        """Generate random terrain."""
        np.random.seed(None)  # Random seed
        crater_height = np.random.uniform(20, 80)
        rock_density = np.random.uniform(1, 4)
        
        self.slider_crater.set_val(crater_height)
        self.slider_rocks.set_val(rock_density)
        
        self.terrain = self.generate_terrain(crater_height, rock_density)
        self.update_visualization()
        
        print(f"Generated random terrain: crater={crater_height:.1f}, rocks={rock_density:.2f}")
    
    def on_save_click(self, event):
        """Save current visualization."""
        from datetime import datetime
        
        output_dir = Path("data/exports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"interactive_test_{timestamp}.png"
        
        self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
        
        print(f" Saved visualization to: {output_path}")
    
    def show(self):
        """Show interactive plot."""
        plt.show()


def main():
    """Run interactive visual testing."""
    
    model_path = "models/hazard_detector_latest.pkl"
    
    if not Path(model_path).exists():
        print(" Model not found:", model_path)
        print("Please train the model first: python scripts/train_production_model.py")
        return 1
    
    print("\n" + "="*70)
    print(" INTERACTIVE MARS HAZARD DETECTION MODEL TESTING")
    print("="*70)
    
    try:
        tester = InteractiveModelTester(model_path)
        tester.show()
        
        print("\n Interactive testing session ended")
        return 0
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
