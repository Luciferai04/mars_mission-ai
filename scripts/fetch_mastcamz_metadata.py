#!/usr/bin/env python3
"""
Fetch Mastcam-Z metadata from NASA PDS for specific sol.

Usage:
    python scripts/fetch_mastcamz_metadata.py --sol 1400 --output data/cache/mastcamz_sol1400.json
"""

import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.nasa_api_client import NASAAPIClient


def main():
    parser = argparse.ArgumentParser(
        description='Fetch Mastcam-Z metadata from NASA PDS'
    )
    parser.add_argument('--sol', type=int, required=True,
                       help='Mars sol number')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--limit', type=int, default=100,
                       help='Maximum number of images (default: 100)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Create client
    client = NASAAPIClient()
    
    print(f"Fetching Mastcam-Z metadata for Sol {args.sol}...")
    
    # Fetch metadata
    images = client.get_mastcamz_metadata(args.sol, limit=args.limit)
    
    if images:
        print(f"Found {len(images)} images")
        
        # Save to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(images, f, indent=2)
        
        print(f"Saved to {output_path}")
        
        if args.verbose:
            print("\nImage summary:")
            for i, img in enumerate(images[:5], 1):
                print(f"  {i}. {img['product_id']} ({img['instrument']})")
            if len(images) > 5:
                print(f"  ... and {len(images) - 5} more")
    else:
        print("No images found")
        sys.exit(1)


if __name__ == '__main__':
    main()
