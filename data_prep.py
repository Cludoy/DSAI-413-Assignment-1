import os
from pathlib import Path
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

def compile_fan_kit_images(fankit_dir: str, output_path: str, num_images: int = 5):
    """
    Recursively scans the provided fankit directory for `.png` and `.jpg` files,
    selects a limited subset, and concatenates them into a single PDF.
    Handles corrupt images or bad color modes intrinsically.
    """
    print(f"Scanning {fankit_dir} for visual assets...")
    root_path = Path(fankit_dir)
    image_paths = []
    
    # Recursively find images
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(list(root_path.rglob(ext)))
        
    if not image_paths:
        print("No images found in fankit directory.")
        return
        
    print(f"Found {len(image_paths)} images. Processing subset...")
    # Select the subset
    selected_paths = image_paths[:num_images]
    pdf_images = []
    
    for path in selected_paths:
        try:
            with Image.open(path) as img:
                # Convert to RGB to avoid alpha channel / mode mismatch errors during PDF serialization
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Load the full data payload into memory before context manager closes
                pdf_images.append(img.copy())
                print(f"Loaded {path.name}")
        except Exception as e:
            print(f"Failed to process {path.name}: {e}")

    if pdf_images:
        try:
             pdf_images[0].save(
                 output_path,
                 save_all=True,
                 append_images=pdf_images[1:]
             )
             print(f"Successfully compiled Fan Kit to {output_path}")
        except Exception as e:
             print(f"Error saving PDF: {e}")

def main():
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    fankit_dir = "fankit"
    fankit_output_path = str(docs_dir / "fankit_assets.pdf")
    if Path(fankit_dir).exists():
        compile_fan_kit_images(fankit_dir, fankit_output_path, num_images=8)
    else:
        print(f"Warning: Fan Kit directory '{fankit_dir}' not found. Cannot compile fan kit visuals.")

if __name__ == "__main__":
    main()
