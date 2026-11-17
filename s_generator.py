import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE_DIR = Path(__file__).resolve().parent
S_DIR = BASE_DIR / "s_images"
ANTI_S_DIR = BASE_DIR / "anti_s_images"

S_DIR.mkdir(exist_ok=True)
ANTI_S_DIR.mkdir(exist_ok=True)

def generate_full_dataset(num_samples=5000):
    print(f"Generating {num_samples} 'S' and 'Anti-S' images...")
    
    for i in range(num_samples):
        # Create 35x35 white RGBA canvas
        canvas = Image.new("RGBA", (35, 35), color="white")
        
        # Random scale and rotation
        font_size = random.randint(10, 23)
        angle = random.randint(0, 360)
        
        # Draw 'S'
        d = ImageDraw.Draw(canvas)
        try:
            fnt = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Italic.ttf", font_size)
        except OSError:
            fnt = ImageFont.load_default()
            
        d.text((10, 10), "S", font=fnt, fill=(0, 0, 0))
        
        # Apply rotation
        rotated_s = canvas.rotate(angle, expand=False, fillcolor="white")
        rotated_s.save(S_DIR / f"s_image_{i}.png")
        
        # Mirror for Anti-S
        anti_s = ImageOps.mirror(rotated_s)
        anti_s.save(ANTI_S_DIR / f"anti_s_image_{i}.png")

if __name__ == "__main__":
    generate_full_dataset(5000)
    print("✓ Generated 10,000 full-resolution images.")
