# new_filter.py — Neon + Experimental tweaks

import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os

def neon_filter(image_path, output_path="output.png", sat=2.0, noise_level=0.05, sharpen=True):
    try:
        # open and convert to RGB
        img = Image.open(image_path).convert("RGB")

        # boost saturation (default 2.0 = double color intensity)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(sat)

        # add random noise (grain effect)
        if noise_level > 0:
            arr = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, noise_level, arr.shape)
            arr = np.clip(arr + noise, 0, 1)
            img = Image.fromarray((arr * 255).astype(np.uint8))

        # optional sharpen
        if sharpen:
            img = img.filter(ImageFilter.SHARPEN)

        # tint the whole image cyan-blue for that neon vibe
        arr = np.array(img)
        arr[:, :, 0] = arr[:, :, 0] * 0.5   # reduce red
        arr[:, :, 1] = arr[:, :, 1] * 1.2   # boost green
        arr[:, :, 2] = arr[:, :, 2] * 1.5   # boost blue
        arr = np.clip(arr, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

        img.save(output_path)
        print(f"Saved → {output_path}")
    except Exception as e:
        print(f"[Error] {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, help="Path to input image")
    parser.add_argument("-o", "--output", default="neon_output.png", help="Output filename")
    parser.add_argument("--sat", type=float, default=2.0, help="Color saturation (default=2.0)")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level (0-0.2)")
    parser.add_argument("--no-sharpen", action="store_true", help="Disable sharpening")
    args = parser.parse_args()

    neon_filter(
        args.image,
        args.output,
        sat=args.sat,
        noise_level=args.noise,
        sharpen = not args.no_sharpen
    )

if __name__ == "__main__":
    main()
