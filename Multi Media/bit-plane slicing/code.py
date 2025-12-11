import numpy as np
from PIL import Image
import os
# Read grayscale images
dark_img = np.array(Image.open("dark_image.jpg").convert("L"))
brt_img = np.array(Image.open("bright_image.jpg").convert("L"))
out_dir = "bitplane_results/"
os.makedirs(out_dir, exist_ok=True)
# Function: reconstruct using lowest 3 bit-planes
# keeps bits 0,1,2 → scales back to 0–255 range
def rec_low3(img):
    bits3 = img & 7          # keep only last 3 bits (0–7)
    return (bits3 * 255 // 7).astype(np.uint8)
# ---- Low-light image ----
dark_rec = rec_low3(dark_img)                                # reconstructed
dark_diff = np.abs(dark_img - dark_rec).astype(np.uint8)    # diff of original-reconstructed using lowest 3 bit-planes for light img
Image.fromarray(dark_rec).save(f"{out_dir}/low_reconstructed.png")
Image.fromarray(dark_diff).save(f"{out_dir}/low_diff.png")
# ---- Bright-light image ----
brt_rec = rec_low3(brt_img)                                # reconstructed
brt_diff = np.abs(brt_img - brt_rec).astype(np.uint8)      # diff of original-reconstructed using lowest 3 bit-planes for bright img
Image.fromarray(brt_rec).save(f"{out_dir}/bright_reconstructed.png")
Image.fromarray(brt_diff).save(f"{out_dir}/bright_diff.png")
print("Saved: dark_reconstructed.png, dark_diff.png, bright_reconstructed.png, bright_diff.png")