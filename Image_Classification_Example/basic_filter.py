from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import os
import random

def apply_blur_filter(image_path, output_path="blurred_image.png"):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))

        plt.imshow(img_blurred)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Processed image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")


def apply_spaghetti_filter(image_path, output_path="spaghetti_monster.png", noodle_count=50, meatball_count=10):
    """
    Transforms the image into a 'spaghetti monster' by overlaying noodly spaghetti strands
    and meatball-like circles.
    """
    try:
        img = Image.open(image_path).convert('RGBA')
        img_resized = img.resize((256, 256))
        overlay = Image.new('RGBA', img_resized.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw noodles
        for _ in range(noodle_count):
            # random noodle color (pale yellow/orange)
            color = (random.randint(200, 255), random.randint(180, 230), random.randint(50, 100), 180)
            # random start and end points
            x0, y0 = random.randint(0, 255), random.randint(0, 255)
            x1, y1 = random.randint(0, 255), random.randint(0, 255)
            # generate intermediate points for a wiggly noodle
            points = [(x0, y0)]
            for t in range(1, 5):
                xt = x0 + (x1 - x0) * t / 5 + random.randint(-20, 20)
                yt = y0 + (y1 - y0) * t / 5 + random.randint(-20, 20)
                points.append((xt, yt))
            points.append((x1, y1))
            # draw the noodle
            draw.line(points, fill=color, width=random.randint(5, 12))

        # Draw meatballs
        for _ in range(meatball_count):
            # random position and size
            cx, cy = random.randint(0, 255), random.randint(0, 255)
            r = random.randint(10, 25)
            bbox = [cx - r, cy - r, cx + r, cy + r]
            # meatball color (brownish)
            color = (random.randint(80, 120), random.randint(40, 60), random.randint(20, 30), 200)
            draw.ellipse(bbox, fill=color)

        # Composite overlay
        combined = Image.alpha_composite(img_resized, overlay).convert('RGB')
        combined.save(output_path)
        print(f"Spaghetti monster image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    print("Image Processor (type 'exit' to quit)\nAvailable filters: blur, spaghetti")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue

        # choose filter
        choice = input("Choose filter ('blur' or 'spaghetti'): ").strip().lower()
        base, ext = os.path.splitext(image_path)

        if choice == 'blur':
            output_file = f"{base}_blurred{ext}"
            apply_blur_filter(image_path, output_file)
        elif choice == 'spaghetti':
            output_file = f"{base}_spaghetti{ext}"
            apply_spaghetti_filter(image_path, output_file)
        else:
            print("Unknown filter choice. Please select 'blur' or 'spaghetti'.")