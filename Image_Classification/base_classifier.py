# base_classifier.py — MobileNetV2 classifier with Grad-CAM

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from tensorflow.keras.preprocessing import image as keras_image

# quiet TensorFlow logs
tf.get_logger().setLevel("ERROR")

IMG_SIZE = (224, 224)
model = MobileNetV2(weights="imagenet")

def _find_last_conv_layer_name(m):
    # Walk backwards through layers, grab the last one that looks like a conv map
    for layer in reversed(m.layers):
        try:
            shp = layer.output_shape
            if len(shp) == 4:
                return layer.name
        except Exception:
            continue
    # If the above doesn’t hit, fall back to the usual MobileNetV2 layers
    for name in ["Conv_1", "out_relu"]:
        if name in [l.name for l in m.layers]:
            return name
    raise RuntimeError("Couldn’t find a conv layer to use for Grad-CAM.")

LAST_CONV = _find_last_conv_layer_name(model)

def load_batch(img_path):
    # load image, resize, preprocess, batchify
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    return img, arr

def classify(img_path):
    # run the image through the model and print the top-3 guesses
    pil_img, batch = load_batch(img_path)
    preds = model.predict(batch, verbose=0)
    top3 = decode_predictions(preds, top=3)[0]
    print(f"\nTop-3 Predictions for {img_path}")
    for i, (_, label, score) in enumerate(top3, 1):
        print(f"{i}: {label} ({score:.2f})")
    idx = int(np.argmax(preds[0]))
    return pil_img, batch, idx

def make_gradcam_heatmap(batch, class_index=None):
    # grab feature maps from the last conv layer + gradients from target class
    conv_layer = model.get_layer(LAST_CONV)
    grad_model = tf.keras.models.Model(
        [model.inputs], [conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        class_score = preds[:, class_index]

    grads = tape.gradient(class_score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))       # avg gradients across map
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)   # weight maps by gradients
    heatmap = tf.maximum(heatmap, 0)                      # throw out negatives
    maxv = tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    heatmap = heatmap / maxv
    return heatmap.numpy()

def save_overlay(pil_img, heatmap, out_path, alpha=0.4):
    # resize heatmap, color it, blend it on top of the original image
    import matplotlib.cm as cm
    from PIL import Image

    heatmap = np.uint8(255 * heatmap)
    hmap_img = Image.fromarray(heatmap).resize(pil_img.size)

    col = cm.get_cmap("jet")(np.array(hmap_img)/255.0)[:, :, :3]
    col = (col * 255).astype(np.uint8)
    overlay = Image.fromarray(col)

    base = pil_img.convert("RGBA")
    overlay = overlay.convert("RGBA")
    blended = Image.blend(base, overlay, alpha)
    blended.save(out_path)
    return out_path

def run_one(img_path):
    # run classification + Grad-CAM on a single image
    pil_img, batch, idx = classify(img_path)
    try:
        heat = make_gradcam_heatmap(batch, idx)
        base = os.path.splitext(os.path.basename(img_path))[0]
        out = os.path.join(os.path.dirname(img_path), f"gradcam_{base}.png")
        save_overlay(pil_img, heat, out)
        print(f"Grad-CAM saved → {out}")
    except Exception as e:
        print(f"[WARN] Grad-CAM failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="Path to image (png/jpg).")
    args = parser.parse_args()

    if args.image and os.path.isfile(args.image):
        run_one(args.image)
        return

    print("Image Classifier with Grad-CAM (type 'exit' to quit)\n")
    while True:
        path = input("Enter image filename: ").strip().strip('"')
        if path.lower() == "exit":
            print("Goodbye!")
            break
        if os.path.isfile(path):
            run_one(path)
        else:
            print("File not found — try again.\n")

if __name__ == "__main__":
    main()

