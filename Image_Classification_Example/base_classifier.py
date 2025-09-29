import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')

# Load model with imagenet weights
base_model = MobileNetV2(weights="imagenet")
# Identify the last convolutional layer
last_conv_layer = base_model.get_layer('Conv_1')
# Create a model that maps input to both conv outputs and predictions
grad_model = Model(inputs=base_model.input,
                   outputs=[last_conv_layer.output, base_model.output])

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Compute gradient model outputs
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    # Gradient of the predicted class wrt conv outputs
    grads = tape.gradient(class_channel, conv_outputs)
    # Compute guided gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # Multiply each channel by corresponding gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    # Overlay
    overlay = heatmap_color * alpha + img
    overlay = np.uint8(overlay)
    return overlay

def classify_and_gradcam(image_path, top=3):
    # Preprocess input
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = base_model.predict(img_array)
    decoded = decode_predictions(preds, top=top)[0]

    # Compute heatmap
    heatmap = make_gradcam_heatmap(img_array, base_model, 'Conv_1')
    overlay = overlay_heatmap(image_path, heatmap)

    # Save or display results
    out_path = image_path.rsplit('.', 1)[0] + '_gradcam.jpg'
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Top-{top} Predictions for {image_path}:")
    for i, (_, label, score) in enumerate(decoded):
        print(f"  {i+1}: {label} ({score:.2f})")
    print(f"GradCAM overlay saved to: {out_path}")

if __name__ == "__main__":
    print("GradCAM Image Classifier (type 'exit' to quit)\n")
    while True:
        path = input("Enter image filename: ").strip()
        if path.lower() == 'exit':
            print('Goodbye!')
            break
        classify_and_gradcam(path)