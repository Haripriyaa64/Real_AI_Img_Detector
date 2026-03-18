
# from flask import Flask, render_template, request
# import tensorflow as tf
# import cv2
# import numpy as np
# import os

# app = Flask(__name__)

# # Load trained model
# model = tf.keras.models.load_model("ai_image_detector.keras")

# # Upload folder
# UPLOAD_FOLDER = "static/uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# IMG_SIZE = 224

# def preprocess(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = img / 255.0
#     return np.expand_dims(img, axis=0)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     confidence = None
#     image_path = None

#     if request.method == "POST":
#         file = request.files["image"]

#         if file:
#             image_path = os.path.join(UPLOAD_FOLDER, file.filename)
#             file.save(image_path)

#             img = preprocess(image_path)
#             pred = model.predict(img)[0][0]

#             if pred > 0.5:
#                 prediction = "AI Generated Image"
#                 confidence = round(pred * 100, 2)
#             else:
#                 prediction = "Real Image"
#                 confidence = round((1 - pred) * 100, 2)

#     return render_template(
#         "index.html",
#         prediction=prediction,
#         confidence=confidence,
#         image_path=image_path
#     )

# if __name__ == "__main__":
#     app.run(debug=True)

import tensorflow as tf
import cv2
import numpy as np
import gradio as gr
import tempfile
import os

# Load trained model
model = tf.keras.models.load_model("ai_image_detector.keras")
IMG_SIZE = 224


# ---------------- IMAGE DETECTION ---------------- #
def detect_image(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)[0][0]

    ai_conf = round(pred * 100, 2)
    real_conf = round((1 - pred) * 100, 2)

    label = "🧠 AI Generated Image" if pred > 0.5 else "📸 Real Image"
    confidence = f"AI: {ai_conf}% | Real: {real_conf}%"

    return label, confidence


# ---------------- VIDEO DETECTION ---------------- #
def detect_video(video):
    cap = cv2.VideoCapture(video)
    predictions = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // 20, 1)  # sample ~20 frames

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frame = np.expand_dims(frame, axis=0)

            pred = model.predict(frame)[0][0]
            predictions.append(pred)

        frame_id += 1

    cap.release()

    if len(predictions) == 0:
        return "❌ Unable to analyze video", "N/A"

    avg_pred = np.mean(predictions)

    ai_conf = round(avg_pred * 100, 2)
    real_conf = round((1 - avg_pred) * 100, 2)

    label = "🧠 AI Generated Video" if avg_pred > 0.5 else "📹 Real Video"
    confidence = f"AI: {ai_conf}% | Real: {real_conf}%"

    return label, confidence


# ---------------- GRADIO UI ---------------- #
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🔍 AI vs Real Image & Video Detector")
    gr.Markdown(
        "Detect whether an **image or video** is AI-generated or real.\n\n"
        "⚠️ Results depend on quality, lighting, and dataset coverage."
    )

    with gr.Tabs():
        with gr.Tab("🖼️ Image Detection"):
            img_input = gr.Image(type="numpy", label="Upload Image")
            img_btn = gr.Button("Detect Image")
            img_label = gr.Label(label="Prediction")
            img_conf = gr.Textbox(label="Confidence")

            img_btn.click(
                detect_image,
                inputs=img_input,
                outputs=[img_label, img_conf]
            )

        with gr.Tab("🎥 Video Detection"):
            vid_input = gr.Video(label="Upload Video")
            vid_btn = gr.Button("Detect Video")
            vid_label = gr.Label(label="Prediction")
            vid_conf = gr.Textbox(label="Confidence")

            vid_btn.click(
                detect_video,
                inputs=vid_input,
                outputs=[vid_label, vid_conf]
            )

demo.launch(show_error=True)

from flask import Flask, render_template, Response, jsonify
from detect_emotion import generate_frames, get_last_emotion

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/emotion_data')
def emotion_data():
    """Returns the latest detected emotion + confidence scores as JSON."""
    return jsonify(get_last_emotion())


if __name__ == "__main__":
    app.run(debug=True)

