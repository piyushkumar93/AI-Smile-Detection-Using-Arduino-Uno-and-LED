import cv2
import numpy as np
import serial
import time
from PIL import Image, ImageOps
import tensorflow as tf  # Full TensorFlow for TFLite support on Windows

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Get I/O details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and clean labels
labels = [line.strip().lower() for line in open("labels.txt", "r").readlines()]

# Setup Serial (change COM port if needed)
try:
    arduino = serial.Serial('COM7', 9600)
    time.sleep(2)
    print("Connected to Arduino.")
except Exception as e:
    print("Serial Error:", e)
    arduino = None

# Open webcam
cap = cv2.VideoCapture(0)

# Anti-spam delay
last_sent = None
send_interval = 2  # seconds

while True:
    print("Loop running...")

    ret, frame = cap.read()
    if not ret:
        continue

    # Flip camera view like a mirror
    frame = cv2.flip(frame, 1)

    # Preprocess for model
    image = Image.fromarray(cv2.resize(frame, (224, 224)))
    image = ImageOps.fit(image, (224, 224))
    input_data = np.expand_dims(np.asarray(image).astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    top_idx = np.argmax(output_data)
    label = labels[top_idx]
    confidence = output_data[top_idx]

    # Show output
    display = f"{label}: {confidence*100:.2f}%"
    cv2.putText(frame, display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Smile Detector", frame)
    print(f"Detected: {label}, Confidence: {confidence:.2f}")

    # Serial send with debounce
    current_time = time.time()
    if confidence > 0.85:
        if label == "smiling":
            if arduino and (last_sent is None or current_time - last_sent > send_interval):
                arduino.write(b'smile\n')
                print("Sent: smile")
                last_sent = current_time
        elif label == "not smiling":
            if arduino and (last_sent is None or current_time - last_sent > send_interval):
                arduino.write(b'nosmile\n')
                print("Sent: nosmile")
                last_sent = current_time

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
