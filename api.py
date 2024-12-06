from flask import Flask, jsonify, send_file
import cv2
import tensorflow as tf
import numpy as np
import sys
import os
import io
import requests
import base64

app = Flask(__name__)

@app.route('/detect_suspicious_activity')
def run_suspicious_activity_detection():
    # Load the model
    model = tf.keras.models.load_model('model.h5')

    # Determine the input shape from the model
    input_shape = model.input_shape[1:]  # Get all dimensions except batch size
    print(f"Model input shape: {input_shape}")

    # Initialize webcam capture
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    def preprocess_frame(frame):
        frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))  # Resize to (width, height)
        frame_array = frame_resized / 255.0  # Normalize
        return frame_array

    # Define class labels
    class_labels = {
        0: "Normal",
        1: "Suspicious"
    }

    # Initialize a buffer to hold multiple frames
    frame_buffer = []
    frames_needed = input_shape[0]  # Number of frames the model expects

    suspicious_count = 0
    alert_duration = 100  # Number of frames to display the alert message
    alert_counter = 0
    photo_count = 0
    max_photos = 5
    encoded_images = []

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Video recording setup
    recording = False
    out = None
    video_path = None

    def encode_image(frame):
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{jpg_as_text}"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        
        # Add the processed frame to the buffer
        frame_buffer.append(processed_frame)
        
        # If we have enough frames, make a prediction
        if len(frame_buffer) == frames_needed:
            # Stack frames into a single 5D array
            input_data = np.stack(frame_buffer, axis=0)
            input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
            
            # Make prediction
            predictions = model.predict(input_data)
            
            # Process predictions
            predicted_class = np.argmax(predictions, axis=1)[0]
            activity_label = class_labels.get(predicted_class, "Unknown")
            
            # Determine color based on activity (green for normal, red for suspicious)
            color = (0, 255, 0) if activity_label == "Normal" else (0, 0, 255)
            
            # Display the frame with predictions
            label = f'Activity: {activity_label}'
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Check if suspicious activity is detected
            if activity_label == "Suspicious":
                suspicious_count += 1
                if suspicious_count >= 2:
                    alert_counter = alert_duration

                    # Start video recording if not already recording
                    if not recording:
                        video_path = os.path.join(os.getcwd(), 'suspicious_activity.avi')
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                        recording = True
                        print(f"Recording video to {video_path}")

                    # Capture and encode suspicious activity images
                    if photo_count < max_photos:
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

                        if len(faces) > 0:
                            for (x, y, w, h) in faces:
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            
                            encoded_image = encode_image(frame)
                            encoded_images.append(encoded_image)
                            photo_count += 1
                            print(f"Suspicious activity image {photo_count} captured")

                            # Send encoded images to API when 5 are captured
                            if photo_count == max_photos:
                                try:
                                    api_url = "https://cheerful-apparent-hound.ngrok-free.app/upload/"
                                    payload = {"images": encoded_images}
                                    response = requests.post(api_url, json=payload)
                                    if response.status_code == 200:
                                        print("Suspicious activity images sent to API successfully")
                                    else:
                                        print(f"Failed to send images to API. Status code: {response.status_code}")
                                except Exception as e:
                                    print(f"Error sending images to API: {str(e)}")
                                
                                # Reset the counter and list after sending
                                photo_count = 0
                                encoded_images = []

        # Always detect and draw rectangles around all faces in the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display alert message on screen if suspicious activity was recently detected
        if alert_counter > 0:
            alert_counter -= 1

        # If recording, write the frame to the video file
        if recording:
            out.write(frame)
        
        # Always display the frame
        cv2.imshow('Webcam Feed', frame)

        # Remove the oldest frame from the buffer if it's full
        if len(frame_buffer) >= frames_needed:
            frame_buffer.pop(0)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if recording:
        out.release()
        # We're not sending the video to the API anymore
        os.remove(video_path)  # Remove the local video file

    cv2.destroyAllWindows()

    # Prepare response data
    response_data = {
        "message": "Session ended. Suspicious activity was detected." if alert_counter > 0 else "Session ended normally. No suspicious activity detected.",
        "encoded_images": encoded_images
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)