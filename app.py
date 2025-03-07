import cv2
import streamlit as st
# import mediapipe as mp

# RTSP stream URL
# noinspection PyPackageRequirements
rtsp = "rtsp://admin:admin@192.168.2.9/ch0_0.264"

print(cv2.getBuildInformation())
cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))

st.title("Video Capture with OpenCV and MediaPipe Face Detection")

# Initialize MediaPipe Face Detection and Drawing utilities.
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils

# Create a FaceDetection object with a minimum detection confidence.
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

frame_placeholder = st.empty()
stop_button = st.button("Stop")
start_button = st.button("Start")

if not cap.isOpened():
    print("Failed to open stream")

# When the start button is pressed, begin processing the video stream.
if start_button:
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.write("The video has ended or could not be read.")
            break

        # Convert the frame to RGB (MediaPipe requires RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for face detection.
        # results = face_detection.process(frame_rgb)

        # If faces are detected, draw the bounding boxes.
        # if results.detections:
        #     for detection in results.detections:
        #         pass
                # mp_drawing.draw_detection(frame_rgb, detection)
        # Update the image display in Streamlit.
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Optional: Allow exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
# cv2.destroyAllWindows()