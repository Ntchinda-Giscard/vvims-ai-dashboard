import cv2
import streamlit as st
# import numpy as np
import tempfile


rtsp = f"rtsp://admin:admin@192.168.2.71/ch0_0.264"
cap = cv2.VideoCapture(rtsp)

st.title("Video Capture with OpenCV")

frame_placeholder = st.empty()
stop_button = st.button("stop")

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()

    if not ret:
        st.write("The video has ended.")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels='RGB')
    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button:
        break

cap.release()
cv2.destroyAllWindows()
