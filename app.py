import cv2
import streamlit as st
import numpy as np
import tempfile

cap = cv2.VideoCapture(0)

st.title("Video Capture with OpenCV")

frame_placeholder = st.empty()
stop_button = st.button("stop")