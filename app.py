import streamlit as st
import time
from PIL import Image


# with st.empty():
#     for seconds in range(10):
#         st.write(f'{seconds} seconds haves passed')
#         time.sleep(1)
#     st.write('COMPLETE!')

def detect_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    image_pil = raw_image.resize((720,400))
    return image_pil

video_file = open('test-files/video2.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

with st.empty():
    for seconds in range(1, 6):
        st.image(detect_image(f'test-files/out{seconds}.jpg'))
        time.sleep(0.8)
