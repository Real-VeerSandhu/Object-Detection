from numpy import cbrt
import streamlit as st
import time
from PIL import Image
import cv2
import tempfile

def detect_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    image_pil = raw_image.resize((720,400))
    return image_pil

def detect_video(vidcap, counter):
    success,image = vidcap.read()
    count = 0
    
    
    imgs = []
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) 

    for i in range(1, frame_count + 1):
        success,image = vidcap.read()
        imgs.append(image)
        count += 1

    preds = []
    for frame in range(1, frame_count-1, counter):
        pil_converted = Image.fromarray(cv2.cvtColor(imgs[frame], cv2.COLOR_BGR2RGB))
        image_pil = pil_converted.resize((720,400))

        # plt.figure(figsize=(15,7))
        # output = scan_image(image_pil, obj_thresh=0.55)
        # plt.title(f'Frame {frame}, {output}')
        # plt.imshow(output)
        # preds.append(output)
        preds.append(image_pil)

    return preds

file = st.file_uploader("Upload A Video")

if file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    vidcap = cv2.VideoCapture(tfile.name)
    st.write(vidcap)
    st.write(f'`Total Frames: {int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))}`')
    frames = detect_video(vidcap, 1)

    if st.button('Run'):
        st.markdown('----')
        with st.empty():
            for i in range(len(frames)):
                st.image(frames[i])
                time.sleep(0.01)