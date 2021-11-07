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

def main():
    st.set_page_config(page_title="Object Detection", page_icon="🚥", layout='centered', initial_sidebar_state="expanded")

    st.title('Object Detection')
    st.caption('An interactive project built by the Yolomites (Inspirit AI)')
    navigation = st.selectbox('Navigation', ('Home','App Demo'))
    if navigation == 'Home':    
        st.sidebar.write('Navigate to the **"App Demo"** section to view the project in action!')

        st.write('## Summary')
        st.write('This project involves the usage of a YoloV3 Neural Network that is capable of locating and classifying objects in visual data. The YoloV3 model \
            is able to work in real time and output predictions based on specific use cases. The model contains a total of `252` layers and `62,001,757` parameters')
        st.image('https://miro.medium.com/max/2000/1*d4Eg17IVJ0L41e7CTWLLSg.png')
        st.write('## Functionality')
        st.write('Given an image or a video, the model will identify where objects are located through the use of bounding boxes and further classify each object \
            based into a specific category. Each classification also includes a numeric probability representing the "likelyhood" of an object being a specific class.')
        st.image('https://cdn.analyticsvidhya.com/wp-content/uploads/2018/12/Screenshot-from-2018-11-29-13-03-17.png')
        st.write('## Resources')
        st.markdown('- [Github Repository](https://github.com/Real-VeerSandhu/Object-Detection)')
        st.markdown('- [App Source Code](https://github.com/Real-VeerSandhu/Object-Detection/blob/master/app.py)')
    else:
        vid_file = st.sidebar.file_uploader("Upload A Video (mp4)", help='Select a video on your local device for the Object Detection model to process and output', type=['mp4'])
        st.sidebar.markdown('----')
        stutter_speed = st.sidebar.slider('Video Stutter Speed (Frametime)', 1, 10)
        if vid_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(vid_file.read())
            vidcap = cv2.VideoCapture(tfile.name)
            st.write(vidcap)
            st.write(f'`Total Frames: {int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))}`')
            frames = detect_video(vidcap, stutter_speed)

            if st.button('Run'):
                st.markdown('----')
                with st.empty():
                    for i in range(len(frames)):
                        st.image(frames[i])
                        time.sleep(0.01 * stutter_speed ** 1.2)

if __name__ == '__main__':
    main()