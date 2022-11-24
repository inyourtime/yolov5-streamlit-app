import streamlit as st
import tempfile

from PIL import Image
import numpy as np
import yolo
import cv2
import time
import control

ESP32_IP = '172.20.10.4'
ESP32_STREAM_PORT = '80'
ESP32_CMD_PORT = '8080'

def main():
    # Project title
    st.title('Object Detection Application')

    # Sidebar title
    st.sidebar.title('Settings')

    st.sidebar.markdown('---')

    # Select YOLOv5 model for Object Detection
    model_type = st.sidebar.selectbox('Select your yolov5 models', ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'])

    # Load model
    model = yolo.load_model(model_type)

    # Confidence bar
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)

    # Todo: IoU threshold
    iou = st.sidebar.slider('IoU', min_value=0.0, max_value=1.0, value=0.45)

    # Input type
    input_type = st.sidebar.selectbox('Select your input type', ['image', 'video', 'webcam', 'esp32'])

    # TODO: use CPU or GPU
    save_result = st.sidebar.checkbox('Save your result')
    enable_GPU = st.sidebar.checkbox('Enable GPU')

    # Single image inferrence
    if input_type == 'image':
        uploaded_img = st.sidebar.file_uploader('Upload your image', type=['jpg', 'jpeg', 'png'])
        if uploaded_img is not None:
            original_img = Image.open(uploaded_img)
            original_img = np.array(original_img.convert('RGB'))
            st.sidebar.image(original_img)

            st.header('Result')
            results = yolo.score_frame(model, original_img)
            _, _, dis_result = results
            detected_img = yolo.plot_boxes(results, original_img, confidence, model)
            st.image(detected_img)
            st.dataframe(dis_result.pandas().xyxy[0])

    # TODO: cannot use for inference now
    if input_type == 'video':
        video_file_buffer = st.sidebar.file_uploader('Upload your video', type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
        tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        if video_file_buffer is not None:
            tffile.write(video_file_buffer.read())
            demo_vid = open(tffile.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_bytes)
            # print(demo_vid.name)

    # Webcam inferrence        
    if input_type == 'webcam':
        frame_window = st.empty()
        stop_btn = st.button('Stop')
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = yolo.score_frame(model, frame)
            _, _, dis_result = results
            detected_img = yolo.plot_boxes(results, frame, confidence, model)
            end_time = time.time()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(detected_img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            frame_window.image(detected_img)

            if stop_btn:
                break
    
    # ESP32-CAM inferrence
    if input_type == 'esp32':
        # x = ['UXGA (1600x1200)', 'SXGA (1280x1024)']
        st.sidebar.header('Camera config')
        resolution = st.sidebar.selectbox('Camera resolution', ['VGA (640x480)',
                                                        'SVGA (800x600)', 'XGA (1024x768)', 'CIF (400x296)', 'QVGA (320x240)'])
        
        quality = st.sidebar.slider('quality', min_value=6, max_value=63, value=12)
        brightness = st.sidebar.slider('brightness', min_value=-2, max_value=2, value=0)
        contrast = st.sidebar.slider('contrast', min_value=-2, max_value=2, value=0)
        saturation = st.sidebar.slider('saturation', min_value=-2, max_value=2, value=0)
        _, _, col3 = st.sidebar.columns(3)
        with col3:
            cam_config = st.button('Save')
        if cam_config:
            # if resolution == 'UXGA (1600x1200)':
            #     val = 13
            #     query = f'http://192.168.111.26:8080/control?var=framesize&val={val}'
            #     requests.get(query)
            # if resolution == 'SXGA (1280x1024)':
            #     val = 12
            #     query = f'http://192.168.111.26:8080/control?var=framesize&val={val}'
            #     requests.get(query)
            if resolution == 'XGA (1024x768)':
                control.command(ESP32_IP, ESP32_CMD_PORT, 'framesize', 10)
            if resolution == 'SVGA (800x600)':
                control.command(ESP32_IP, ESP32_CMD_PORT, 'framesize', 9)
            if resolution == 'VGA (640x480)':
                control.command(ESP32_IP, ESP32_CMD_PORT, 'framesize', 8)
            if resolution == 'CIF (400x296)':
                control.command(ESP32_IP, ESP32_CMD_PORT, 'framesize', 6)
            if resolution == 'QVGA (320x240)':
                control.command(ESP32_IP, ESP32_CMD_PORT, 'framesize', 5)

            control.command(ESP32_IP, ESP32_CMD_PORT, 'quality', quality)
            control.command(ESP32_IP, ESP32_CMD_PORT, 'brightness', brightness)
            control.command(ESP32_IP, ESP32_CMD_PORT, 'contrast', contrast)
            control.command(ESP32_IP, ESP32_CMD_PORT, 'saturation', saturation)
       
        frame_window = st.empty()
        stop_btn = st.button('Stop')
        esp32_host = f'http://{ESP32_IP}:{ESP32_STREAM_PORT}/cam'
        cap = cv2.VideoCapture(esp32_host)
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = yolo.score_frame(model, frame)
            _, _, dis_result = results
            detected_img = yolo.plot_boxes(results, frame, confidence, model)
            end_time = time.time()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(detected_img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            frame_window.image(detected_img)

            if stop_btn:
                break


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass