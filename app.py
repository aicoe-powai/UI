# Python In-built packages
from pathlib import Path
import PIL
from PIL import Image
import PIL.Image
import cv2
import shutil
# External packages
import streamlit as st
import tempfile
import numpy as np
import pandas as pd
import os
import time
# Local Modules
import settings
import helper
from streamlit_extras.app_logo import add_logo

#global param
global_check = False
frame_global_output = {}
bom_name_id = { 
"ACOS" : ["ACOS","SPE0101172133"],
"EMIS" : ["Emis Filter","IPMS1064C2A01"],
"ETHSW" : ["Ethernet Switch","IPMS105146201"],
"EXNWC" : ["Extended RJ45 Network Connector","SPE0101186257"],
"FAN" : ["Fan","SPE0101186501"],
"DTTB" : ["Mini Terminal Block","SPE0101173665"],
"THMT" : ["Thermostat","IPMS1108KQW01"],
"DSBC" : ["DSUB Backshell Connector","IPMS105AWRU01"],
"MCB" : ["MCB","-"],
"FRM" : ["Fuse Relay Module","IPMS1019O3S01"],
"IORT" : ["Input Output Relay Terminal","IPMS1091WAZ01"],
"PLI" : ["Power Light Indicator","-"],
"ISOC" : ["Isolated Converter","DD900067-6"],
"RSR" : ["RS232 Repeater","-"]

} #com_class_name , com_name, bom_item_code

# Setting page layout
st.set_page_config(
    page_title="L&T PES BOM Verification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
# def logo():
#     add_logo(settings.LOGO_PATH, height=300)


# Main page heading
st.title("L&T PES BOM Verification")

with st.sidebar:
    st.image("/home/aicoe-lnx/Desktop/xyz/UI/assets/logo.png", width=250)

confidence = 0.25
model_path = settings.DETECTION_MODEL
# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Setup")
source_radio = st.sidebar.radio(
    "**Select Source**", settings.SOURCES_LIST)

source_component = st.sidebar.radio(
    "**Select Component**", settings.SOURCES_COMPONENT)

source_sections = st.sidebar.radio(
    "**Select Sections**", settings.SOURCES_SECTIONS
)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                # default_image = PIL.Image.open(default_image_path)
                default_image = cv2.imread(default_image_path)
                st.image(default_image_path, caption="Input Image",
                          width = 480)
            else:
                uploaded_image = PIL.Image.open(source_img)
                uploaded_image = uploaded_image.crop((0,0,2448,2448))
                uploaded_image = uploaded_image.resize((480,480))

                st.image(source_img, caption="Uploaded Image",
                          width = 480 )
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     width=480)
            #use_column_width=True
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         width=480)
                #use_column_width=True
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")
    #Table logic
    with col2:
        # Static table
        data = {
            "S.No" : [1,2,3],
            "Component Name": ["Fan","Acos","Ethernet Switch"],
            "Component Class" : ["FAN" , "ACOS" , "ETHSW"],
            "BOM Class" : ["SPE0101186501","SPE0101172133","IPMS105146201"],
            "Quantity" : [1,2,3]
        }
        df = pd.DataFrame(data)
        st.table(df)

elif source_radio == settings.VIDEO:
    source_video = st.sidebar.file_uploader(
        "Choose a video...", type=("mp4", "avi", "mov", "wmv"))
    
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0
            
    if 'frames' not in st.session_state:
        st.session_state.frames = []

    def extract_frames_from_video(video_path):
        frames = []
        vid_cap = cv2.VideoCapture(video_path)
        
        while True:
            success, frame = vid_cap.read()
            if not success:
                break
            # Convert frame to RGB and store as PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        
        vid_cap.release()
        return frames
    
    with st.container():
        col1, col2 = st.columns(2)
        # print(col1, col2)

        with col1:
            try:
                if source_video is None:
                    default_video_path = str(settings.DEFAULT_VIDEO)
                    st.video(default_video_path, format="video/mp4")
                    # Add a caption below the video with custom styling
                    caption = """
                    <div style="text-align: center; color: black; font-size: 25px;">
                        Input Video
                    </div>
                    """

                    st.markdown(caption, unsafe_allow_html=True)
                else:
                    st.video(source_video, format = "video/mp4") #Uploaded video

            except Exception as ex:
                st.error("Error occurred while opening the video.")
                st.error(ex)

            caption = """
                    
                    """
            # Giving some empty space between the Input Video and Detected video
            st.markdown(caption, unsafe_allow_html=True)

        # Table logic
        with col2:
        
            if source_video is None:
                default_detected_video_path = str(settings.DEFAULT_DETECT_VIDEO)
                st.video(default_detected_video_path, format="video/mp4")
                caption = """
                    <div style="text-align: center; color: black; font-size: 25px;">
                        Detected Video
                    </div>
                    """

                st.markdown(caption, unsafe_allow_html=True)
            else:  
                if st.sidebar.button('Detect Objects'):
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file.write(source_video.read())
                    frames = extract_frames_from_video(temp_file.name)
                    
                    processed_frames, frame_global_output = helper.process_frames(frames, confidence, model)
                    output_video_path = os.path.join(os.getcwd(), 'processed_video.mp4')
                    height, width, _ = processed_frames[0].shape
                    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

                    for frame in processed_frames:
                        out.write(frame)

                    out.release()

                    absolute_video_path = os.path.abspath(output_video_path)
                    
                    

                # Once detection is complete, build the component count table
                if frame_global_output:
                    component_totals = {}

                    for counts in frame_global_output:  # No need to unpack frame_index
                        for class_name, count in counts.items():
                            if class_name in component_totals:
                                component_totals[class_name] += count
                            else:
                                component_totals[class_name] = count
                    # Prepare data for table display
                    table_data = []
                    s_no = 1  
                    for class_name, total_quantity in component_totals.items():
                        if class_name in bom_name_id:
                            component_name, bom_item_code = bom_name_id[class_name]
                        else:
                            component_name = None
                            bom_item_code = None

                        table_data.append({
                            "S.No": s_no,
                            "Component Class": class_name,
                            "Component Name": component_name,
                            "BOM Item Code": bom_item_code,
                            "Quantity": total_quantity
                        })
                        s_no += 1

                    # Convert to DataFrame and display in Streamlit table
                    df = pd.DataFrame(table_data)
                    st.table(df)
                    print(os.path.exists(absolute_video_path))
                    print(os.stat(absolute_video_path))
                    st.video('processed_video.mp4', format="video/mp4")

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)
