from ultralytics import YOLO
import time
import streamlit as st
import cv2
from collections import defaultdict
import os
import settings

frame_class_counts = {}


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


# def display_tracker_options():
#     display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
#     is_display_tracker = True if display_tracker == 'Yes' else False
#     if is_display_tracker:
#         tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
#         return is_display_tracker, tracker_type
#     return is_display_tracker, None

def process_videos(input_dir, output_dir):
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            file_path = os.path.join(input_dir, filename)
            cap = cv2.VideoCapture(file_path)

            if not cap.isOpened():
                # print(f"Error opening video file {filename}")
                continue

            # Get original video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create VideoWriter object for output video
            output_path = os.path.join(output_dir, filename)
            out = cv2.VideoWriter(output_path, fourcc, fps)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w, _ = frame.shape
                min_dim = min(h, w)
                if h > w:
                    # Crop the top
                    start_y = 0
                    # cropped_frame = frame[start_y:min_dim, 0:w] 
                else:
                    # Center crop
                    start_x = (w - min_dim) // 2
                    # cropped_frame = frame[0:h, start_x:start_x + min_dim]
                frame_resized = cv2.resize(frame)

                # Write the frame to the output video
                out.write(frame_resized)

            cap.release()
            out.release()
            # print(f"Processed and saved {filename}")
            return out



def _detect_objects(conf, model, image):
    """
    Detect objects in the frame using YOLOv8 model.
    
    Args:
    - conf (float): Confidence threshold for object detection.
    - model: YOLOv8 object detection model.
    - image: The input image (frame) to detect objects.

    Returns:
    - res: Detection results with bounding boxes and class IDs.
    - class_count: A dictionary of object class counts in the current frame.
    """
    res = model.predict(image, conf=conf)
    names = model.names
    class_count = defaultdict(int)
    
    for r in res:
        for c in r.boxes.cls:
            class_name = names[int(c)]  
            class_count[class_name] += 1  # Count detected objects by class
    
    return res, class_count

def _display_detected_frames(conf, model, frame):
    """
    Display detected objects in a single video frame.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model: YOLOv8 object detection model.
    - frame: The current frame to process and display.

    Returns:
    - res_plotted: Image with detected objects plotted.
    """
    res, _ = _detect_objects(conf, model, frame)
    res_plotted = res[0].plot()  # Draw bounding boxes and labels on frame
    return res_plotted

def process_frames(frames, confidence, model):
    """
    Process a list of video frames to detect objects and return processed frames.

    Args:
    - frames (list): List of video frames (images) to process.
    - confidence (float): Confidence threshold for object detection.
    - model: YOLOv8 object detection model.

    Returns:
    - processed_frames (list): List of frames with detected objects plotted.
    - frame_class_counts (list of dict): List of object counts for each frame.
    """
    processed_frames = []
    frame_class_counts = []

    for frame in frames:
        res, class_count = _detect_objects(confidence, model, frame) 
        res_plotted = res[0].plot()
        processed_frames.append(res_plotted)
        frame_class_counts.append(class_count)

    return processed_frames, frame_class_counts

def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    # is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image
                                            #  is_display_tracker,
                                            #  tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
