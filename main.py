import streamlit as st
import os
import cv2
import imutils
import torch
import time
import timm
import einops
import tempfile
import tqdm
import numpy as np
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer


def parse_video(video_file):
    vs = cv2.VideoCapture(video_file)

    frames = []
    while True:
        (gotit, frame) = vs.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if not gotit:
            break

    return np.stack(frames)


# Function to run cotracker_demo
def cotracker_demo(
    input_video,
    grid_size: int = 10,
    grid_query_frame: int = 0,
    backward_tracking: bool = False,
    tracks_leave_trace: bool = False
):
    load_video = parse_video(input_video)
    grid_query_frame = min(len(load_video) - 1, grid_query_frame)
    load_video = torch.from_numpy(load_video).permute(0, 3, 1, 2)[None].float()

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker_w8")

    if torch.cuda.is_available():
        model = model.cuda()
        load_video = load_video.cuda()
    pred_tracks, pred_visibility = model(
        load_video,
        grid_size=grid_size,
        grid_query_frame=grid_query_frame,
        backward_tracking=backward_tracking
    )
    linewidth = 2
    if grid_size < 10:
        linewidth = 4
    elif grid_size < 20:
        linewidth = 3

    vis = Visualizer(
        save_dir=os.path.join(os.path.dirname(__file__), "results"),
        grayscale=False,
        pad_value=100,
        fps=10,
        linewidth=linewidth,
        show_first_frame=5,
        tracks_leave_trace=-1 if tracks_leave_trace else 0,
    )

    def current_milli_time():
        return round(time.time() * 1000)

    filename = str(current_milli_time())
    vis.visualize(
        load_video.cpu(),
        tracks=pred_tracks.cpu(),
        visibility=pred_visibility.cpu(),
        filename=filename,
        query_frame=grid_query_frame,
    )
    return os.path.join(
        os.path.dirname(__file__), "results", f"{filename}_pred_track.mp4"
    )

@st.cache_data
def cotracker_demo_cached(input_video, grid_size, grid_query_frame, backward_tracking, visualize_track_traces):
    if isinstance(input_video, str):
        # Use the provided video path
        video_path = input_video
    else:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(input_video.read())
            video_path = temp_video.name

    return cotracker_demo(video_path, grid_size, grid_query_frame, backward_tracking, visualize_track_traces)

# Sample video file paths
apple = os.path.join(os.path.dirname(__file__), "assets", "apple.mp4")
bear = os.path.join(os.path.dirname(__file__), "assets", "bear.mp4")
paragliding_launch = os.path.join(os.path.dirname(__file__), "assets", "paragliding-launch.mp4")
paragliding = os.path.join(os.path.dirname(__file__), "assets", "paragliding.mp4")

# Streamlit app
st.set_page_config(
        page_title="Video Point Tracking Demo",
        page_icon="⏯️",
        layout="wide",
    )
st.title("⏯️ Video Point Tracking Demo")
st.markdown(
    """
    Welcome to the Video Point Tracking Demo!

    Here, we showcase how to track points (pixels) in videos. These points are sampled on a grid and tracked together.

    **Getting Started:**
    - Upload your video in .mp4 format with landscape orientation or choose one of our example videos.
    - Short videos (2-7 seconds) are recommended for quicker processing.

    **Key Features:**
    - **Grid Size:** Adjust the Grid Size to control the number of grid points.
    - **Grid Query Frame:** Specify the starting frame for tracking; tracks will appear after this frame.
    - **Backward Tracking:** Enable Backward Tracking to track points in both directions.
    - **Visualize Track Traces:** Check this option to see traces of all tracked points.

    Enjoy exploring video point tracking capabilities!
    """
)

st.markdown("***")

# Create two columns, one for instructions and one for inputs
col1, col2 = st.columns(2)

# Initialize result_video with a default value
result_video = None

# Sample video selection
with col1:
    sample_videos = {
        "Apple": apple,
        "Bear": bear,
        "Paragliding Launch": paragliding_launch,
        "Paragliding": paragliding,
    }

    user_video = st.file_uploader("Upload a Video (MP4 format)", type=["mp4"])

    selected_sample_video = st.selectbox("Select a Sample Video", list(sample_videos.keys()))

    st.markdown("***")

    # Parameters
    grid_size = st.slider("Grid Size", 1, 30, 10)
    grid_query_frame = st.slider("Grid Query Frame", 0, 30, 0)
    backward_tracking = st.checkbox("Backward Tracking")
    visualize_track_traces = st.checkbox("Visualize Track Traces")


    # "Run" button
    if st.button("Run"):
        if user_video is not None:
            # Use the uploaded video
            result_video = cotracker_demo_cached(user_video, grid_size, grid_query_frame, backward_tracking,
                                                 visualize_track_traces)
        else:
            # Use the selected sample video
            sample_video_path = sample_videos[selected_sample_video]
            result_video = cotracker_demo_cached(sample_video_path, grid_size, grid_query_frame, backward_tracking,
                                                 visualize_track_traces)

# Display the chosen video
if user_video is not None:
    with col2:
        st.markdown("### Preview of Chosen Video")
        st.video(user_video, format="video/mp4")
    with col2:
        st.markdown("### Result")
        st.video(result_video, format="video/mp4")
else:
    with col2:
        sample_video_path = sample_videos[selected_sample_video]
        st.markdown("### Preview of Chosen Video")
        st.video(sample_video_path, format="video/mp4")
    with col2:
        st.markdown("### Result")
        st.video(result_video, format="video/mp4")


