import os
import platform
import subprocess
import sys
import cv2
import torch
import time
from time import sleep
import numpy as np
import einops
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QCheckBox, \
    QFileDialog, QProgressBar, QComboBox, QSizePolicy
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPalette, QFont
from cotracker.utils.visualizer import Visualizer, read_video_from_path

# Specify the default device based on GPU availability
DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')


class CoTrackerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize UI components
        self.init_ui()

        # Initialize CoTracker parameters
        self.grid_size = 10
        self.grid_query_frame = 0
        self.backward_tracking = False
        self.tracks_leave_trace = False
        self.result_video_path = None

        # Initialize video variables
        self.input_video = None
        self.load_video = None
        self.video_frame_count = 0
        self.current_frame = 0
        self.is_playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_frame)

        # Initialize CoTracker model
        self.init_cotracker_model()

        # Set up the video display area
        self.init_video_display()

    def init_ui(self):
        # Set up the main window
        self.setWindowTitle("CoTracker")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create layout for UI elements
        layout = QVBoxLayout()

        # Create and add the Load Video button
        load_button = QPushButton("Load Video", self)
        load_button.clicked.connect(self.load_video_file)
        layout.addWidget(load_button)

        # Create and add the Grid Size slider
        self.grid_size_label = QLabel("Grid Size: 10")
        self.grid_size_slider = QSlider(Qt.Horizontal)
        self.grid_size_slider.setRange(1, 30)
        self.grid_size_slider.setValue(10)
        self.grid_size_slider.valueChanged.connect(self.update_grid_size)
        layout.addWidget(self.grid_size_label)
        layout.addWidget(self.grid_size_slider)

        # Create and add the Grid Query Frame slider
        self.grid_query_frame_label = QLabel("Grid Query Frame: 0")
        self.grid_query_frame_slider = QSlider(Qt.Horizontal)
        self.grid_query_frame_slider.setRange(0, 30)
        self.grid_query_frame_slider.setValue(0)
        self.grid_query_frame_slider.valueChanged.connect(self.update_grid_query_frame)
        layout.addWidget(self.grid_query_frame_label)
        layout.addWidget(self.grid_query_frame_slider)

        # Create and add the Backward Tracking checkbox
        self.backward_tracking_checkbox = QCheckBox("Backward Tracking", self)
        self.backward_tracking_checkbox.stateChanged.connect(self.update_backward_tracking)
        layout.addWidget(self.backward_tracking_checkbox)

        # Create and add the Visualize Track Traces checkbox
        self.visualize_track_traces_checkbox = QCheckBox("Visualize Track Traces", self)
        self.visualize_track_traces_checkbox.stateChanged.connect(self.update_visualize_track_traces)
        layout.addWidget(self.visualize_track_traces_checkbox)

        # Create and add the Play/Pause button
        self.play_pause_button = QPushButton("Play", self)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        layout.addWidget(self.play_pause_button)

        # Create and add the Video Display Label
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Create and add the Progress Bar
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        # Set the layout for the central widget
        central_widget.setLayout(layout)

    def init_video_display(self):
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Background, Qt.black)
        self.video_label.setPalette(palette)

    def init_cotracker_model(self):
        # Load CoTracker model
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker_w8")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def load_video_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)",
                                                   options=options)
        if file_path:
            self.input_video = file_path
            self.load_video = read_video_from_path(self.input_video)
            self.video_frame_count = len(self.load_video)
            self.current_frame = 0
            self.update_grid_query_frame(0)  # Reset grid query frame
            self.play_pause_button.setEnabled(True)
            self.timer.start(1000 // 30)  # 30 FPS playback
            self.play_pause_button.setText("Pause")
            self.progress_bar.setValue(0)
            self.result_video_path = None
            self.tracks_leave_trace = False
            self.visualize_track_traces_checkbox.setChecked(False)
            self.update_video_frame()

    def update_grid_size(self, value):
        self.grid_size = value
        self.grid_size_label.setText(f"Grid Size: {value}")

    def update_grid_query_frame(self, value):
        self.grid_query_frame = value
        self.grid_query_frame_label.setText(f"Grid Query Frame: {value}")
        if self.current_frame < self.grid_query_frame:
            self.current_frame = self.grid_query_frame
            self.update_video_frame()

    def update_backward_tracking(self, state):
        self.backward_tracking = state == Qt.Checked

    def update_visualize_track_traces(self, state):
        self.tracks_leave_trace = state == Qt.Checked

    def toggle_play_pause(self):
        if self.is_playing:
            self.timer.stop()
            self.play_pause_button.setText("Play")
        else:
            self.timer.start(1000 // 10)  # 30 FPS playback
            self.play_pause_button.setText("Pause")
        self.is_playing = not self.is_playing

    def update_video_frame(self):
        if self.load_video is not None:
            target_fps = 10  # Set the target FPS to 10
            frame_skip = int(30 / target_fps)
            if self.current_frame < len(self.load_video):
                frame = self.load_video[self.current_frame]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image)
                self.video_label.setPixmap(pixmap)
                self.progress_bar.setValue((self.current_frame / self.video_frame_count) * 100)
                if self.current_frame == self.grid_query_frame:
                    self.result_video_path = self.run_cotracker()
                self.current_frame += frame_skip  # Skip frames for faster playback
            else:
                self.timer.stop()
                self.play_pause_button.setText("Play")
                self.is_playing = False
                self.current_frame = 0

    def run_cotracker(self):
        grid_query_frame = min(self.grid_query_frame, self.video_frame_count - 1)
        load_video = torch.from_numpy(self.load_video).permute(0, 3, 1, 2)[None].float()
        if torch.cuda.is_available():
            load_video = load_video.cuda()
        pred_tracks, pred_visibility = self.model(
            load_video,
            grid_size=self.grid_size,
            grid_query_frame=grid_query_frame,
            backward_tracking=self.backward_tracking
        )
        linewidth = 2
        if self.grid_size < 10:
            linewidth = 4
        elif self.grid_size < 20:
            linewidth = 3

        # Get the directory where the script is located
        script_directory = os.path.dirname(os.path.abspath(__file__))

        vis = Visualizer(
            save_dir=script_directory,
            grayscale=False,
            pad_value=100,
            fps=10,
            linewidth=linewidth,
            show_first_frame=5,
            tracks_leave_trace=-1 if self.tracks_leave_trace else 0,
        )

        # Print the save_dir
        print(f"Save Directory: {vis.save_dir}")  # Add this line to print the save_dir

        def current_milli_time():
            return round(time.time() * 1000)

        filename = str(current_milli_time())

        # Create a signal object for updating the progress bar
        progress_signal = ProgressSignal()

        # Connect the signal to a slot that updates the progress bar
        progress_signal.progress_update.connect(self.update_progress_bar)

        # Create a signal object for updating the progress bar
        progress_signal = ProgressSignal()

        # Connect the signal to a slot that updates the progress bar
        progress_signal.progress_update.connect(self.update_progress_bar)

        # Start a separate thread to run the CoTracker and visualization process
        def cotracker_thread():
            nonlocal filename
            vis.visualize(
                load_video,
                tracks=pred_tracks,
                visibility=pred_visibility,
                filename=filename,
                query_frame=grid_query_frame,
            )
            result_video_path = os.path.join(os.path.dirname(__file__), f"{filename}_pred_track.mp4")
            self.result_video_path = result_video_path
            # Emit the signal to update the progress bar
            progress_signal.progress_update.emit(100)

            # Open the result video using the system's default video player
            if platform.system() == 'Windows':
                subprocess.Popen(['start', result_video_path], shell=True)
            elif platform.system() == 'Linux':
                subprocess.Popen(['xdg-open', result_video_path])
            # Add additional checks for other operating systems if needed

        # Create and start the thread
        cotracker_thread = threading.Thread(target=cotracker_thread)
        cotracker_thread.start()

        return self.result_video_path


    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

# Define a QObject with a signal for progress updates
class ProgressSignal(QObject):
    progress_update = pyqtSignal(int)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = CoTrackerApp()
    main_win.show()
    sys.exit(app.exec_())