from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
RTU = 'Remote Terminal Unit'
PDU_SP = 'Power Distribution Unit-SP'
SAU = 'Servo Amplifier Unit'
SECTION_1 = 'Section 1'
SECTION_2 = 'Section 2'
SECTION_3 = 'Section 3'
SECTION_4 = 'Section 4'
SECTION_5 = 'Section 5'
# WEBCAM = 'Webcam'
# RTSP = 'RTSP'
# YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO]
SOURCES_COMPONENT = [RTU, PDU_SP,SAU]
SOURCES_SECTIONS = [SECTION_1,SECTION_2,SECTION_3,SECTION_4,SECTION_5]

ASSET_DIR = ROOT / 'assets'
LOGO_PATH = ASSET_DIR / 'logo.png'
# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'video_1': VIDEO_DIR / 'video_1.mp4',
    'video_2': VIDEO_DIR / 'video_2.mp4',
    'video_3': VIDEO_DIR / 'video_3.mp4',
}
DEFAULT_VIDEO = VIDEO_DIR / 'video_1.mp4'
DEFAULT_DETECT_VIDEO = VIDEO_DIR / 'video_1.mp4'
# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
