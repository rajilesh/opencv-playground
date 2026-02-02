import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import json
from datetime import datetime
import base64
from pathlib import Path
import zipfile
import tempfile
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="OpenCV Playground",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Method card */
    .method-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* History item */
    .history-item {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    
    .history-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
    
    /* Image container */
    .image-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Status badges */
    .badge-success {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    
    .badge-info {
        background: #667eea;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e0e3ff, transparent);
        margin: 0.75rem 0;
    }
    
    /* Modern Pipeline Card */
    .pipeline-card {
        background: #fff;
        border: 1px solid #e8eaed;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        transition: all 0.2s ease;
        cursor: grab;
        position: relative;
    }
    
    .pipeline-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-color: #667eea;
    }
    
    .pipeline-card.dragging {
        opacity: 0.5;
        cursor: grabbing;
    }
    
    .pipeline-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .pipeline-number {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 600;
        flex-shrink: 0;
    }
    
    .pipeline-method-name {
        font-weight: 500;
        font-size: 14px;
        color: #333;
        flex-grow: 1;
    }
    
    .pipeline-actions {
        display: flex;
        gap: 4px;
    }
    
    .pipeline-action-btn {
        background: #f5f5f5;
        border: none;
        border-radius: 6px;
        width: 28px;
        height: 28px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.15s ease;
        font-size: 12px;
    }
    
    .pipeline-action-btn:hover {
        background: #667eea;
        color: white;
    }
    
    .pipeline-action-btn.delete:hover {
        background: #dc3545;
    }
    
    /* Drag handle */
    .drag-handle {
        color: #999;
        cursor: grab;
        padding: 4px;
        margin-right: 4px;
    }
    
    .drag-handle:hover {
        color: #667eea;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Code Panel Styling */
    .code-panel {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .code-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #3d3d5c;
    }
    
    .code-panel-title {
        color: #fff;
        font-weight: 600;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .code-step {
        background: #2d2d44;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #667eea;
    }
    
    .code-step-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.5rem;
    }
    
    .code-step-number {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        font-weight: 600;
    }
    
    .code-step-method {
        color: #e0e0e0;
        font-size: 12px;
        font-weight: 500;
    }
    
    .code-block {
        background: #1a1a2e;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 11px;
        color: #a8d4a8;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-all;
    }
    
    .code-param {
        color: #f8b886;
    }
    
    .code-value {
        color: #87ceeb;
    }
    
    .code-func {
        color: #dcdcaa;
    }
    
    .full-code-block {
        background: #0d1117;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 11px;
        color: #c9d1d9;
        overflow-x: auto;
        white-space: pre;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* History thumbnail styling */
    .history-thumbnail {
        position: relative;
        cursor: pointer;
        border-radius: 8px;
        overflow: hidden;
        transition: transform 0.2s;
    }
    
    .history-thumbnail:hover {
        transform: scale(1.05);
    }
    
    .history-thumbnail img {
        width: 100%;
        height: auto;
        object-fit: contain;
        border-radius: 8px;
    }
    
    /* Tooltip for full image preview */
    .thumbnail-tooltip {
        position: relative;
        display: inline-block;
    }
    
    .thumbnail-tooltip .tooltip-image {
        visibility: hidden;
        position: fixed;
        z-index: 1000;
        background: white;
        border-radius: 10px;
        padding: 5px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        max-width: 400px;
        max-height: 400px;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }
    
    .thumbnail-tooltip:hover .tooltip-image {
        visibility: visible;
    }
    
    .tooltip-image img {
        max-width: 100%;
        max-height: 380px;
        object-fit: contain;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'use_as_input' not in st.session_state:
    st.session_state.use_as_input = False
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'video_file' not in st.session_state:
    st.session_state.video_file = None
if 'is_video' not in st.session_state:
    st.session_state.is_video = False
if 'video_frame' not in st.session_state:
    st.session_state.video_frame = 0
if 'compare_mode' not in st.session_state:
    st.session_state.compare_mode = False
if 'effect_pipeline' not in st.session_state:
    st.session_state.effect_pipeline = []  # List of {id, category, method, params}
if 'next_effect_id' not in st.session_state:
    st.session_state.next_effect_id = 1
if 'editing_effect_id' not in st.session_state:
    st.session_state.editing_effect_id = None  # ID of effect being edited
if 'selected_category_index' not in st.session_state:
    st.session_state.selected_category_index = 0  # 0 = "-- Select --" (None)
if 'selected_method_index' not in st.session_state:
    st.session_state.selected_method_index = 0  # 0 = "-- Select --" (None)
if 'show_code_panel' not in st.session_state:
    st.session_state.show_code_panel = False

# Create output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# OpenCV Methods Configuration
OPENCV_METHODS = {
    "Color Transformations": {
        "icon": "🎨",
        "methods": {
            "cvtColor - Grayscale": {
                "description": "Convert image to grayscale",
                "function": "cv2.cvtColor",
                "params": {}
            },
            "cvtColor - HSV": {
                "description": "Convert image to HSV color space",
                "function": "cv2.cvtColor",
                "params": {}
            },
            "cvtColor - LAB": {
                "description": "Convert image to LAB color space",
                "function": "cv2.cvtColor",
                "params": {}
            },
            "Invert Colors": {
                "description": "Invert all colors in the image",
                "function": "cv2.bitwise_not",
                "params": {}
            }
        }
    },
    "Blurring & Smoothing": {
        "icon": "💫",
        "methods": {
            "GaussianBlur": {
                "description": "Apply Gaussian blur to smooth the image",
                "function": "cv2.GaussianBlur",
                "params": {
                    "kernel_size": {"type": "slider", "min": 1, "max": 31, "default": 5, "step": 2, "label": "Kernel Size"},
                    "sigma": {"type": "slider", "min": 0.0, "max": 10.0, "default": 0.0, "step": 0.1, "label": "Sigma"}
                }
            },
            "MedianBlur": {
                "description": "Apply median blur - great for noise reduction",
                "function": "cv2.medianBlur",
                "params": {
                    "kernel_size": {"type": "slider", "min": 1, "max": 31, "default": 5, "step": 2, "label": "Kernel Size"}
                }
            },
            "BilateralFilter": {
                "description": "Apply bilateral filter - preserves edges while smoothing",
                "function": "cv2.bilateralFilter",
                "params": {
                    "d": {"type": "slider", "min": 1, "max": 15, "default": 9, "step": 1, "label": "Diameter"},
                    "sigmaColor": {"type": "slider", "min": 10, "max": 200, "default": 75, "step": 5, "label": "Sigma Color"},
                    "sigmaSpace": {"type": "slider", "min": 10, "max": 200, "default": 75, "step": 5, "label": "Sigma Space"}
                }
            },
            "BoxFilter": {
                "description": "Apply box filter (averaging blur)",
                "function": "cv2.boxFilter",
                "params": {
                    "kernel_size": {"type": "slider", "min": 1, "max": 31, "default": 5, "step": 1, "label": "Kernel Size"}
                }
            }
        }
    },
    "Edge Detection": {
        "icon": "📐",
        "methods": {
            "Canny": {
                "description": "Canny edge detection algorithm",
                "function": "cv2.Canny",
                "params": {
                    "threshold1": {"type": "slider", "min": 0, "max": 255, "default": 100, "step": 1, "label": "Threshold 1"},
                    "threshold2": {"type": "slider", "min": 0, "max": 255, "default": 200, "step": 1, "label": "Threshold 2"}
                }
            },
            "Sobel X": {
                "description": "Sobel edge detection in X direction",
                "function": "cv2.Sobel",
                "params": {
                    "ksize": {"type": "slider", "min": 1, "max": 7, "default": 3, "step": 2, "label": "Kernel Size"}
                }
            },
            "Sobel Y": {
                "description": "Sobel edge detection in Y direction",
                "function": "cv2.Sobel",
                "params": {
                    "ksize": {"type": "slider", "min": 1, "max": 7, "default": 3, "step": 2, "label": "Kernel Size"}
                }
            },
            "Laplacian": {
                "description": "Laplacian edge detection",
                "function": "cv2.Laplacian",
                "params": {
                    "ksize": {"type": "slider", "min": 1, "max": 7, "default": 3, "step": 2, "label": "Kernel Size"}
                }
            }
        }
    },
    "Thresholding": {
        "icon": "⚫",
        "methods": {
            "Binary Threshold": {
                "description": "Apply binary thresholding",
                "function": "cv2.threshold",
                "params": {
                    "thresh": {"type": "slider", "min": 0, "max": 255, "default": 127, "step": 1, "label": "Threshold"},
                    "maxval": {"type": "slider", "min": 0, "max": 255, "default": 255, "step": 1, "label": "Max Value"}
                }
            },
            "Adaptive Threshold Mean": {
                "description": "Adaptive thresholding using mean",
                "function": "cv2.adaptiveThreshold",
                "params": {
                    "block_size": {"type": "slider", "min": 3, "max": 51, "default": 11, "step": 2, "label": "Block Size"},
                    "C": {"type": "slider", "min": -20, "max": 20, "default": 2, "step": 1, "label": "Constant C"}
                }
            },
            "Adaptive Threshold Gaussian": {
                "description": "Adaptive thresholding using Gaussian",
                "function": "cv2.adaptiveThreshold",
                "params": {
                    "block_size": {"type": "slider", "min": 3, "max": 51, "default": 11, "step": 2, "label": "Block Size"},
                    "C": {"type": "slider", "min": -20, "max": 20, "default": 2, "step": 1, "label": "Constant C"}
                }
            },
            "Otsu's Threshold": {
                "description": "Automatic thresholding using Otsu's method",
                "function": "cv2.threshold",
                "params": {}
            }
        }
    },
    "Morphological Operations": {
        "icon": "🔲",
        "methods": {
            "Erosion": {
                "description": "Erode the image - shrinks bright regions",
                "function": "cv2.erode",
                "params": {
                    "kernel_size": {"type": "slider", "min": 1, "max": 21, "default": 5, "step": 1, "label": "Kernel Size"},
                    "iterations": {"type": "slider", "min": 1, "max": 10, "default": 1, "step": 1, "label": "Iterations"},
                    "kernel_shape": {"type": "dropdown", "options": ["Rectangle", "Ellipse", "Cross"], "default": "Rectangle", "label": "Kernel Shape"}
                }
            },
            "Dilation": {
                "description": "Dilate the image - expands bright regions",
                "function": "cv2.dilate",
                "params": {
                    "kernel_size": {"type": "slider", "min": 1, "max": 21, "default": 5, "step": 1, "label": "Kernel Size"},
                    "iterations": {"type": "slider", "min": 1, "max": 10, "default": 1, "step": 1, "label": "Iterations"},
                    "kernel_shape": {"type": "dropdown", "options": ["Rectangle", "Ellipse", "Cross"], "default": "Rectangle", "label": "Kernel Shape"}
                }
            },
            "Opening": {
                "description": "Opening - erosion followed by dilation",
                "function": "cv2.morphologyEx",
                "params": {
                    "kernel_size": {"type": "slider", "min": 1, "max": 21, "default": 5, "step": 1, "label": "Kernel Size"},
                    "kernel_shape": {"type": "dropdown", "options": ["Rectangle", "Ellipse", "Cross"], "default": "Rectangle", "label": "Kernel Shape"}
                }
            },
            "Closing": {
                "description": "Closing - dilation followed by erosion",
                "function": "cv2.morphologyEx",
                "params": {
                    "kernel_size": {"type": "slider", "min": 1, "max": 21, "default": 5, "step": 1, "label": "Kernel Size"},
                    "kernel_shape": {"type": "dropdown", "options": ["Rectangle", "Ellipse", "Cross"], "default": "Rectangle", "label": "Kernel Shape"}
                }
            },
            "Gradient": {
                "description": "Morphological gradient - difference between dilation and erosion",
                "function": "cv2.morphologyEx",
                "params": {
                    "kernel_size": {"type": "slider", "min": 1, "max": 21, "default": 5, "step": 1, "label": "Kernel Size"},
                    "kernel_shape": {"type": "dropdown", "options": ["Rectangle", "Ellipse", "Cross"], "default": "Rectangle", "label": "Kernel Shape"}
                }
            },
            "Top Hat": {
                "description": "Top hat - difference between image and opening",
                "function": "cv2.morphologyEx",
                "params": {
                    "kernel_size": {"type": "slider", "min": 1, "max": 21, "default": 5, "step": 1, "label": "Kernel Size"},
                    "kernel_shape": {"type": "dropdown", "options": ["Rectangle", "Ellipse", "Cross"], "default": "Rectangle", "label": "Kernel Shape"}
                }
            },
            "Black Hat": {
                "description": "Black hat - difference between closing and image",
                "function": "cv2.morphologyEx",
                "params": {
                    "kernel_size": {"type": "slider", "min": 1, "max": 21, "default": 5, "step": 1, "label": "Kernel Size"},
                    "kernel_shape": {"type": "dropdown", "options": ["Rectangle", "Ellipse", "Cross"], "default": "Rectangle", "label": "Kernel Shape"}
                }
            }
        }
    },
    "Image Adjustments": {
        "icon": "🎚️",
        "methods": {
            "Brightness & Contrast": {
                "description": "Adjust brightness and contrast",
                "function": "cv2.convertScaleAbs",
                "params": {
                    "alpha": {"type": "slider", "min": 0.0, "max": 3.0, "default": 1.0, "step": 0.1, "label": "Contrast (Alpha)"},
                    "beta": {"type": "slider", "min": -100, "max": 100, "default": 0, "step": 1, "label": "Brightness (Beta)"}
                }
            },
            "Histogram Equalization": {
                "description": "Enhance contrast using histogram equalization",
                "function": "cv2.equalizeHist",
                "params": {}
            },
            "CLAHE": {
                "description": "Contrast Limited Adaptive Histogram Equalization",
                "function": "cv2.createCLAHE",
                "params": {
                    "clipLimit": {"type": "slider", "min": 1.0, "max": 10.0, "default": 2.0, "step": 0.5, "label": "Clip Limit"},
                    "tileGridSize": {"type": "slider", "min": 2, "max": 16, "default": 8, "step": 1, "label": "Tile Grid Size"}
                }
            },
            "Gamma Correction": {
                "description": "Apply gamma correction",
                "function": "gamma",
                "params": {
                    "gamma": {"type": "slider", "min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1, "label": "Gamma"}
                }
            }
        }
    },
    "Geometric Transformations": {
        "icon": "📐",
        "methods": {
            "Resize": {
                "description": "Resize the image",
                "function": "cv2.resize",
                "params": {
                    "scale": {"type": "slider", "min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1, "label": "Scale Factor"},
                    "interpolation": {"type": "dropdown", "options": ["Nearest", "Linear", "Area", "Cubic", "Lanczos"], "default": "Linear", "label": "Interpolation"}
                }
            },
            "Rotate": {
                "description": "Rotate the image",
                "function": "cv2.rotate",
                "params": {
                    "angle": {"type": "slider", "min": -180, "max": 180, "default": 0, "step": 1, "label": "Angle (degrees)"}
                }
            },
            "Flip Horizontal": {
                "description": "Flip image horizontally",
                "function": "cv2.flip",
                "params": {}
            },
            "Flip Vertical": {
                "description": "Flip image vertically",
                "function": "cv2.flip",
                "params": {}
            }
        }
    },
    "Special Effects": {
        "icon": "✨",
        "methods": {
            "Sharpen": {
                "description": "Sharpen the image",
                "function": "cv2.filter2D",
                "params": {
                    "strength": {"type": "slider", "min": 0.0, "max": 2.0, "default": 1.0, "step": 0.1, "label": "Strength"}
                }
            },
            "Emboss": {
                "description": "Apply emboss effect",
                "function": "cv2.filter2D",
                "params": {}
            },
            "Sketch Effect": {
                "description": "Convert image to pencil sketch",
                "function": "sketch",
                "params": {
                    "blur_sigma": {"type": "slider", "min": 1, "max": 100, "default": 21, "step": 2, "label": "Blur Sigma"}
                }
            },
            "Cartoon Effect": {
                "description": "Apply cartoon effect to image",
                "function": "cartoon",
                "params": {
                    "num_colors": {"type": "slider", "min": 2, "max": 20, "default": 9, "step": 1, "label": "Number of Colors"},
                    "blur_value": {"type": "slider", "min": 1, "max": 21, "default": 7, "step": 2, "label": "Blur Value"}
                }
            },
            "Sepia": {
                "description": "Apply sepia tone effect",
                "function": "sepia",
                "params": {}
            },
            "Vignette": {
                "description": "Add vignette effect",
                "function": "vignette",
                "params": {
                    "strength": {"type": "slider", "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.05, "label": "Strength"}
                }
            }
        }
    },
    "Contour Detection": {
        "icon": "🔍",
        "methods": {
            "Find Contours": {
                "description": "Detect and draw contours",
                "function": "cv2.findContours",
                "params": {
                    "mode": {"type": "dropdown", "options": ["External", "List", "Tree", "Component"], "default": "External", "label": "Retrieval Mode"},
                    "thickness": {"type": "slider", "min": 1, "max": 10, "default": 2, "step": 1, "label": "Line Thickness"},
                    "threshold": {"type": "slider", "min": 0, "max": 255, "default": 127, "step": 1, "label": "Threshold"}
                }
            }
        }
    },
    "Noise": {
        "icon": "📡",
        "methods": {
            "Add Gaussian Noise": {
                "description": "Add Gaussian noise to image",
                "function": "noise_gaussian",
                "params": {
                    "mean": {"type": "slider", "min": 0, "max": 50, "default": 0, "step": 1, "label": "Mean"},
                    "sigma": {"type": "slider", "min": 1, "max": 100, "default": 25, "step": 1, "label": "Sigma"}
                }
            },
            "Add Salt & Pepper Noise": {
                "description": "Add salt and pepper noise",
                "function": "noise_sp",
                "params": {
                    "amount": {"type": "slider", "min": 0.0, "max": 0.5, "default": 0.05, "step": 0.01, "label": "Amount"}
                }
            },
            "Denoise (fastNlMeans)": {
                "description": "Remove noise using Non-local Means",
                "function": "cv2.fastNlMeansDenoisingColored",
                "params": {
                    "h": {"type": "slider", "min": 1, "max": 20, "default": 10, "step": 1, "label": "Filter Strength"},
                    "hColor": {"type": "slider", "min": 1, "max": 20, "default": 10, "step": 1, "label": "Color Filter Strength"}
                }
            }
        }
    },
    "Helpers": {
        "icon": "🔧",
        "methods": {
            "Normalize": {
                "description": "Normalize pixel values to a range (enhances contrast)",
                "function": "cv2.normalize",
                "params": {
                    "alpha": {"type": "slider", "min": 0, "max": 255, "default": 0, "step": 1, "label": "Min Value (Alpha)"},
                    "beta": {"type": "slider", "min": 0, "max": 255, "default": 255, "step": 1, "label": "Max Value (Beta)"},
                    "norm_type": {"type": "dropdown", "options": ["MINMAX", "INF", "L1", "L2"], "default": "MINMAX", "label": "Norm Type"}
                }
            }
        }
    }
}


def get_kernel_shape(shape_name):
    """Get OpenCV kernel shape from string name"""
    shapes = {
        "Rectangle": cv2.MORPH_RECT,
        "Ellipse": cv2.MORPH_ELLIPSE,
        "Cross": cv2.MORPH_CROSS
    }
    return shapes.get(shape_name, cv2.MORPH_RECT)


def get_interpolation(interp_name):
    """Get OpenCV interpolation flag from string name"""
    interps = {
        "Nearest": cv2.INTER_NEAREST,
        "Linear": cv2.INTER_LINEAR,
        "Area": cv2.INTER_AREA,
        "Cubic": cv2.INTER_CUBIC,
        "Lanczos": cv2.INTER_LANCZOS4
    }
    return interps.get(interp_name, cv2.INTER_LINEAR)


def apply_opencv_method(image, category, method_name, params):
    """Apply the selected OpenCV method with given parameters"""
    img = image.copy()
    
    try:
        # Color Transformations
        if method_name == "cvtColor - Grayscale":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "cvtColor - HSV":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif method_name == "cvtColor - LAB":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif method_name == "Invert Colors":
            result = cv2.bitwise_not(img)
        
        # Blurring
        elif method_name == "GaussianBlur":
            ksize = int(params.get("kernel_size", 5))
            if ksize % 2 == 0:
                ksize += 1
            sigma = params.get("sigma", 0)
            result = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        elif method_name == "MedianBlur":
            ksize = int(params.get("kernel_size", 5))
            if ksize % 2 == 0:
                ksize += 1
            result = cv2.medianBlur(img, ksize)
        elif method_name == "BilateralFilter":
            d = int(params.get("d", 9))
            sigmaColor = params.get("sigmaColor", 75)
            sigmaSpace = params.get("sigmaSpace", 75)
            result = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
        elif method_name == "BoxFilter":
            ksize = int(params.get("kernel_size", 5))
            result = cv2.boxFilter(img, -1, (ksize, ksize))
        
        # Edge Detection
        elif method_name == "Canny":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = cv2.Canny(gray, params.get("threshold1", 100), params.get("threshold2", 200))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Sobel X":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ksize = int(params.get("ksize", 3))
            if ksize % 2 == 0:
                ksize += 1
            result = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            result = np.uint8(np.absolute(result))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Sobel Y":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ksize = int(params.get("ksize", 3))
            if ksize % 2 == 0:
                ksize += 1
            result = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            result = np.uint8(np.absolute(result))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Laplacian":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ksize = int(params.get("ksize", 3))
            if ksize % 2 == 0:
                ksize += 1
            result = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
            result = np.uint8(np.absolute(result))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # Thresholding
        elif method_name == "Binary Threshold":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, result = cv2.threshold(gray, params.get("thresh", 127), params.get("maxval", 255), cv2.THRESH_BINARY)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Adaptive Threshold Mean":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            block_size = int(params.get("block_size", 11))
            if block_size % 2 == 0:
                block_size += 1
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, params.get("C", 2))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Adaptive Threshold Gaussian":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            block_size = int(params.get("block_size", 11))
            if block_size % 2 == 0:
                block_size += 1
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, params.get("C", 2))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Otsu's Threshold":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # Morphological Operations
        elif method_name in ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"]:
            ksize = int(params.get("kernel_size", 5))
            shape = get_kernel_shape(params.get("kernel_shape", "Rectangle"))
            kernel = cv2.getStructuringElement(shape, (ksize, ksize))
            
            if method_name == "Erosion":
                iterations = int(params.get("iterations", 1))
                result = cv2.erode(img, kernel, iterations=iterations)
            elif method_name == "Dilation":
                iterations = int(params.get("iterations", 1))
                result = cv2.dilate(img, kernel, iterations=iterations)
            elif method_name == "Opening":
                result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            elif method_name == "Closing":
                result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            elif method_name == "Gradient":
                result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
            elif method_name == "Top Hat":
                result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
            elif method_name == "Black Hat":
                result = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        
        # Image Adjustments
        elif method_name == "Brightness & Contrast":
            alpha = params.get("alpha", 1.0)
            beta = params.get("beta", 0)
            result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        elif method_name == "Histogram Equalization":
            if len(img.shape) == 3:
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                result = cv2.equalizeHist(img)
        elif method_name == "CLAHE":
            clip_limit = params.get("clipLimit", 2.0)
            tile_size = int(params.get("tileGridSize", 8))
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            if len(img.shape) == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                result = clahe.apply(img)
        elif method_name == "Gamma Correction":
            gamma = params.get("gamma", 1.0)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            result = cv2.LUT(img, table)
        elif method_name == "Normalize":
            alpha = int(params.get("alpha", 0))
            beta = int(params.get("beta", 255))
            norm_type_str = params.get("norm_type", "MINMAX")
            norm_types = {
                "MINMAX": cv2.NORM_MINMAX,
                "INF": cv2.NORM_INF,
                "L1": cv2.NORM_L1,
                "L2": cv2.NORM_L2
            }
            norm_type = norm_types.get(norm_type_str, cv2.NORM_MINMAX)
            result = cv2.normalize(img, None, alpha, beta, norm_type)
        
        # Geometric Transformations
        elif method_name == "Resize":
            scale = params.get("scale", 1.0)
            interp = get_interpolation(params.get("interpolation", "Linear"))
            height, width = img.shape[:2]
            new_size = (int(width * scale), int(height * scale))
            result = cv2.resize(img, new_size, interpolation=interp)
        elif method_name == "Rotate":
            angle = params.get("angle", 0)
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(img, matrix, (width, height))
        elif method_name == "Flip Horizontal":
            result = cv2.flip(img, 1)
        elif method_name == "Flip Vertical":
            result = cv2.flip(img, 0)
        
        # Special Effects
        elif method_name == "Sharpen":
            strength = params.get("strength", 1.0)
            kernel = np.array([[-1, -1, -1],
                              [-1, 9 + strength, -1],
                              [-1, -1, -1]])
            result = cv2.filter2D(img, -1, kernel)
        elif method_name == "Emboss":
            kernel = np.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])
            result = cv2.filter2D(img, -1, kernel)
        elif method_name == "Sketch Effect":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inv_gray = 255 - gray
            blur_sigma = int(params.get("blur_sigma", 21))
            if blur_sigma % 2 == 0:
                blur_sigma += 1
            blur = cv2.GaussianBlur(inv_gray, (blur_sigma, blur_sigma), 0)
            result = cv2.divide(gray, 255 - blur, scale=256)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Cartoon Effect":
            num_colors = int(params.get("num_colors", 9))
            blur_val = int(params.get("blur_value", 7))
            if blur_val % 2 == 0:
                blur_val += 1
            
            # Edge detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, blur_val)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            
            # Color quantization
            data = np.float32(img).reshape((-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
            _, labels, centers = cv2.kmeans(data, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            quantized = centers[labels.flatten()].reshape(img.shape)
            
            # Combine
            result = cv2.bitwise_and(quantized, quantized, mask=edges)
        elif method_name == "Sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            result = cv2.transform(img, kernel)
            result = np.clip(result, 0, 255).astype(np.uint8)
        elif method_name == "Vignette":
            strength = params.get("strength", 0.5)
            rows, cols = img.shape[:2]
            X = np.arange(0, cols)
            Y = np.arange(0, rows)
            X, Y = np.meshgrid(X, Y)
            centerX, centerY = cols / 2, rows / 2
            mask = np.sqrt((X - centerX) ** 2 + (Y - centerY) ** 2)
            mask = mask / mask.max()
            mask = 1 - mask * strength
            mask = np.dstack([mask] * 3)
            result = (img * mask).astype(np.uint8)
        
        # Contour Detection
        elif method_name == "Find Contours":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh_val = int(params.get("threshold", 127))
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            
            mode_map = {
                "External": cv2.RETR_EXTERNAL,
                "List": cv2.RETR_LIST,
                "Tree": cv2.RETR_TREE,
                "Component": cv2.RETR_CCOMP
            }
            mode = mode_map.get(params.get("mode", "External"), cv2.RETR_EXTERNAL)
            thickness = int(params.get("thickness", 2))
            
            contours, _ = cv2.findContours(thresh, mode, cv2.CHAIN_APPROX_SIMPLE)
            result = img.copy()
            cv2.drawContours(result, contours, -1, (0, 255, 0), thickness)
        
        # Noise
        elif method_name == "Add Gaussian Noise":
            mean = params.get("mean", 0)
            sigma = params.get("sigma", 25)
            noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
            result = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif method_name == "Add Salt & Pepper Noise":
            amount = params.get("amount", 0.05)
            result = img.copy()
            num_salt = np.ceil(amount * img.size * 0.5)
            num_pepper = np.ceil(amount * img.size * 0.5)
            
            # Salt
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
            result[coords[0], coords[1], :] = 255
            
            # Pepper
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
            result[coords[0], coords[1], :] = 0
        elif method_name == "Denoise (fastNlMeans)":
            h = int(params.get("h", 10))
            hColor = int(params.get("hColor", 10))
            result = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, 7, 21)
        
        else:
            result = img
        
        return result
    
    except Exception as e:
        st.error(f"Error applying {method_name}: {str(e)}")
        return img


def apply_effect_pipeline(image, pipeline):
    """Apply a full pipeline of effects to the image"""
    result = image.copy()
    for effect in pipeline:
        result = apply_opencv_method(
            result,
            effect['category'],
            effect['method'],
            effect['params']
        )
    return result


def get_pipeline_summary(pipeline):
    """Get a readable summary of the effect pipeline"""
    if not pipeline:
        return "No effects"
    return " → ".join([e['method'] for e in pipeline])


def generate_opencv_code(method_name, params, category=None):
    """Generate OpenCV Python code for a given method and parameters"""
    code_lines = []
    param_info = []
    
    # Color Transformations
    if method_name == "cvtColor - Grayscale":
        code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        code_lines.append("result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)")
        param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_BGR2GRAY"}})
    elif method_name == "cvtColor - HSV":
        code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)")
        param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_BGR2HSV"}})
    elif method_name == "cvtColor - LAB":
        code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)")
        param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_BGR2LAB"}})
    elif method_name == "Invert Colors":
        code_lines.append("result = cv2.bitwise_not(img)")
        param_info.append({"function": "cv2.bitwise_not", "params": {}})
    
    # Blurring
    elif method_name == "GaussianBlur":
        ksize = int(params.get("kernel_size", 5))
        if ksize % 2 == 0:
            ksize += 1
        sigma = params.get("sigma", 0)
        code_lines.append(f"result = cv2.GaussianBlur(img, ({ksize}, {ksize}), {sigma})")
        param_info.append({"function": "cv2.GaussianBlur", "params": {"ksize": f"({ksize}, {ksize})", "sigmaX": sigma}})
    elif method_name == "MedianBlur":
        ksize = int(params.get("kernel_size", 5))
        if ksize % 2 == 0:
            ksize += 1
        code_lines.append(f"result = cv2.medianBlur(img, {ksize})")
        param_info.append({"function": "cv2.medianBlur", "params": {"ksize": ksize}})
    elif method_name == "BilateralFilter":
        d = int(params.get("d", 9))
        sigmaColor = params.get("sigmaColor", 75)
        sigmaSpace = params.get("sigmaSpace", 75)
        code_lines.append(f"result = cv2.bilateralFilter(img, {d}, {sigmaColor}, {sigmaSpace})")
        param_info.append({"function": "cv2.bilateralFilter", "params": {"d": d, "sigmaColor": sigmaColor, "sigmaSpace": sigmaSpace}})
    elif method_name == "BoxFilter":
        ksize = int(params.get("kernel_size", 5))
        code_lines.append(f"result = cv2.boxFilter(img, -1, ({ksize}, {ksize}))")
        param_info.append({"function": "cv2.boxFilter", "params": {"ddepth": -1, "ksize": f"({ksize}, {ksize})"}})
    
    # Edge Detection
    elif method_name == "Canny":
        t1 = params.get("threshold1", 100)
        t2 = params.get("threshold2", 200)
        code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        code_lines.append(f"edges = cv2.Canny(gray, {t1}, {t2})")
        code_lines.append("result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)")
        param_info.append({"function": "cv2.Canny", "params": {"threshold1": t1, "threshold2": t2}})
    elif method_name == "Sobel X":
        ksize = int(params.get("ksize", 3))
        if ksize % 2 == 0:
            ksize += 1
        code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        code_lines.append(f"sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize={ksize})")
        code_lines.append("result = np.uint8(np.absolute(sobel))")
        code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
        param_info.append({"function": "cv2.Sobel", "params": {"ddepth": "cv2.CV_64F", "dx": 1, "dy": 0, "ksize": ksize}})
    elif method_name == "Sobel Y":
        ksize = int(params.get("ksize", 3))
        if ksize % 2 == 0:
            ksize += 1
        code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        code_lines.append(f"sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize={ksize})")
        code_lines.append("result = np.uint8(np.absolute(sobel))")
        code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
        param_info.append({"function": "cv2.Sobel", "params": {"ddepth": "cv2.CV_64F", "dx": 0, "dy": 1, "ksize": ksize}})
    elif method_name == "Laplacian":
        ksize = int(params.get("ksize", 3))
        if ksize % 2 == 0:
            ksize += 1
        code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        code_lines.append(f"laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize={ksize})")
        code_lines.append("result = np.uint8(np.absolute(laplacian))")
        code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
        param_info.append({"function": "cv2.Laplacian", "params": {"ddepth": "cv2.CV_64F", "ksize": ksize}})
    
    # Thresholding
    elif method_name == "Binary Threshold":
        thresh = params.get("thresh", 127)
        maxval = params.get("maxval", 255)
        code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        code_lines.append(f"_, result = cv2.threshold(gray, {thresh}, {maxval}, cv2.THRESH_BINARY)")
        code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
        param_info.append({"function": "cv2.threshold", "params": {"thresh": thresh, "maxval": maxval, "type": "cv2.THRESH_BINARY"}})
    elif method_name == "Adaptive Threshold Mean":
        block_size = int(params.get("block_size", 11))
        if block_size % 2 == 0:
            block_size += 1
        C = params.get("C", 2)
        code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        code_lines.append(f"result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, {block_size}, {C})")
        code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
        param_info.append({"function": "cv2.adaptiveThreshold", "params": {"maxValue": 255, "adaptiveMethod": "MEAN_C", "blockSize": block_size, "C": C}})
    elif method_name == "Adaptive Threshold Gaussian":
        block_size = int(params.get("block_size", 11))
        if block_size % 2 == 0:
            block_size += 1
        C = params.get("C", 2)
        code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        code_lines.append(f"result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, {block_size}, {C})")
        code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
        param_info.append({"function": "cv2.adaptiveThreshold", "params": {"maxValue": 255, "adaptiveMethod": "GAUSSIAN_C", "blockSize": block_size, "C": C}})
    elif method_name == "Otsu's Threshold":
        code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        code_lines.append("_, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)")
        code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
        param_info.append({"function": "cv2.threshold", "params": {"type": "THRESH_BINARY + THRESH_OTSU"}})
    
    # Morphological Operations
    elif method_name in ["Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"]:
        ksize = int(params.get("kernel_size", 5))
        shape_name = params.get("kernel_shape", "Rectangle")
        shape_map = {"Rectangle": "cv2.MORPH_RECT", "Ellipse": "cv2.MORPH_ELLIPSE", "Cross": "cv2.MORPH_CROSS"}
        shape = shape_map.get(shape_name, "cv2.MORPH_RECT")
        code_lines.append(f"kernel = cv2.getStructuringElement({shape}, ({ksize}, {ksize}))")
        
        if method_name == "Erosion":
            iterations = int(params.get("iterations", 1))
            code_lines.append(f"result = cv2.erode(img, kernel, iterations={iterations})")
            param_info.append({"function": "cv2.erode", "params": {"kernel_size": ksize, "kernel_shape": shape_name, "iterations": iterations}})
        elif method_name == "Dilation":
            iterations = int(params.get("iterations", 1))
            code_lines.append(f"result = cv2.dilate(img, kernel, iterations={iterations})")
            param_info.append({"function": "cv2.dilate", "params": {"kernel_size": ksize, "kernel_shape": shape_name, "iterations": iterations}})
        elif method_name == "Opening":
            code_lines.append("result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)")
            param_info.append({"function": "cv2.morphologyEx", "params": {"op": "MORPH_OPEN", "kernel_size": ksize, "kernel_shape": shape_name}})
        elif method_name == "Closing":
            code_lines.append("result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)")
            param_info.append({"function": "cv2.morphologyEx", "params": {"op": "MORPH_CLOSE", "kernel_size": ksize, "kernel_shape": shape_name}})
        elif method_name == "Gradient":
            code_lines.append("result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)")
            param_info.append({"function": "cv2.morphologyEx", "params": {"op": "MORPH_GRADIENT", "kernel_size": ksize, "kernel_shape": shape_name}})
        elif method_name == "Top Hat":
            code_lines.append("result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)")
            param_info.append({"function": "cv2.morphologyEx", "params": {"op": "MORPH_TOPHAT", "kernel_size": ksize, "kernel_shape": shape_name}})
        elif method_name == "Black Hat":
            code_lines.append("result = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)")
            param_info.append({"function": "cv2.morphologyEx", "params": {"op": "MORPH_BLACKHAT", "kernel_size": ksize, "kernel_shape": shape_name}})
    
    # Image Adjustments
    elif method_name == "Brightness & Contrast":
        alpha = params.get("alpha", 1.0)
        beta = params.get("beta", 0)
        code_lines.append(f"result = cv2.convertScaleAbs(img, alpha={alpha}, beta={beta})")
        param_info.append({"function": "cv2.convertScaleAbs", "params": {"alpha (contrast)": alpha, "beta (brightness)": beta}})
    elif method_name == "Histogram Equalization":
        code_lines.append("ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)")
        code_lines.append("ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])")
        code_lines.append("result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)")
        param_info.append({"function": "cv2.equalizeHist", "params": {}})
    elif method_name == "CLAHE":
        clip_limit = params.get("clipLimit", 2.0)
        tile_size = int(params.get("tileGridSize", 8))
        code_lines.append(f"clahe = cv2.createCLAHE(clipLimit={clip_limit}, tileGridSize=({tile_size}, {tile_size}))")
        code_lines.append("lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)")
        code_lines.append("lab[:, :, 0] = clahe.apply(lab[:, :, 0])")
        code_lines.append("result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)")
        param_info.append({"function": "cv2.createCLAHE", "params": {"clipLimit": clip_limit, "tileGridSize": f"({tile_size}, {tile_size})"}})
    elif method_name == "Gamma Correction":
        gamma = params.get("gamma", 1.0)
        code_lines.append(f"gamma = {gamma}")
        code_lines.append("inv_gamma = 1.0 / gamma")
        code_lines.append("table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype('uint8')")
        code_lines.append("result = cv2.LUT(img, table)")
        param_info.append({"function": "cv2.LUT", "params": {"gamma": gamma}})
    elif method_name == "Normalize":
        alpha = int(params.get("alpha", 0))
        beta = int(params.get("beta", 255))
        norm_type = params.get("norm_type", "MINMAX")
        code_lines.append(f"result = cv2.normalize(img, None, {alpha}, {beta}, cv2.NORM_{norm_type})")
        param_info.append({"function": "cv2.normalize", "params": {"alpha": alpha, "beta": beta, "norm_type": norm_type}})
    
    # Geometric Transformations
    elif method_name == "Resize":
        scale = params.get("scale", 1.0)
        interp = params.get("interpolation", "Linear")
        interp_map = {"Nearest": "INTER_NEAREST", "Linear": "INTER_LINEAR", "Area": "INTER_AREA", "Cubic": "INTER_CUBIC", "Lanczos": "INTER_LANCZOS4"}
        code_lines.append(f"height, width = img.shape[:2]")
        code_lines.append(f"new_size = (int(width * {scale}), int(height * {scale}))")
        code_lines.append(f"result = cv2.resize(img, new_size, interpolation=cv2.{interp_map.get(interp, 'INTER_LINEAR')})")
        param_info.append({"function": "cv2.resize", "params": {"scale": scale, "interpolation": interp}})
    elif method_name == "Rotate":
        angle = params.get("angle", 0)
        code_lines.append("height, width = img.shape[:2]")
        code_lines.append("center = (width // 2, height // 2)")
        code_lines.append(f"matrix = cv2.getRotationMatrix2D(center, {angle}, 1.0)")
        code_lines.append("result = cv2.warpAffine(img, matrix, (width, height))")
        param_info.append({"function": "cv2.warpAffine", "params": {"angle": angle}})
    elif method_name == "Flip Horizontal":
        code_lines.append("result = cv2.flip(img, 1)")
        param_info.append({"function": "cv2.flip", "params": {"flipCode": "1 (horizontal)"}})
    elif method_name == "Flip Vertical":
        code_lines.append("result = cv2.flip(img, 0)")
        param_info.append({"function": "cv2.flip", "params": {"flipCode": "0 (vertical)"}})
    
    # Special Effects
    elif method_name == "Sharpen":
        strength = params.get("strength", 1.0)
        code_lines.append(f"kernel = np.array([[-1, -1, -1], [-1, 9 + {strength}, -1], [-1, -1, -1]])")
        code_lines.append("result = cv2.filter2D(img, -1, kernel)")
        param_info.append({"function": "cv2.filter2D", "params": {"strength": strength}})
    elif method_name == "Emboss":
        code_lines.append("kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])")
        code_lines.append("result = cv2.filter2D(img, -1, kernel)")
        param_info.append({"function": "cv2.filter2D", "params": {"effect": "emboss"}})
    elif method_name == "Sketch Effect":
        blur_sigma = int(params.get("blur_sigma", 21))
        if blur_sigma % 2 == 0:
            blur_sigma += 1
        code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        code_lines.append("inv_gray = 255 - gray")
        code_lines.append(f"blur = cv2.GaussianBlur(inv_gray, ({blur_sigma}, {blur_sigma}), 0)")
        code_lines.append("result = cv2.divide(gray, 255 - blur, scale=256)")
        code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
        param_info.append({"function": "sketch_effect", "params": {"blur_sigma": blur_sigma}})
    elif method_name == "Cartoon Effect":
        num_colors = int(params.get("num_colors", 9))
        blur_val = int(params.get("blur_value", 7))
        if blur_val % 2 == 0:
            blur_val += 1
        code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        code_lines.append(f"gray = cv2.medianBlur(gray, {blur_val})")
        code_lines.append("edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)")
        code_lines.append("data = np.float32(img).reshape((-1, 3))")
        code_lines.append("criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)")
        code_lines.append(f"_, labels, centers = cv2.kmeans(data, {num_colors}, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)")
        code_lines.append("centers = np.uint8(centers)")
        code_lines.append("quantized = centers[labels.flatten()].reshape(img.shape)")
        code_lines.append("result = cv2.bitwise_and(quantized, quantized, mask=edges)")
        param_info.append({"function": "cartoon_effect", "params": {"num_colors": num_colors, "blur_value": blur_val}})
    elif method_name == "Sepia":
        code_lines.append("kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])")
        code_lines.append("result = cv2.transform(img, kernel)")
        code_lines.append("result = np.clip(result, 0, 255).astype(np.uint8)")
        param_info.append({"function": "cv2.transform", "params": {"effect": "sepia"}})
    elif method_name == "Vignette":
        strength = params.get("strength", 0.5)
        code_lines.append("rows, cols = img.shape[:2]")
        code_lines.append("X, Y = np.meshgrid(np.arange(cols), np.arange(rows))")
        code_lines.append("centerX, centerY = cols / 2, rows / 2")
        code_lines.append("mask = np.sqrt((X - centerX) ** 2 + (Y - centerY) ** 2)")
        code_lines.append("mask = mask / mask.max()")
        code_lines.append(f"mask = 1 - mask * {strength}")
        code_lines.append("mask = np.dstack([mask] * 3)")
        code_lines.append("result = (img * mask).astype(np.uint8)")
        param_info.append({"function": "vignette_effect", "params": {"strength": strength}})
    
    # Contour Detection
    elif method_name == "Find Contours":
        thresh_val = int(params.get("threshold", 127))
        mode = params.get("mode", "External")
        thickness = int(params.get("thickness", 2))
        mode_map = {"External": "RETR_EXTERNAL", "List": "RETR_LIST", "Tree": "RETR_TREE", "Component": "RETR_CCOMP"}
        code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
        code_lines.append(f"_, thresh = cv2.threshold(gray, {thresh_val}, 255, cv2.THRESH_BINARY)")
        code_lines.append(f"contours, _ = cv2.findContours(thresh, cv2.{mode_map.get(mode, 'RETR_EXTERNAL')}, cv2.CHAIN_APPROX_SIMPLE)")
        code_lines.append("result = img.copy()")
        code_lines.append(f"cv2.drawContours(result, contours, -1, (0, 255, 0), {thickness})")
        param_info.append({"function": "cv2.findContours", "params": {"threshold": thresh_val, "mode": mode, "thickness": thickness}})
    
    # Noise
    elif method_name == "Add Gaussian Noise":
        mean = params.get("mean", 0)
        sigma = params.get("sigma", 25)
        code_lines.append(f"noise = np.random.normal({mean}, {sigma}, img.shape).astype(np.float32)")
        code_lines.append("result = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)")
        param_info.append({"function": "gaussian_noise", "params": {"mean": mean, "sigma": sigma}})
    elif method_name == "Add Salt & Pepper Noise":
        amount = params.get("amount", 0.05)
        code_lines.append("result = img.copy()")
        code_lines.append(f"num_salt = int({amount} * img.size * 0.5)")
        code_lines.append("coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]")
        code_lines.append("result[coords[0], coords[1], :] = 255  # Salt")
        code_lines.append("coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]")
        code_lines.append("result[coords[0], coords[1], :] = 0  # Pepper")
        param_info.append({"function": "salt_pepper_noise", "params": {"amount": amount}})
    elif method_name == "Denoise (fastNlMeans)":
        h = int(params.get("h", 10))
        hColor = int(params.get("hColor", 10))
        code_lines.append(f"result = cv2.fastNlMeansDenoisingColored(img, None, {h}, {hColor}, 7, 21)")
        param_info.append({"function": "cv2.fastNlMeansDenoisingColored", "params": {"h": h, "hColor": hColor}})
    
    else:
        code_lines.append("result = img.copy()  # No operation")
        param_info.append({"function": "copy", "params": {}})
    
    return {
        "code_lines": code_lines,
        "param_info": param_info,
        "method": method_name
    }


def generate_full_pipeline_code(pipeline):
    """Generate complete Python code for the entire pipeline"""
    if not pipeline:
        return "# No effects in pipeline"
    
    code = """import cv2
import numpy as np

# Load your image
img = cv2.imread('your_image.jpg')

"""
    
    for i, effect in enumerate(pipeline):
        code += f"# Step {i + 1}: {effect['method']}\n"
        code_info = generate_opencv_code(effect['method'], effect['params'], effect.get('category'))
        
        for line in code_info['code_lines']:
            # Replace 'img' with 'result' for steps after the first
            if i > 0:
                line = line.replace('(img,', '(result,').replace('(img)', '(result)').replace(' img.', ' result.').replace(' img,', ' result,').replace('= img.', '= result.')
                if line.strip().startswith('img.'):
                    line = line.replace('img.', 'result.')
            code += line + "\n"
        
        # Rename result to img for next step (except for the last step)
        if i < len(pipeline) - 1:
            code += "img = result.copy()\n"
        code += "\n"
    
    code += """# Save or display the result
cv2.imwrite('output.jpg', result)
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
"""
    
    return code


def save_to_history(image, method_name, params, pipeline=None):
    """Save processed image to history with optional full pipeline"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Create filename based on pipeline or single method
    if pipeline and len(pipeline) > 0:
        method_str = "_".join([e['method'].replace(' ', '-') for e in pipeline[:3]])  # First 3 methods
        if len(pipeline) > 3:
            method_str += f"_+{len(pipeline)-3}more"
        filename = f"{timestamp}_{method_str}.png"
    else:
        filename = f"{timestamp}_{method_name.replace(' ', '_')}.png"
    
    filepath = OUTPUT_DIR / filename
    
    # Save the image
    cv2.imwrite(str(filepath), image)
    
    # Create thumbnail with aspect ratio preserved
    h, w = image.shape[:2]
    max_thumb_size = 80
    if w > h:
        new_w = max_thumb_size
        new_h = int(h * max_thumb_size / w)
    else:
        new_h = max_thumb_size
        new_w = int(w * max_thumb_size / h)
    thumbnail = cv2.resize(image, (new_w, new_h))
    
    # Create serializable pipeline (without numpy arrays)
    serializable_pipeline = []
    if pipeline:
        for effect in pipeline:
            serializable_pipeline.append({
                'category': effect['category'],
                'method': effect['method'],
                'params': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                          for k, v in effect['params'].items()}
            })
    
    # Add to history
    history_item = {
        "timestamp": timestamp,
        "method": method_name if not pipeline else get_pipeline_summary(pipeline),
        "params": params,
        "pipeline": serializable_pipeline,  # Store full pipeline
        "filepath": str(filepath),
        "filename": filename,
        "thumbnail": thumbnail
    }
    st.session_state.history.insert(0, history_item)
    
    # Keep only last 50 items
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[:50]
    
    return filepath


def load_image_from_file(uploaded_file):
    """Load image from uploaded file"""
    uploaded_file.seek(0)  # Reset file pointer to beginning
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)  # Reset again for potential re-reads
    return image


def load_video_frame(uploaded_file, frame_number=0):
    """Load a specific frame from video file"""
    import tempfile
    
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    uploaded_file.seek(0)
    
    # Read video
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_number >= total_frames:
        frame_number = total_frames - 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    # Clean up temp file
    os.unlink(tfile.name)
    
    if ret:
        return frame, total_frames
    return None, total_frames


def get_video_info(uploaded_file):
    """Get video information"""
    import tempfile
    
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    uploaded_file.seek(0)
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    os.unlink(tfile.name)
    
    return {
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': total_frames / fps if fps > 0 else 0
    }
    return image


def image_to_base64(image):
    """Convert image to base64 for display"""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode()


# ============ Callback Functions for Pipeline Actions ============
# These execute BEFORE re-render, so no st.rerun() needed and no scroll-to-top

def add_effect_callback():
    """Callback for adding effect to pipeline"""
    # Get current selections from session state
    cat_idx = st.session_state.get('category_select', 0)
    if cat_idx > 0:
        categories = list(OPENCV_METHODS.keys())
        selected_cat = categories[cat_idx - 1]
        method_idx = st.session_state.get('method_select', 0)
        if method_idx > 0:
            methods = list(OPENCV_METHODS[selected_cat]["methods"].keys())
            selected_meth = methods[method_idx - 1]
            
            # Gather param values
            params = {}
            method_info = OPENCV_METHODS[selected_cat]["methods"][selected_meth]
            for param_name, param_config in method_info.get("params", {}).items():
                param_key = f"param_{param_name}"
                if param_key in st.session_state:
                    params[param_name] = st.session_state[param_key]
                else:
                    params[param_name] = param_config.get("default")
            
            new_effect = {
                'id': st.session_state.next_effect_id,
                'category': selected_cat,
                'method': selected_meth,
                'params': params
            }
            st.session_state.effect_pipeline.append(new_effect)
            st.session_state.next_effect_id += 1
            # Reset dropdowns to None
            st.session_state.selected_category_index = 0
            st.session_state.selected_method_index = 0

def move_effect_up(idx):
    """Callback to move effect up in pipeline"""
    if idx > 0:
        st.session_state.effect_pipeline[idx], st.session_state.effect_pipeline[idx-1] = \
            st.session_state.effect_pipeline[idx-1], st.session_state.effect_pipeline[idx]

def move_effect_down(idx):
    """Callback to move effect down in pipeline"""
    if idx < len(st.session_state.effect_pipeline) - 1:
        st.session_state.effect_pipeline[idx], st.session_state.effect_pipeline[idx+1] = \
            st.session_state.effect_pipeline[idx+1], st.session_state.effect_pipeline[idx]

def toggle_edit_effect(effect_id):
    """Callback to toggle edit mode for an effect"""
    if st.session_state.editing_effect_id == effect_id:
        st.session_state.editing_effect_id = None
    else:
        st.session_state.editing_effect_id = effect_id

def remove_effect(idx, effect_id):
    """Callback to remove effect from pipeline"""
    if idx < len(st.session_state.effect_pipeline):
        st.session_state.effect_pipeline.pop(idx)
    if st.session_state.editing_effect_id == effect_id:
        st.session_state.editing_effect_id = None

def clear_pipeline():
    """Callback to clear all effects"""
    st.session_state.effect_pipeline = []
    st.session_state.editing_effect_id = None


# Main App Layout
st.markdown("""
<div class="main-header">
    <h1>🎨 OpenCV Playground</h1>
    <p>Explore and apply OpenCV image processing methods in real-time</p>
</div>
""", unsafe_allow_html=True)

# Toggle for Code Panel
code_panel_col1, code_panel_col2 = st.columns([3, 1])
with code_panel_col2:
    if st.button("💻 Code Panel" if not st.session_state.show_code_panel else "❌ Close Code", 
                  type="primary" if not st.session_state.show_code_panel else "secondary",
                  key="toggle_code_panel"):
        st.session_state.show_code_panel = not st.session_state.show_code_panel

# Create columns based on whether code panel is shown
if st.session_state.show_code_panel:
    left_col, middle_col, right_col, code_col = st.columns([1.2, 1.8, 0.8, 1.2])
else:
    left_col, middle_col, right_col = st.columns([1.2, 2, 1])

# Left Sidebar - Effect Pipeline
with left_col:
    st.markdown("### ➕ Add Effect")
    
    # Category selection with "None" as default
    categories = list(OPENCV_METHODS.keys())
    category_options = ["-- Select a category --"] + [f"{OPENCV_METHODS[cat]['icon']} {cat}" for cat in categories]
    
    selected_category_index = st.selectbox(
        "Select Category",
        range(len(category_options)),
        format_func=lambda x: category_options[x],
        index=st.session_state.selected_category_index if st.session_state.selected_category_index < len(category_options) else 0,
        key="category_select"
    )
    
    # Update session state
    st.session_state.selected_category_index = selected_category_index
    
    # Check if a category is selected
    category_selected = selected_category_index > 0
    selected_category = categories[selected_category_index - 1] if category_selected else None
    
    # Only show method selection if category is selected
    param_values = {}
    method_selected = False
    selected_method = None
    
    if category_selected and selected_category:
        # Method selection - with "None" as default
        methods = list(OPENCV_METHODS[selected_category]["methods"].keys())
        method_options = ["-- Select an effect --"] + methods
        
        selected_method_index = st.selectbox(
            "Select Method",
            range(len(method_options)),
            format_func=lambda x: method_options[x],
            index=st.session_state.selected_method_index if st.session_state.selected_method_index < len(method_options) else 0,
            key="method_select"
        )
        
        # Update session state
        st.session_state.selected_method_index = selected_method_index
        
        # Check if a method is selected (not "-- Select --")
        method_selected = selected_method_index > 0
        selected_method = methods[selected_method_index - 1] if method_selected else None
    else:
        # Reset method index when category changes to None
        st.session_state.selected_method_index = 0
    
    # Only show method info and params if a method is selected
    if method_selected and selected_method:
        # Get method info
        method_info = OPENCV_METHODS[selected_category]["methods"][selected_method]
        
        # Display method description
        st.markdown(f"""
        <div class="method-card">
            <strong>📋 {selected_method}</strong><br>
            <span style="font-size: 12px;">{method_info['description']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Dynamic parameter controls
        params_config = method_info.get("params", {})
        
        if params_config:
            st.markdown("**⚙️ Parameters**")
            for param_name, param_config in params_config.items():
                param_type = param_config.get("type", "slider")
                label = param_config.get("label", param_name)
                
                if param_type == "slider":
                    if isinstance(param_config.get("default"), float) or isinstance(param_config.get("step"), float):
                        param_values[param_name] = st.slider(
                            label,
                            min_value=float(param_config.get("min", 0)),
                            max_value=float(param_config.get("max", 100)),
                            value=float(param_config.get("default", 0)),
                            step=float(param_config.get("step", 1)),
                            key=f"param_{param_name}"
                        )
                    else:
                        param_values[param_name] = st.slider(
                            label,
                            min_value=int(param_config.get("min", 0)),
                            max_value=int(param_config.get("max", 100)),
                            value=int(param_config.get("default", 0)),
                            step=int(param_config.get("step", 1)),
                            key=f"param_{param_name}"
                        )
                
                elif param_type == "dropdown":
                    options = param_config.get("options", [])
                    default = param_config.get("default", options[0] if options else "")
                    param_values[param_name] = st.selectbox(
                        label,
                        options,
                        index=options.index(default) if default in options else 0,
                        key=f"param_{param_name}"
                    )
                
                elif param_type == "checkbox":
                    param_values[param_name] = st.checkbox(
                        label,
                        value=param_config.get("default", False),
                        key=f"param_{param_name}"
                    )
        
        # Add effect button - uses callback for immediate update without scroll
        st.button("➕ Add to Pipeline", type="primary", key="add_effect_btn", on_click=add_effect_callback)
    else:
        st.info("👆 Select a category and effect from the dropdowns above")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Display current pipeline
    st.markdown("### 🔗 Effect Pipeline")
    
    if st.session_state.effect_pipeline:
        st.caption(f"{len(st.session_state.effect_pipeline)} effect(s)")
        
        # Display each effect with modern card design
        for idx, effect in enumerate(st.session_state.effect_pipeline):
            category_icon = OPENCV_METHODS.get(effect['category'], {}).get('icon', '🔧')
            
            # Check if this effect is being edited
            is_editing = st.session_state.editing_effect_id == effect['id']
            
            # Use unique key combining id and index to avoid duplicates
            unique_key = f"{effect['id']}_{idx}"
            
            # Modern card container
            with st.container():
                # Card header with number, name
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                    <span class="pipeline-number">{idx + 1}</span>
                    <span style="font-weight: 500; font-size: 13px;">{category_icon} {effect['method']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons using on_click callbacks (no scroll-to-top)
                btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                with btn_col1:
                    if idx > 0:
                        st.button("↑", key=f"up_{unique_key}", help="Move up", 
                                  on_click=move_effect_up, args=(idx,))
                with btn_col2:
                    if idx < len(st.session_state.effect_pipeline) - 1:
                        st.button("↓", key=f"down_{unique_key}", help="Move down",
                                  on_click=move_effect_down, args=(idx,))
                with btn_col3:
                    edit_icon = "✓" if is_editing else "✎"
                    st.button(edit_icon, key=f"edit_{unique_key}", help="Edit",
                              on_click=toggle_edit_effect, args=(effect['id'],))
                with btn_col4:
                    st.button("×", key=f"remove_{unique_key}", help="Remove",
                              on_click=remove_effect, args=(idx, effect['id']))
                
                # Expanded edit panel
                if is_editing:
                    with st.expander("Edit Effect", expanded=True):
                        # Store effect index for callbacks
                        effect_idx = idx
                        
                        # Category selection for editing
                        edit_categories = list(OPENCV_METHODS.keys())
                        edit_category_options = [f"{OPENCV_METHODS[cat]['icon']} {cat}" for cat in edit_categories]
                        current_cat_idx = edit_categories.index(effect['category']) if effect['category'] in edit_categories else 0
                        
                        # Initialize session state for this effect's category/method if not exists
                        cat_state_key = f"edit_cat_state_{effect['id']}"
                        method_state_key = f"edit_method_state_{effect['id']}"
                        if cat_state_key not in st.session_state:
                            st.session_state[cat_state_key] = effect['category']
                        if method_state_key not in st.session_state:
                            st.session_state[method_state_key] = effect['method']
                        
                        def on_category_change(eff_idx, eff_id):
                            cat_key = f"edit_cat_{eff_id}_{eff_idx}"
                            cat_state = f"edit_cat_state_{eff_id}"
                            method_state = f"edit_method_state_{eff_id}"
                            if cat_key in st.session_state:
                                new_cat_display = st.session_state[cat_key]
                                cats = list(OPENCV_METHODS.keys())
                                cat_opts = [f"{OPENCV_METHODS[c]['icon']} {c}" for c in cats]
                                new_cat = cats[cat_opts.index(new_cat_display)]
                                st.session_state[cat_state] = new_cat
                                # Reset method to first in new category
                                first_method = list(OPENCV_METHODS[new_cat]["methods"].keys())[0]
                                st.session_state[method_state] = first_method
                                # Update effect
                                st.session_state.effect_pipeline[eff_idx]['category'] = new_cat
                                st.session_state.effect_pipeline[eff_idx]['method'] = first_method
                                # Reset params to defaults
                                method_info = OPENCV_METHODS[new_cat]["methods"][first_method]
                                st.session_state.effect_pipeline[eff_idx]['params'] = {
                                    k: v.get("default") for k, v in method_info.get("params", {}).items()
                                }
                        
                        def on_method_change(eff_idx, eff_id):
                            method_key = f"edit_method_{eff_id}_{eff_idx}"
                            method_state = f"edit_method_state_{eff_id}"
                            if method_key in st.session_state:
                                new_method = st.session_state[method_key]
                                st.session_state[method_state] = new_method
                                cat = st.session_state.effect_pipeline[eff_idx]['category']
                                st.session_state.effect_pipeline[eff_idx]['method'] = new_method
                                # Reset params to defaults
                                method_info = OPENCV_METHODS[cat]["methods"][new_method]
                                st.session_state.effect_pipeline[eff_idx]['params'] = {
                                    k: v.get("default") for k, v in method_info.get("params", {}).items()
                                }
                        
                        def on_param_change(eff_idx, param_name, param_key):
                            if param_key in st.session_state:
                                st.session_state.effect_pipeline[eff_idx]['params'][param_name] = st.session_state[param_key]
                        
                        new_category_display = st.selectbox(
                            "Category",
                            edit_category_options,
                            index=current_cat_idx,
                            key=f"edit_cat_{effect['id']}_{idx}",
                            on_change=on_category_change,
                            args=(effect_idx, effect['id'])
                        )
                        
                        # Get current category from state
                        current_category = st.session_state.get(cat_state_key, effect['category'])
                        
                        # Method selection for editing
                        edit_methods = list(OPENCV_METHODS[current_category]["methods"].keys())
                        current_method = st.session_state.get(method_state_key, effect['method'])
                        if current_method in edit_methods:
                            current_method_idx = edit_methods.index(current_method)
                        else:
                            current_method_idx = 0
                        
                        new_method = st.selectbox(
                            "Effect",
                            edit_methods,
                            index=current_method_idx,
                            key=f"edit_method_{effect['id']}_{idx}",
                            on_change=on_method_change,
                            args=(effect_idx, effect['id'])
                        )
                        
                        # Get method info for parameter controls
                        method_info = OPENCV_METHODS.get(effect['category'], {}).get('methods', {}).get(effect['method'], {})
                        params_config = method_info.get("params", {})
                        
                        if params_config:
                            st.markdown("**Parameters**")
                            for param_name, param_config in params_config.items():
                                param_type = param_config.get("type", "slider")
                                label = param_config.get("label", param_name)
                                current_value = effect['params'].get(param_name, param_config.get("default"))
                                param_key = f"edit_param_{effect['id']}_{idx}_{param_name}"
                                
                                if param_type == "slider":
                                    if isinstance(param_config.get("default"), float) or isinstance(param_config.get("step"), float):
                                        st.slider(
                                            label,
                                            min_value=float(param_config.get("min", 0)),
                                            max_value=float(param_config.get("max", 100)),
                                            value=float(current_value) if current_value is not None else float(param_config.get("default", 0)),
                                            step=float(param_config.get("step", 1)),
                                            key=param_key,
                                            on_change=on_param_change,
                                            args=(effect_idx, param_name, param_key)
                                        )
                                    else:
                                        st.slider(
                                            label,
                                            min_value=int(param_config.get("min", 0)),
                                            max_value=int(param_config.get("max", 100)),
                                            value=int(current_value) if current_value is not None else int(param_config.get("default", 0)),
                                            step=int(param_config.get("step", 1)),
                                            key=param_key,
                                            on_change=on_param_change,
                                            args=(effect_idx, param_name, param_key)
                                        )
                                
                                elif param_type == "dropdown":
                                    options = param_config.get("options", [])
                                    default = current_value if current_value in options else param_config.get("default", options[0] if options else "")
                                    st.selectbox(
                                        label,
                                        options,
                                        index=options.index(default) if default in options else 0,
                                        key=param_key,
                                        on_change=on_param_change,
                                        args=(effect_idx, param_name, param_key)
                                    )
                                
                                elif param_type == "checkbox":
                                    st.checkbox(
                                        label,
                                        value=current_value if current_value is not None else param_config.get("default", False),
                                        key=param_key,
                                        on_change=on_param_change,
                                        args=(effect_idx, param_name, param_key)
                                    )
                        else:
                            st.info("No adjustable parameters")
                
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Clear all button - uses callback
        st.button("🗑️ Clear All", key="clear_pipeline", on_click=clear_pipeline)
        
        # Export/Import Pipeline Settings
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**📤 Export/Import Pipeline**")
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            # Export pipeline as JSON
            pipeline_export = []
            for effect in st.session_state.effect_pipeline:
                pipeline_export.append({
                    'category': effect['category'],
                    'method': effect['method'],
                    'params': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                              for k, v in effect['params'].items()}
                })
            pipeline_json = json.dumps(pipeline_export, indent=2)
            st.download_button(
                "💾 Export",
                data=pipeline_json,
                file_name="pipeline_settings.json",
                mime="application/json",
                help="Export pipeline settings",
                key="export_pipeline_settings"
            )
        
        with exp_col2:
            # Import will be handled below
            st.caption("Import below ⬇️")
        
        # Import pipeline settings
        import_pipeline = st.file_uploader(
            "Import Pipeline",
            type=["json"],
            key="import_pipeline_settings",
            label_visibility="collapsed"
        )
        
        if import_pipeline is not None:
            try:
                imported_pipeline = json.load(import_pipeline)
                if st.button("✅ Apply Pipeline", type="primary", key="apply_imported_pipeline"):
                    st.session_state.effect_pipeline = []
                    for effect in imported_pipeline:
                        st.session_state.effect_pipeline.append({
                            'id': st.session_state.next_effect_id,
                            'category': effect['category'],
                            'method': effect['method'],
                            'params': effect['params']
                        })
                        st.session_state.next_effect_id += 1
                    st.success(f"Loaded {len(imported_pipeline)} effects!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.caption("No effects yet. Add effects above or import a pipeline!")
    
    # Import Pipeline Settings - ALWAYS VISIBLE
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("**📥 Import Pipeline Settings**")
    
    import_pipeline = st.file_uploader(
        "Import Pipeline JSON",
        type=["json"],
        key="import_pipeline_always",
        label_visibility="collapsed"
    )
    
    if import_pipeline is not None:
        try:
            imported_pipeline = json.load(import_pipeline)
            st.success(f"✅ Found {len(imported_pipeline)} effects")
            
            # Show preview of effects
            for i, eff in enumerate(imported_pipeline[:3]):
                st.caption(f"  {i+1}. {eff['method']}")
            if len(imported_pipeline) > 3:
                st.caption(f"  ... and {len(imported_pipeline) - 3} more")
            
            if st.button("✅ Apply Pipeline", type="primary", key="apply_pipeline_always"):
                st.session_state.effect_pipeline = []
                for effect in imported_pipeline:
                    st.session_state.effect_pipeline.append({
                        'id': st.session_state.next_effect_id,
                        'category': effect['category'],
                        'method': effect['method'],
                        'params': effect['params']
                    })
                    st.session_state.next_effect_id += 1
                st.success(f"Loaded {len(imported_pipeline)} effects!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Middle Section - Image Upload and Processing
with middle_col:
    st.markdown("### 📷 Image & Video Processing")
    
    # File uploader for both images and videos
    uploaded_file = st.file_uploader(
        "Upload an image or video",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "mp4", "avi", "mov", "mkv"],
        key="file_uploader"
    )
    
    # Check if new file uploaded
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_ext = file_name.lower().split('.')[-1]
        is_video = file_ext in ['mp4', 'avi', 'mov', 'mkv']
        
        # Check if this is a new file
        if st.session_state.uploaded_file_name != file_name:
            st.session_state.uploaded_file_name = file_name
            st.session_state.is_video = is_video
            st.session_state.use_as_input = False
            
            if is_video:
                # Load first frame of video
                st.session_state.video_file = uploaded_file
                frame, total_frames = load_video_frame(uploaded_file, 0)
                if frame is not None:
                    st.session_state.original_image = frame.copy()
                    st.session_state.current_image = frame.copy()
                    st.session_state.video_frame = 0
            else:
                # Load image
                new_image = load_image_from_file(uploaded_file)
                if new_image is not None:
                    st.session_state.original_image = new_image.copy()
                    st.session_state.current_image = new_image.copy()
    
    # Display content based on what we have
    if st.session_state.current_image is not None:
        # Video frame selector
        if st.session_state.is_video and uploaded_file is not None:
            video_info = get_video_info(uploaded_file)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea22, #764ba222); padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;">
                <strong>🎬 Video Info:</strong> {video_info['width']}x{video_info['height']} | {video_info['fps']:.1f} FPS | {video_info['duration']:.1f}s | {video_info['total_frames']} frames
            </div>
            """, unsafe_allow_html=True)
            
            frame_num = st.slider(
                "Select Frame",
                min_value=0,
                max_value=max(0, video_info['total_frames'] - 1),
                value=st.session_state.video_frame,
                key="frame_slider"
            )
            
            if frame_num != st.session_state.video_frame:
                st.session_state.video_frame = frame_num
                frame, _ = load_video_frame(uploaded_file, frame_num)
                if frame is not None:
                    st.session_state.current_image = frame.copy()
                    if not st.session_state.use_as_input:
                        st.session_state.original_image = frame.copy()
        
        # Apply the effect pipeline or show original
        if st.session_state.effect_pipeline:
            processed = apply_effect_pipeline(
                st.session_state.current_image,
                st.session_state.effect_pipeline
            )
            # Show pipeline summary
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #28a74522, #20c99722); padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #28a745;">
                <strong>🔗 Pipeline ({len(st.session_state.effect_pipeline)}):</strong> {get_pipeline_summary(st.session_state.effect_pipeline)}
            </div>
            """, unsafe_allow_html=True)
        else:
            processed = st.session_state.current_image.copy()
        
        # LIVE PREVIEW: Apply the currently selected effect on top of pipeline
        # Only apply preview if a method is selected
        if method_selected and selected_method:
            preview_with_current = apply_opencv_method(
                processed,
                selected_category,
                selected_method,
                param_values
            )
            
            # Show preview indicator
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffc10722, #ff980022); padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #ff9800;">
                <strong>👁️ Preview:</strong> {selected_method} <span style="font-size: 11px; color: #666;">(adjust sliders to see changes instantly)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            preview_with_current = processed
        
        st.session_state.processed_image = preview_with_current
        
        # View mode toggle
        view_col1, view_col2 = st.columns([3, 1])
        with view_col2:
            compare_mode = st.checkbox("🔀 Compare Slider", value=st.session_state.compare_mode, key="compare_toggle")
            st.session_state.compare_mode = compare_mode
        
        if st.session_state.compare_mode:
            # Compare slider view with interactive JS slider
            st.markdown("**🔀 Drag the slider on the image to compare**")
            
            # Convert images to base64 for HTML
            input_rgb = cv2.cvtColor(st.session_state.current_image, cv2.COLOR_BGR2RGB)
            if len(preview_with_current.shape) == 2:
                processed_display = cv2.cvtColor(preview_with_current, cv2.COLOR_GRAY2RGB)
            else:
                processed_display = cv2.cvtColor(preview_with_current, cv2.COLOR_BGR2RGB)
            
            # Encode images
            _, input_buffer = cv2.imencode('.jpg', cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, processed_buffer = cv2.imencode('.jpg', cv2.cvtColor(processed_display, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            input_b64 = base64.b64encode(input_buffer).decode()
            processed_b64 = base64.b64encode(processed_buffer).decode()
            
            # Get image dimensions for aspect ratio
            img_height, img_width = input_rgb.shape[:2]
            aspect_ratio = img_height / img_width
            container_height = int(600 * aspect_ratio)  # Assuming ~600px width
            
            # Interactive HTML/CSS/JS comparison slider using st.components
            import streamlit.components.v1 as components
            
            html_code = f'''
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
                    
                    .comparison-container {{
                        position: relative;
                        width: 100%;
                        max-width: 100%;
                        overflow: hidden;
                        border-radius: 12px;
                        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
                        cursor: ew-resize;
                        user-select: none;
                        -webkit-user-select: none;
                    }}
                    
                    .comparison-container img {{
                        width: 100%;
                        display: block;
                        pointer-events: none;
                    }}
                    
                    .img-overlay {{
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 50%;
                        height: 100%;
                        overflow: hidden;
                    }}
                    
                    .img-overlay img {{
                        position: absolute;
                        top: 0;
                        left: 0;
                        height: 100%;
                        width: auto;
                        max-width: none;
                    }}
                    
                    .slider-handle {{
                        position: absolute;
                        top: 0;
                        left: 50%;
                        width: 4px;
                        height: 100%;
                        background: linear-gradient(180deg, #667eea, #764ba2);
                        transform: translateX(-50%);
                        z-index: 10;
                        box-shadow: 0 0 10px rgba(0,0,0,0.3);
                    }}
                    
                    .slider-handle::before {{
                        content: '';
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        width: 44px;
                        height: 44px;
                        background: linear-gradient(135deg, #667eea, #764ba2);
                        border-radius: 50%;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    }}
                    
                    .slider-handle::after {{
                        content: '◀ ▶';
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        color: white;
                        font-size: 11px;
                        font-weight: bold;
                        white-space: nowrap;
                        z-index: 11;
                        letter-spacing: 2px;
                    }}
                    
                    .label {{
                        position: absolute;
                        padding: 6px 14px;
                        border-radius: 20px;
                        font-size: 12px;
                        font-weight: 600;
                        color: white;
                        z-index: 5;
                        top: 12px;
                    }}
                    
                    .label-left {{
                        left: 12px;
                        background: rgba(102, 126, 234, 0.9);
                    }}
                    
                    .label-right {{
                        right: 12px;
                        background: rgba(118, 75, 162, 0.9);
                    }}
                </style>
            </head>
            <body>
                <div class="comparison-container" id="container">
                    <img src="data:image/jpeg;base64,{processed_b64}" alt="Processed" id="bgImg">
                    
                    <div class="img-overlay" id="overlay">
                        <img src="data:image/jpeg;base64,{input_b64}" alt="Original" id="overlayImg">
                    </div>
                    
                    <div class="slider-handle" id="handle"></div>
                    
                    <span class="label label-left">Original</span>
                    <span class="label label-right">Processed</span>
                </div>
                
                <script>
                    const container = document.getElementById('container');
                    const overlay = document.getElementById('overlay');
                    const handle = document.getElementById('handle');
                    const overlayImg = document.getElementById('overlayImg');
                    const bgImg = document.getElementById('bgImg');
                    
                    let isDragging = false;
                    
                    function updateSlider(x) {{
                        const rect = container.getBoundingClientRect();
                        let percentage = ((x - rect.left) / rect.width) * 100;
                        percentage = Math.max(0, Math.min(100, percentage));
                        
                        overlay.style.width = percentage + '%';
                        handle.style.left = percentage + '%';
                        
                        // Scale the overlay image properly
                        if (percentage > 0) {{
                            overlayImg.style.width = (rect.width) + 'px';
                        }}
                    }}
                    
                    // Initialize overlay image width
                    bgImg.onload = function() {{
                        overlayImg.style.width = container.offsetWidth + 'px';
                    }};
                    
                    // Set initial state
                    setTimeout(function() {{
                        overlayImg.style.width = container.offsetWidth + 'px';
                    }}, 100);
                    
                    container.addEventListener('mousedown', function(e) {{
                        isDragging = true;
                        updateSlider(e.clientX);
                    }});
                    
                    document.addEventListener('mousemove', function(e) {{
                        if (isDragging) {{
                            updateSlider(e.clientX);
                        }}
                    }});
                    
                    document.addEventListener('mouseup', function() {{
                        isDragging = false;
                    }});
                    
                    container.addEventListener('touchstart', function(e) {{
                        isDragging = true;
                        updateSlider(e.touches[0].clientX);
                    }});
                    
                    document.addEventListener('touchmove', function(e) {{
                        if (isDragging) {{
                            updateSlider(e.touches[0].clientX);
                        }}
                    }});
                    
                    document.addEventListener('touchend', function() {{
                        isDragging = false;
                    }});
                    
                    // Click to move
                    container.addEventListener('click', function(e) {{
                        updateSlider(e.clientX);
                    }});
                </script>
            </body>
            </html>
            '''
            
            components.html(html_code, height=container_height + 20)
        else:
            # Display images side by side
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("**📥 Original / Input**")
                input_rgb = cv2.cvtColor(st.session_state.current_image, cv2.COLOR_BGR2RGB)
                st.image(input_rgb, use_column_width=True)
            
            with img_col2:
                st.markdown("**📤 Preview**")
                if len(preview_with_current.shape) == 2:
                    st.image(preview_with_current, use_column_width=True)
                else:
                    processed_rgb = cv2.cvtColor(preview_with_current, cv2.COLOR_BGR2RGB)
                    st.image(processed_rgb, use_column_width=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Action buttons
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        
        with btn_col1:
            if st.button("🔄 Use as Input", type="primary"):
                st.session_state.current_image = preview_with_current.copy()
                st.session_state.use_as_input = True
                # Auto-save when using as input
                pipeline_summary = get_pipeline_summary(st.session_state.effect_pipeline)
                save_to_history(preview_with_current, pipeline_summary, {}, st.session_state.effect_pipeline)
                st.success("Saved & set as input!")
        
        with btn_col2:
            if st.button("💾 Save Output"):
                pipeline_summary = get_pipeline_summary(st.session_state.effect_pipeline)
                filepath = save_to_history(preview_with_current, pipeline_summary, {}, st.session_state.effect_pipeline)
                st.success(f"Saved!")
        
        with btn_col3:
            if st.button("↩️ Reset to Original"):
                if st.session_state.original_image is not None:
                    st.session_state.current_image = st.session_state.original_image.copy()
                    st.session_state.use_as_input = False
        
        with btn_col4:
            # Download button
            if preview_with_current is not None:
                _, buffer = cv2.imencode('.png', preview_with_current)
                pipeline_name = get_pipeline_summary(st.session_state.effect_pipeline).replace(' → ', '_').replace(' ', '-')
                if not pipeline_name or pipeline_name == "No effects":
                    pipeline_name = selected_method.replace(' ', '_') if selected_method else "processed"
                st.download_button(
                    "⬇️ Download",
                    data=buffer.tobytes(),
                    file_name=f"{pipeline_name[:50]}.png",
                    mime="image/png"
                )
        
        # Show current processing info
        if st.session_state.use_as_input:
            st.info("🔗 Using processed output as input - add more effects to the pipeline!")
    else:
        # Empty state
        st.markdown("""
        <div class="upload-area">
            <h3>📤 Drop an image or video above</h3>
            <p>Supported formats: JPG, PNG, BMP, TIFF, WebP, MP4, AVI, MOV, MKV</p>
        </div>
        """, unsafe_allow_html=True)

# Right Sidebar - History
with right_col:
    st.markdown("### 📜 History")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("No processing history yet. Save some outputs to see them here!")
    else:
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Display history items
        for i, item in enumerate(st.session_state.history[:20]):  # Show last 20
            with st.container():
                # Load full image for hover preview
                if os.path.exists(item["filepath"]):
                    full_img = cv2.imread(item["filepath"])
                    if full_img is not None:
                        full_img_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
                        # Resize for preview (max 300px)
                        h, w = full_img_rgb.shape[:2]
                        max_preview = 300
                        if w > h:
                            preview_w = max_preview
                            preview_h = int(h * max_preview / w)
                        else:
                            preview_h = max_preview
                            preview_w = int(w * max_preview / h)
                        preview_img = cv2.resize(full_img_rgb, (preview_w, preview_h))
                        
                        # Encode for HTML
                        _, preview_buffer = cv2.imencode('.png', cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
                        preview_b64 = base64.b64encode(preview_buffer).decode()
                        
                        # Thumbnail with aspect ratio preserved
                        thumbnail_rgb = cv2.cvtColor(item["thumbnail"], cv2.COLOR_BGR2RGB)
                        _, thumb_buffer = cv2.imencode('.png', cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2BGR))
                        thumb_b64 = base64.b64encode(thumb_buffer).decode()
                        
                        # HTML with hover preview
                        st.markdown(f"""
                        <div class="thumbnail-tooltip" style="text-align: center; margin-bottom: 8px;">
                            <img src="data:image/png;base64,{thumb_b64}" style="max-width: 80px; max-height: 60px; border-radius: 6px; cursor: pointer; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div class="tooltip-image">
                                <img src="data:image/png;base64,{preview_b64}">
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Fallback to regular thumbnail
                    thumbnail_rgb = cv2.cvtColor(item["thumbnail"], cv2.COLOR_BGR2RGB)
                    st.image(thumbnail_rgb, width=80)
                
                # Info
                st.markdown(f"<p style='font-size: 12px; margin: 2px 0; font-weight: 600;'>{item['method']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 10px; color: #888; margin: 0;'>{item['timestamp'][:8]}</p>", unsafe_allow_html=True)
                
                # Show pipeline effects count
                pipeline = item.get('pipeline', [])
                if pipeline and len(pipeline) > 0:
                    st.markdown(f"<p style='font-size: 9px; color: #28a745; margin: 2px 0; background: #e8f5e9; padding: 2px 6px; border-radius: 4px; display: inline-block;'>🔗 {len(pipeline)} effect(s)</p>", unsafe_allow_html=True)
                
                # Show parameters if any (only for single effects)
                elif item.get('params') and len(item['params']) > 0:
                    params_str = " | ".join([f"{k}={v}" for k, v in item['params'].items()])
                    st.markdown(f"<p style='font-size: 9px; color: #667eea; margin: 2px 0; background: #f0f2ff; padding: 2px 6px; border-radius: 4px; display: inline-block;'>{params_str}</p>", unsafe_allow_html=True)
                
                # Actions - 3 columns for image, pipeline, download
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📥", key=f"load_{i}", help="Load this image"):
                        loaded_img = cv2.imread(item["filepath"])
                        if loaded_img is not None:
                            st.session_state.current_image = loaded_img.copy()
                            st.session_state.use_as_input = True
                
                with col2:
                    # Load pipeline button
                    if pipeline and len(pipeline) > 0:
                        if st.button("🔗", key=f"pipeline_{i}", help="Load this pipeline"):
                            # Restore the pipeline with new IDs
                            st.session_state.effect_pipeline = []
                            for effect in pipeline:
                                st.session_state.effect_pipeline.append({
                                    'id': st.session_state.next_effect_id,
                                    'category': effect['category'],
                                    'method': effect['method'],
                                    'params': effect['params']
                                })
                                st.session_state.next_effect_id += 1
                            st.success(f"Loaded {len(pipeline)} effects!")
                
                with col3:
                    if os.path.exists(item["filepath"]):
                        with open(item["filepath"], "rb") as f:
                            st.download_button(
                                "⬇️",
                                data=f.read(),
                                file_name=item["filename"],
                                mime="image/png",
                                key=f"download_{i}"
                            )
                
                st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid #eee;'>", unsafe_allow_html=True)

# Code Generation Panel - Only shown when toggled
if st.session_state.show_code_panel:
    with code_col:
        st.markdown("### 💻 Code Generator")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        if not st.session_state.effect_pipeline:
            st.info("Add effects to your pipeline to see the generated code!")
        else:
            # Show each effect's code
            st.markdown(f"**📝 Pipeline Code ({len(st.session_state.effect_pipeline)} effects)**")
            
            for i, effect in enumerate(st.session_state.effect_pipeline):
                code_info = generate_opencv_code(effect['method'], effect['params'], effect.get('category'))
                
                # Effect header
                st.markdown(f"""
                <div class="code-step">
                    <div class="code-step-header">
                        <span class="code-step-number">{i + 1}</span>
                        <span class="code-step-method">{effect['method']}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show parameters
                if code_info['param_info'] and code_info['param_info'][0]['params']:
                    params_display = code_info['param_info'][0]['params']
                    params_html = ", ".join([f"<span class='code-param'>{k}</span>=<span class='code-value'>{v}</span>" for k, v in params_display.items()])
                    st.markdown(f"""
                    <div style="font-size: 10px; color: #999; margin-bottom: 6px; padding-left: 28px;">
                        {params_html}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show code
                code_text = "\\n".join(code_info['code_lines'])
                st.markdown(f"""
                    <div class="code-block">{code_text}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # Full pipeline code section
            with st.expander("📄 Complete Python Script", expanded=False):
                full_code = generate_full_pipeline_code(st.session_state.effect_pipeline)
                st.code(full_code, language="python")
                
                # Download button for full code
                st.download_button(
                    "⬇️ Download Script",
                    data=full_code,
                    file_name="opencv_pipeline.py",
                    mime="text/plain",
                    key="download_full_code"
                )
            
            # Copy-friendly summary
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("**📋 Quick Reference**")
            
            # Create a summary table of all functions used
            functions_used = []
            for effect in st.session_state.effect_pipeline:
                code_info = generate_opencv_code(effect['method'], effect['params'], effect.get('category'))
                if code_info['param_info']:
                    func_name = code_info['param_info'][0].get('function', 'custom')
                    params = code_info['param_info'][0].get('params', {})
                    functions_used.append({
                        "Effect": effect['method'],
                        "Function": func_name,
                        "Key Params": ", ".join([f"{k}={v}" for k, v in list(params.items())[:3]])
                    })
            
            if functions_used:
                df = pd.DataFrame(functions_used)
                st.dataframe(df, use_container_width=True, hide_index=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>Built with ❤️ using Streamlit, OpenCV By Rajilesh Panoli</p>
    <p style="font-size: 0.8rem;">Stack multiple effects, reorder, and export your pipelines!</p>
</div>
""", unsafe_allow_html=True)
