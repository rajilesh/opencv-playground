"""
OpenCV Playground - Interactive Image Processing Application

This application uses a modular effect system following SOLID principles:
- Effects are loaded dynamically from the 'effects' folder
- Add new effects by creating new Python files in the effects folder
- Remove effects by deleting their files from the effects folder

The effect system uses:
- Single Responsibility: Each effect file handles one category
- Open/Closed: Add new effects without modifying existing code
- Liskov Substitution: All effects can be used interchangeably
- Interface Segregation: Effects implement only required methods
- Dependency Inversion: High-level modules depend on abstractions
"""

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
import streamlit.components.v1 as components

# Try to import streamlit-drawable-canvas for better mask drawing
try:
    from streamlit_javascript import st_javascript
    HAS_ST_JAVASCRIPT = True
except ImportError:
    HAS_ST_JAVASCRIPT = False
    print("Note: streamlit-javascript not installed.")
    print("Install with: pip install streamlit-javascript")

# Try to import streamlit-webrtc for live video processing
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False
    print("Note: streamlit-webrtc not installed.")
    print("Install with: pip install streamlit-webrtc av")

# We'll use our own HTML canvas approach - no need for broken st_canvas
HAS_DRAWABLE_CANVAS = True  # Always use our custom implementation

# Import the effects package - it automatically discovers all effects
from effects import get_opencv_methods_config, registry

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
    
    /* Pipeline row buttons - Streamlit specific */
    .pipeline-row .stButton > button {
        padding: 4px 8px !important;
        min-height: 32px !important;
        width: 32px !important;
        font-size: 14px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .pipeline-row [data-testid="column"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
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

# Get OPENCV_METHODS from the registry (dynamically loaded)
OPENCV_METHODS = get_opencv_methods_config()

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
if 'video_playing' not in st.session_state:
    st.session_state.video_playing = False
if 'video_total_frames' not in st.session_state:
    st.session_state.video_total_frames = 0
if 'compare_mode' not in st.session_state:
    st.session_state.compare_mode = False
if 'effect_pipeline' not in st.session_state:
    st.session_state.effect_pipeline = []
if 'next_effect_id' not in st.session_state:
    st.session_state.next_effect_id = 1
if 'pipeline_loaded_from_storage' not in st.session_state:
    st.session_state.pipeline_loaded_from_storage = False
if 'editing_effect_id' not in st.session_state:
    st.session_state.editing_effect_id = None
if 'selected_category_index' not in st.session_state:
    st.session_state.selected_category_index = 0
if 'selected_method_index' not in st.session_state:
    st.session_state.selected_method_index = 0
if 'show_code_panel' not in st.session_state:
    st.session_state.show_code_panel = False
if 'mask_data' not in st.session_state:
    st.session_state.mask_data = None
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = []
if 'interactive_mode' not in st.session_state:
    st.session_state.interactive_mode = None  # None, 'mask', or 'points'
if 'mask_binary' not in st.session_state:
    st.session_state.mask_binary = True
if 'mask_threshold' not in st.session_state:
    st.session_state.mask_threshold = 127
if 'image_source' not in st.session_state:
    st.session_state.image_source = "Upload File"
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'webcam_frame' not in st.session_state:
    st.session_state.webcam_frame = None
if 'image_url' not in st.session_state:
    st.session_state.image_url = ""

# Load pipeline from localStorage on first load
if HAS_ST_JAVASCRIPT:
    if 'pipeline_load_attempt' not in st.session_state:
        st.session_state.pipeline_load_attempt = 0
    
    if not st.session_state.pipeline_loaded_from_storage:
        try:
            saved_pipeline = st_javascript("localStorage.getItem('opencv_playground_pipeline')", key="load_pipeline_js")
            saved_next_id = st_javascript("localStorage.getItem('opencv_playground_next_id')", key="load_next_id_js")
            
            # st_javascript returns 0 on first render, need to wait for actual value
            if saved_pipeline == 0:
                # First render - increment attempt and will try again
                st.session_state.pipeline_load_attempt += 1
                if st.session_state.pipeline_load_attempt < 3:
                    # Don't set loaded flag yet, try again
                    pass
                else:
                    # After 3 attempts, assume no saved data
                    st.session_state.pipeline_loaded_from_storage = True
            elif saved_pipeline and saved_pipeline not in ("null", "undefined", None):
                try:
                    loaded_pipeline = json.loads(saved_pipeline)
                    if isinstance(loaded_pipeline, list) and len(loaded_pipeline) > 0:
                        st.session_state.effect_pipeline = loaded_pipeline
                        if saved_next_id and saved_next_id not in ("null", "undefined", None, 0):
                            st.session_state.next_effect_id = int(saved_next_id)
                        else:
                            # Calculate next_effect_id from loaded pipeline
                            max_id = max([e.get('id', 0) for e in loaded_pipeline], default=0)
                            st.session_state.next_effect_id = max_id + 1
                        st.toast(f"✅ Restored {len(loaded_pipeline)} effect(s) from saved pipeline")
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
                st.session_state.pipeline_loaded_from_storage = True
            else:
                # No saved data found
                st.session_state.pipeline_loaded_from_storage = True
        except Exception as e:
            st.session_state.pipeline_loaded_from_storage = True

# Create output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_pipeline_to_storage():
    """Save the current pipeline to localStorage via st_javascript"""
    if not HAS_ST_JAVASCRIPT:
        return
        
    if st.session_state.effect_pipeline:
        # Convert numpy types to Python types for JSON serialization
        pipeline_export = []
        for effect in st.session_state.effect_pipeline:
            pipeline_export.append({
                'id': effect['id'],
                'category': effect['category'],
                'method': effect['method'],
                'params': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else 
                              bool(v) if isinstance(v, np.bool_) else v) 
                          for k, v in effect['params'].items()}
            })
        pipeline_json = json.dumps(pipeline_export)
        next_id = st.session_state.next_effect_id
        
        # Escape the JSON string for JavaScript
        escaped_json = pipeline_json.replace('\\', '\\\\').replace("'", "\\'")
        
        # Use st_javascript for more reliable localStorage access with unique keys
        st_javascript(f"localStorage.setItem('opencv_playground_pipeline', '{escaped_json}')", key="save_pipeline_js")
        st_javascript(f"localStorage.setItem('opencv_playground_next_id', '{next_id}')", key="save_next_id_js")
    else:
        # Clear localStorage if pipeline is empty
        st_javascript("localStorage.removeItem('opencv_playground_pipeline')", key="clear_pipeline_js")
        st_javascript("localStorage.removeItem('opencv_playground_next_id')", key="clear_next_id_js")


def clear_saved_pipeline():
    """Clear saved pipeline from localStorage"""
    if HAS_ST_JAVASCRIPT:
        st_javascript("localStorage.removeItem('opencv_playground_pipeline')", key="manual_clear_pipeline_js")
        st_javascript("localStorage.removeItem('opencv_playground_next_id')", key="manual_clear_next_id_js")


def apply_opencv_method(image, category, method_name, params):
    """Apply the selected OpenCV method with given parameters using the registry"""
    try:
        # Add mask or points data if available in session state
        enhanced_params = params.copy()
        if st.session_state.mask_data is not None:
            mask = st.session_state.mask_data
            
            # Convert to grayscale if needed
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Convert to binary if option is enabled
            if st.session_state.get('mask_binary', True):
                threshold = st.session_state.get('mask_threshold', 127)
                _, mask = cv2.threshold(mask.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
            
            enhanced_params['_mask'] = mask.astype(np.uint8)
            
        if st.session_state.selected_points:
            enhanced_params['_points'] = st.session_state.selected_points
        return registry.apply_effect(image, category, method_name, enhanced_params)
    except Exception as e:
        st.error(f"Error applying {method_name}: {str(e)}")
        return image


def render_mask_canvas(image: np.ndarray, key: str = "mask_canvas", height: int = 400):
    """Render an interactive mask drawing canvas with automatic mask capture"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Mask options
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        brush_size = st.slider("🖌️ Brush Size", 5, 50, 20, key=f"{key}_brush")
    with col2:
        binary_threshold = st.slider("Binary Threshold", 1, 254, 127, key=f"{key}_threshold")
    with col3:
        convert_binary = st.checkbox("Convert to Binary", value=True, key=f"{key}_binary",
                                     help="Convert mask to binary (recommended for inpainting)")
    
    # Calculate canvas dimensions
    aspect_ratio = w / h
    canvas_width = min(700, int(height * aspect_ratio))
    canvas_height = int(canvas_width / aspect_ratio)
    
    # Resize image for display
    display_img = cv2.resize(img_rgb, (canvas_width, canvas_height))
    _, buffer = cv2.imencode('.png', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
    img_b64 = base64.b64encode(buffer).decode()
    
    stored_mask_key = f"mask_storage_{key}"
    
    st.markdown("🖌️ **Draw on the image below** (white strokes = area to inpaint):")
    
    # HTML Canvas with drawing functionality
    html_code = f'''
    <div id="canvas-container-{key}" style="position: relative; display: inline-block;">
        <img id="bgImage-{key}" src="data:image/png;base64,{img_b64}" style="display: block; border-radius: 8px;" />
        <canvas id="drawCanvas-{key}" style="position: absolute; top: 0; left: 0; cursor: crosshair; border-radius: 8px;"></canvas>
    </div>
    <div style="margin-top: 10px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
        <button onclick="clearCanvas_{key}()" style="padding: 8px 16px; border: none; border-radius: 6px; background: #f0f0f0; cursor: pointer; font-weight: 600;">🗑️ Clear</button>
        <button onclick="applyMask_{key}()" style="padding: 8px 16px; border: none; border-radius: 6px; background: linear-gradient(135deg, #28a745, #20c997); color: white; cursor: pointer; font-weight: 600;">✓ Apply Mask</button>
        <span id="status-{key}" style="font-size: 12px; color: #667eea; padding: 4px 12px; background: #f0f2ff; border-radius: 12px;">Ready to draw</span>
    </div>
    <script>
        (function() {{
            const img = document.getElementById('bgImage-{key}');
            const canvas = document.getElementById('drawCanvas-{key}');
            const ctx = canvas.getContext('2d');
            const status = document.getElementById('status-{key}');
            let isDrawing = false, lastX = 0, lastY = 0;
            const brushSize = {brush_size};
            const origW = {w}, origH = {h};
            const storageKey = '{stored_mask_key}';
            
            function initCanvas() {{
                canvas.width = img.width || {canvas_width};
                canvas.height = img.height || {canvas_height};
            }}
            
            img.onload = initCanvas;
            if (img.complete) initCanvas();
            
            function getPos(e) {{
                const rect = canvas.getBoundingClientRect();
                const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
                const y = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
                return [x, y];
            }}
            
            function startDraw(e) {{ 
                isDrawing = true; 
                [lastX, lastY] = getPos(e); 
                e.preventDefault();
            }}
            
            function draw(e) {{
                if (!isDrawing) return;
                e.preventDefault();
                const [x, y] = getPos(e);
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
                ctx.lineWidth = brushSize;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.beginPath();
                ctx.moveTo(lastX, lastY);
                ctx.lineTo(x, y);
                ctx.stroke();
                [lastX, lastY] = [x, y];
                status.textContent = 'Drawing...';
            }}
            
            function stopDraw() {{ 
                isDrawing = false; 
                status.textContent = 'Click "Apply Mask" when done';
            }}
            
            window.clearCanvas_{key} = function() {{ 
                ctx.clearRect(0, 0, canvas.width, canvas.height); 
                localStorage.removeItem(storageKey);
                status.textContent = 'Cleared';
            }};
            
            window.applyMask_{key} = function() {{
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = origW;
                tempCanvas.height = origH;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.fillStyle = 'black';
                tempCtx.fillRect(0, 0, origW, origH);
                tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, origW, origH);
                
                const maskData = tempCanvas.toDataURL('image/png').split(',')[1];
                localStorage.setItem(storageKey, maskData);
                
                // Also set a flag to indicate mask is ready
                localStorage.setItem(storageKey + '_ready', 'true');
                
                status.textContent = '✓ Mask applied! Click "Load Mask" below.';
                status.style.background = '#d4edda';
                status.style.color = '#155724';
            }};
            
            canvas.addEventListener('mousedown', startDraw);
            canvas.addEventListener('mousemove', draw);
            canvas.addEventListener('mouseup', stopDraw);
            canvas.addEventListener('mouseout', stopDraw);
            canvas.addEventListener('touchstart', startDraw);
            canvas.addEventListener('touchmove', draw);
            canvas.addEventListener('touchend', stopDraw);
        }})();
    </script>
    '''
    
    components.html(html_code, height=canvas_height + 70)
    
    # Use st_javascript to read mask from localStorage - must be called unconditionally
    if HAS_ST_JAVASCRIPT:
        # Always try to read the mask data from localStorage
        mask_data = st_javascript(f"localStorage.getItem('mask_storage_{key}')", key=f"{key}_mask_data_js")
        mask_ready = st_javascript(f"localStorage.getItem('mask_storage_{key}_ready')", key=f"{key}_mask_ready_js")
        
        # Show load button and process if data exists
        col1, col2 = st.columns([1, 2])
        with col1:
            load_clicked = st.button("🎯 Load Mask", key=f"{key}_load_mask", type="primary")
        with col2:
            if mask_ready == 'true':
                st.success("✓ Mask ready to load!")
            else:
                st.info("Draw on image, click 'Apply Mask', then 'Load Mask'")
        
        if load_clicked and mask_data and len(str(mask_data)) > 100:
            try:
                mask_bytes = base64.b64decode(mask_data)
                mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
                mask_img = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    if mask_img.shape[0] != h or mask_img.shape[1] != w:
                        mask_img = cv2.resize(mask_img, (w, h))
                    if convert_binary:
                        _, mask_img = cv2.threshold(mask_img, binary_threshold, 255, cv2.THRESH_BINARY)
                    st.session_state.mask_data = mask_img
                    
                    # Clear the ready flag
                    st_javascript(f"localStorage.removeItem('mask_storage_{key}_ready')")
                    st.success("✅ Mask loaded successfully!")
            except Exception as e:
                st.error(f"Error loading mask: {e}")
        elif load_clicked:
            st.warning("No mask found. Draw on the image and click 'Apply Mask' first.")
    else:
        # Fallback: file upload approach
        st.info("💡 Click 'Apply Mask' above, then upload the mask file below.")
        
        # Add a download button that gets mask from localStorage
        download_js = f'''
        <button onclick="downloadMask()" style="padding: 8px 16px; border: none; border-radius: 6px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; cursor: pointer; font-weight: 600;">
            ⬇️ Download Mask File
        </button>
        <script>
            function downloadMask() {{
                const maskData = localStorage.getItem('mask_storage_{key}');
                if (maskData && maskData.length > 100) {{
                    const blob = new Blob([maskData], {{type: 'text/plain'}});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'mask_data.txt';
                    a.click();
                    URL.revokeObjectURL(url);
                }} else {{
                    alert('No mask drawn yet! Draw on the image first.');
                }}
            }}
        </script>
        '''
        components.html(download_js, height=50)
        
        mask_file = st.file_uploader(
            "📤 Upload mask_data.txt:",
            type=["txt"],
            key=f"{key}_mask_upload"
        )
        
        if mask_file is not None:
            try:
                mask_b64 = mask_file.read().decode('utf-8')
                if len(mask_b64) > 100:
                    mask_bytes = base64.b64decode(mask_b64)
                    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
                    mask_img = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        if mask_img.shape[0] != h or mask_img.shape[1] != w:
                            mask_img = cv2.resize(mask_img, (w, h))
                        if convert_binary:
                            _, mask_img = cv2.threshold(mask_img, binary_threshold, 255, cv2.THRESH_BINARY)
                        st.session_state.mask_data = mask_img
                        st.success("✅ Mask loaded!")
                        return mask_img
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Show current mask if exists
    if 'mask_data' in st.session_state and st.session_state.mask_data is not None:
        st.markdown("**Current Mask:**")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.mask_data, caption="Mask Preview", width=200)
        with col2:
            white_pixels = np.sum(st.session_state.mask_data > 127)
            percentage = (white_pixels / st.session_state.mask_data.size) * 100
            st.metric("Mask Coverage", f"{percentage:.1f}%")
        return st.session_state.mask_data
    
    return None


def render_point_selector(image: np.ndarray, num_points: int, point_labels: list = None, 
                          key: str = "point_canvas", height: int = 400):
    """Render an interactive point selection canvas"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    aspect_ratio = w / h
    canvas_width = min(int(height * aspect_ratio), 700)
    canvas_height = int(canvas_width / aspect_ratio)
    
    display_img = cv2.resize(img_rgb, (canvas_width, canvas_height))
    _, buffer = cv2.imencode('.png', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
    img_b64 = base64.b64encode(buffer).decode()
    
    scale_x = w / canvas_width
    scale_y = h / canvas_height
    
    if point_labels is None:
        point_labels = [f"Point {i+1}" for i in range(num_points)]
    
    labels_js = str(point_labels).replace("'", '"')
    colors = ["#667eea", "#28a745", "#ffc107", "#dc3545", "#17a2b8", "#6c757d", "#e83e8c", "#fd7e14"]
    
    html_code = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: transparent; }}
            .container {{ position: relative; display: inline-block; }}
            canvas {{ border-radius: 8px; cursor: crosshair; }}
            .controls {{ display: flex; gap: 10px; margin-bottom: 10px; align-items: center; flex-wrap: wrap; }}
            .btn {{ padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; transition: all 0.2s; }}
            .btn-primary {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; }}
            .btn-secondary {{ background: #f0f0f0; color: #333; }}
            .btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }}
            .status {{ font-size: 12px; color: #667eea; padding: 4px 12px; background: #f0f2ff; border-radius: 12px; }}
            .info {{ font-size: 11px; color: #666; margin-bottom: 8px; padding: 8px 12px; background: #f8f9ff; border-radius: 6px; border-left: 3px solid #667eea; }}
            .point-list {{ margin-top: 8px; font-size: 11px; max-height: 60px; overflow-y: auto; }}
            .point-item {{ display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 12px; font-weight: 500; }}
        </style>
    </head>
    <body>
        <div class="info">📍 Click to select {num_points} points. Currently: <strong id="currentPoint">{point_labels[0] if point_labels else 'Point 1'}</strong></div>
        <div class="controls">
            <button class="btn btn-secondary" onclick="clearPoints()">🗑️ Clear</button>
            <button class="btn btn-secondary" onclick="undoPoint()">↩️ Undo</button>
            <button class="btn btn-primary" onclick="savePoints()">✓ Apply Points</button>
            <span class="status" id="status">0/{num_points} points</span>
        </div>
        <div class="container">
            <canvas id="canvas" width="{canvas_width}" height="{canvas_height}"></canvas>
        </div>
        <div class="point-list" id="pointList"></div>
        <script>
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const status = document.getElementById('status');
            const currentPoint = document.getElementById('currentPoint');
            const pointList = document.getElementById('pointList');
            const labels = {labels_js};
            const colors = {str(colors)};
            const numPoints = {num_points};
            const scaleX = {scale_x};
            const scaleY = {scale_y};
            let points = [];
            
            const img = new Image();
            img.onload = function() {{ ctx.drawImage(img, 0, 0, {canvas_width}, {canvas_height}); }};
            img.src = 'data:image/png;base64,{img_b64}';
            
            function getColor(i) {{ return colors[i % colors.length]; }}
            
            function redraw() {{
                ctx.clearRect(0, 0, {canvas_width}, {canvas_height});
                ctx.drawImage(img, 0, 0, {canvas_width}, {canvas_height});
                points.forEach((p, i) => {{
                    const color = getColor(i);
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 8, 0, 2 * Math.PI);
                    ctx.fillStyle = color;
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 10px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(i + 1, p.x, p.y);
                    ctx.fillStyle = color;
                    ctx.font = '10px Arial';
                    ctx.textAlign = 'left';
                    ctx.fillText(labels[i] || ('Point ' + (i+1)), p.x + 12, p.y + 3);
                }});
                if (points.length > 1) {{
                    ctx.beginPath();
                    ctx.moveTo(points[0].x, points[0].y);
                    for (let i = 1; i < Math.min(points.length, numPoints/2); i++) ctx.lineTo(points[i].x, points[i].y);
                    if (points.length >= numPoints/2) {{ ctx.lineTo(points[0].x, points[0].y); }}
                    ctx.strokeStyle = 'rgba(102, 126, 234, 0.5)';
                    ctx.lineWidth = 1;
                    ctx.setLineDash([5, 5]);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }}
                updateStatus();
            }}
            
            function updateStatus() {{
                status.textContent = points.length + '/' + numPoints + ' points';
                currentPoint.textContent = points.length < numPoints ? (labels[points.length] || ('Point ' + (points.length + 1))) : 'All selected!';
                pointList.innerHTML = points.map((p, i) => {{
                    const origX = Math.round(p.x * scaleX);
                    const origY = Math.round(p.y * scaleY);
                    return '<span class="point-item" style="background:' + getColor(i) + '22; color:' + getColor(i) + ';">' + (labels[i] || ('P' + (i+1))) + ': (' + origX + ', ' + origY + ')</span>';
                }}).join('');
            }}
            
            canvas.addEventListener('click', function(e) {{
                if (points.length >= numPoints) {{ status.textContent = 'Max points! Clear to restart.'; return; }}
                const rect = canvas.getBoundingClientRect();
                points.push({{ x: e.clientX - rect.left, y: e.clientY - rect.top }});
                redraw();
            }});
            
            function clearPoints() {{ points = []; redraw(); }}
            function undoPoint() {{ if (points.length > 0) {{ points.pop(); redraw(); }} }}
            function savePoints() {{
                const origPoints = points.map(p => [Math.round(p.x * scaleX), Math.round(p.y * scaleY)]);
                window.parent.postMessage({{ type: 'streamlit:setComponentValue', value: JSON.stringify(origPoints) }}, '*');
                status.textContent = '✓ Points applied!';
            }}
            
            updateStatus();
        </script>
    </body>
    </html>
    '''
    
    result = components.html(html_code, height=canvas_height + 120)
    return result


def apply_effect_pipeline(image, pipeline):
    """Apply a full pipeline of effects to the image, tracking intermediate results"""
    result = image.copy()
    # Store intermediate results for pipeline mask operations
    # Key 0 = original image, Key 1 = step 1 result, etc.
    pipeline_results = {0: image.copy()}
    
    for i, effect in enumerate(pipeline):
        # Add pipeline results to params for effects that need them
        enhanced_params = effect['params'].copy()
        enhanced_params['_pipeline_results'] = pipeline_results
        
        result = apply_opencv_method(
            result,
            effect['category'],
            effect['method'],
            enhanced_params
        )
        # Store this step's result (1-indexed for user clarity)
        pipeline_results[i + 1] = result.copy()
    
    return result


def get_pipeline_summary(pipeline):
    """Get a readable summary of the effect pipeline"""
    if not pipeline:
        return "No effects"
    return " → ".join([e['method'] for e in pipeline])


def generate_opencv_code(method_name, params, category=None):
    """Generate OpenCV Python code for a given method and parameters"""
    return registry.generate_code(category, method_name, params)


def generate_full_pipeline_code(pipeline):
    """Generate complete Python code for the entire pipeline"""
    if not pipeline:
        return "# No effects in pipeline"
    
    code = """import cv2
import numpy as np

# Load your image
img = cv2.imread('your_image.jpg')

# Store step results for reuse (masks, intermediate images, etc.)
step_0_result = img.copy()  # Original image

"""
    
    for i, effect in enumerate(pipeline):
        step_num = i + 1
        code += f"# Step {step_num}: {effect['method']}\n"
        code_info = generate_opencv_code(effect['method'], effect['params'], effect.get('category'))
        
        for line in code_info['code_lines']:
            if i > 0:
                line = line.replace('(img,', '(result,').replace('(img)', '(result)').replace(' img.', ' result.').replace(' img,', ' result,').replace('= img.', '= result.')
                if line.strip().startswith('img.'):
                    line = line.replace('img.', 'result.')
            code += line + "\n"
        
        # Store this step's result for later reuse (e.g., as masks for bitwise operations)
        code += f"step_{step_num}_result = result.copy()  # Store step {step_num} result for reuse\n"
        
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
    
    if pipeline and len(pipeline) > 0:
        method_str = "_".join([e['method'].replace(' ', '-') for e in pipeline[:3]])
        if len(pipeline) > 3:
            method_str += f"_+{len(pipeline)-3}more"
        filename = f"{timestamp}_{method_str}.png"
    else:
        filename = f"{timestamp}_{method_name.replace(' ', '_')}.png"
    
    filepath = OUTPUT_DIR / filename
    cv2.imwrite(str(filepath), image)
    
    h, w = image.shape[:2]
    max_thumb_size = 80
    if w > h:
        new_w = max_thumb_size
        new_h = int(h * max_thumb_size / w)
    else:
        new_h = max_thumb_size
        new_w = int(w * max_thumb_size / h)
    thumbnail = cv2.resize(image, (new_w, new_h))
    
    serializable_pipeline = []
    if pipeline:
        for effect in pipeline:
            serializable_pipeline.append({
                'category': effect['category'],
                'method': effect['method'],
                'params': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                          for k, v in effect['params'].items()}
            })
    
    history_item = {
        "timestamp": timestamp,
        "method": method_name if not pipeline else get_pipeline_summary(pipeline),
        "params": params,
        "pipeline": serializable_pipeline,
        "filepath": str(filepath),
        "filename": filename,
        "thumbnail": thumbnail
    }
    st.session_state.history.insert(0, history_item)
    
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[:50]
    
    return filepath


def load_image_from_file(uploaded_file):
    """Load image from uploaded file"""
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)
    return image


def get_temp_video_path(uploaded_file):
    """Get or create a cached temp file path for the video"""
    # Use session state to cache the temp file path
    cache_key = f"video_temp_path_{uploaded_file.name}"
    
    if cache_key not in st.session_state or st.session_state[cache_key] is None:
        # Create new temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        uploaded_file.seek(0)
        st.session_state[cache_key] = tfile.name
    
    return st.session_state[cache_key]


def load_video_frame(uploaded_file, frame_number=0):
    """Load a specific frame from video file"""
    temp_path = get_temp_video_path(uploaded_file)
    
    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_number >= total_frames:
        frame_number = total_frames - 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame, total_frames
    return None, total_frames


def get_video_info(uploaded_file):
    """Get video information"""
    temp_path = get_temp_video_path(uploaded_file)
    
    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': total_frames / fps if fps > 0 else 0
    }


def image_to_base64(image):
    """Convert image to base64 for display"""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode()


# ============ Callback Functions for Pipeline Actions ============

def add_effect_callback():
    """Callback for adding effect to pipeline"""
    cat_idx = st.session_state.get('category_select', 0)
    if cat_idx > 0:
        categories = list(OPENCV_METHODS.keys())
        selected_cat = categories[cat_idx - 1]
        method_idx = st.session_state.get('method_select', 0)
        if method_idx > 0:
            methods = list(OPENCV_METHODS[selected_cat]["methods"].keys())
            selected_meth = methods[method_idx - 1]
            
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
    st.session_state.clear_storage_flag = True


# Main App Layout
st.markdown("""
<div class="main-header">
    <h1>🎨 OpenCV Playground</h1>
    <p>Explore and apply OpenCV image processing methods in real-time</p>
</div>
""", unsafe_allow_html=True)

# Show loaded effects info
# with st.expander("ℹ️ Loaded Effects", expanded=False):
#     st.write(f"**{len(OPENCV_METHODS)} effect categories loaded from the effects folder:**")
#     for cat_name, cat_data in OPENCV_METHODS.items():
#         methods_count = len(cat_data.get('methods', {}))
#         st.write(f"- {cat_data.get('icon', '🔧')} **{cat_name}**: {methods_count} effects")
#     st.info("💡 To add new effects, create a Python file in the `effects/` folder. To remove, delete the file.")

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
    
    categories = list(OPENCV_METHODS.keys())
    category_options = ["-- Select a category --"] + [f"{OPENCV_METHODS[cat]['icon']} {cat}" for cat in categories]
    
    selected_category_index = st.selectbox(
        "Select Category",
        range(len(category_options)),
        format_func=lambda x: category_options[x],
        index=st.session_state.selected_category_index if st.session_state.selected_category_index < len(category_options) else 0,
        key="category_select"
    )
    
    st.session_state.selected_category_index = selected_category_index
    
    category_selected = selected_category_index > 0
    selected_category = categories[selected_category_index - 1] if category_selected else None
    
    param_values = {}
    method_selected = False
    selected_method = None
    
    if category_selected and selected_category:
        methods = list(OPENCV_METHODS[selected_category]["methods"].keys())
        method_options = ["-- Select an effect --"] + methods
        
        selected_method_index = st.selectbox(
            "Select Method",
            range(len(method_options)),
            format_func=lambda x: method_options[x],
            index=st.session_state.selected_method_index if st.session_state.selected_method_index < len(method_options) else 0,
            key="method_select"
        )
        
        st.session_state.selected_method_index = selected_method_index
        
        method_selected = selected_method_index > 0
        selected_method = methods[selected_method_index - 1] if method_selected else None
    else:
        st.session_state.selected_method_index = 0
    
    if method_selected and selected_method:
        method_info = OPENCV_METHODS[selected_category]["methods"][selected_method]
        
        st.markdown(f"""
        <div class="method-card">
            <strong>📋 {selected_method}</strong><br>
            <span style="font-size: 12px;">{method_info['description']}</span>
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # Check if effect requires mask or points
        requires_mask = registry.effect_requires_mask(selected_category)
        requires_points = registry.effect_requires_points(selected_category)
        
        if requires_mask:
            st.markdown("""
            <div style="background: #fff3cd; padding: 8px 12px; border-radius: 6px; margin: 8px 0; border-left: 3px solid #ffc107;">
                🖌️ <strong>Mask Required:</strong> Draw on the image in the preview to create a mask.
            </div>
            """, unsafe_allow_html=True)
        
        if requires_points:
            num_pts = registry.get_num_points_required(selected_category)
            st.markdown(f"""
            <div style="background: #d4edda; padding: 8px 12px; border-radius: 6px; margin: 8px 0; border-left: 3px solid #28a745;">
                📍 <strong>Point Selection Required:</strong> Select {num_pts} points on the image in the preview.
            </div>
            """, unsafe_allow_html=True)
        
        st.button("➕ Add to Pipeline", type="primary", key="add_effect_btn", on_click=add_effect_callback)
    else:
        st.info("👆 Select a category and effect from the dropdowns above")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Display current pipeline
    st.markdown("### 🔗 Effect Pipeline")
    if HAS_ST_JAVASCRIPT:
        st.caption("💾 Auto-saved to browser")
    
    if st.session_state.effect_pipeline:
        st.caption(f"{len(st.session_state.effect_pipeline)} effect(s)")
        
        for idx, effect in enumerate(st.session_state.effect_pipeline):
            category_icon = OPENCV_METHODS.get(effect['category'], {}).get('icon', '🔧')
            is_editing = st.session_state.editing_effect_id == effect['id']
            unique_key = f"{effect['id']}_{idx}"
            
            with st.container():
                # Apply pipeline-row class for styling
                st.markdown('<div class="pipeline-row">', unsafe_allow_html=True)
                
                # Single row: number + name, then action buttons
                cols = st.columns([4, 1, 1, 1, 1])
                
                with cols[0]:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; height: 32px; padding-top: 4px;">
                        <span class="pipeline-number">{idx + 1}</span>
                        <span style="font-weight: 500; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{category_icon} {effect['method']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    if idx > 0:
                        st.button("↑", key=f"up_{unique_key}", help="Move up", 
                                  on_click=move_effect_up, args=(idx,))
                
                with cols[2]:
                    if idx < len(st.session_state.effect_pipeline) - 1:
                        st.button("↓", key=f"down_{unique_key}", help="Move down",
                                  on_click=move_effect_down, args=(idx,))
                
                with cols[3]:
                    edit_icon = "✓" if is_editing else "✎"
                    st.button(edit_icon, key=f"edit_{unique_key}", help="Edit",
                              on_click=toggle_edit_effect, args=(effect['id'],))
                
                with cols[4]:
                    st.button("×", key=f"remove_{unique_key}", help="Remove",
                              on_click=remove_effect, args=(idx, effect['id']))
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if is_editing:
                    with st.expander("Edit Effect", expanded=True):
                        effect_idx = idx
                        
                        edit_categories = list(OPENCV_METHODS.keys())
                        edit_category_options = [f"{OPENCV_METHODS[cat]['icon']} {cat}" for cat in edit_categories]
                        current_cat_idx = edit_categories.index(effect['category']) if effect['category'] in edit_categories else 0
                        
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
                                first_method = list(OPENCV_METHODS[new_cat]["methods"].keys())[0]
                                st.session_state[method_state] = first_method
                                st.session_state.effect_pipeline[eff_idx]['category'] = new_cat
                                st.session_state.effect_pipeline[eff_idx]['method'] = first_method
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
                        
                        current_category = st.session_state.get(cat_state_key, effect['category'])
                        
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
        
        # Pipeline is saved to localStorage at the end of the page
        
        clear_col1, clear_col2 = st.columns(2)
        with clear_col1:
            st.button("🗑️ Clear All", key="clear_pipeline", on_click=clear_pipeline)
        with clear_col2:
            if st.button("🧹 Clear Saved", key="clear_saved", help="Clear saved pipeline from browser storage"):
                clear_saved_pipeline()
                st.success("Saved pipeline cleared!")
        
        # Handle clear storage flag
        if st.session_state.get('clear_storage_flag', False):
            clear_saved_pipeline()
            st.session_state.clear_storage_flag = False
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**📤 Export/Import Pipeline**")
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
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
            st.caption("Import below ⬇️")
        
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
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.caption("No effects yet. Add effects above or import a pipeline!")
        
        # Show option to clear saved data if no current pipeline
        if HAS_ST_JAVASCRIPT:
            st.caption("💡 Tip: Your pipeline is auto-saved to browser storage")
            if st.button("🧹 Clear Saved Data", key="clear_saved_empty", help="Clear any saved pipeline from browser storage"):
                clear_saved_pipeline()
                st.success("Saved pipeline cleared!")
    
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
    
    # Image source selector
    source_options = ["Upload File", "Webcam", "URL", "Sample Images", "Solid Color", "Gradient", "Noise Pattern"]
    image_source = st.selectbox(
        "📥 Select Image Source",
        source_options,
        index=source_options.index(st.session_state.image_source) if st.session_state.image_source in source_options else 0,
        key="source_selector"
    )
    st.session_state.image_source = image_source
    
    uploaded_file = None
    
    if image_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload an image or video",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "mp4", "avi", "mov", "mkv"],
            key="file_uploader"
        )
    
    elif image_source == "Webcam":
        st.markdown("**📹 Webcam Capture**")
        
        webcam_mode = st.radio(
            "Webcam Mode",
            ["📸 Single Photo", "🎬 Live Video (with effects)"],
            horizontal=True,
            key="webcam_mode_selector"
        )
        
        if webcam_mode == "📸 Single Photo":
            st.info("📷 Take a photo using your camera. Click the camera button below to capture.")
            
            # Use Streamlit's built-in camera input
            camera_photo = st.camera_input("Take a photo", key="webcam_camera_input")
            
            if camera_photo is not None:
                # Convert the captured image to OpenCV format
                file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                camera_photo.seek(0)  # Reset file pointer
                
                if frame is not None:
                    st.session_state.webcam_frame = frame.copy()
                    st.session_state.original_image = frame.copy()
                    st.session_state.current_image = frame.copy()
                    st.session_state.uploaded_file_name = f"webcam_capture_{datetime.now().strftime('%H%M%S')}"
                    st.session_state.is_video = False
                    st.success(f"✅ Photo captured! Size: {frame.shape[1]}x{frame.shape[0]}")
        
        else:  # Live Video mode
            if HAS_WEBRTC:
                st.info("🎬 Live video with real-time effect processing. Add effects in the sidebar to see them applied live!")
                
                # Create a video processor class that applies the current pipeline
                class OpenCVVideoProcessor(VideoProcessorBase):
                    def __init__(self):
                        self.pipeline = []
                    
                    def recv(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        
                        # Apply each effect in the pipeline
                        try:
                            pipeline_results = {0: img.copy()}
                            current_img = img.copy()
                            
                            for i, effect in enumerate(self.pipeline):
                                category = effect.get('category', '')
                                method = effect.get('method', '')
                                params = effect.get('params', {}).copy()
                                params['_pipeline_results'] = pipeline_results
                                
                                # Get the effect handler
                                effect_obj = registry.get_effect(category)
                                if effect_obj:
                                    current_img = effect_obj.apply(current_img, method, params)
                                    pipeline_results[i + 1] = current_img.copy()
                            
                            img = current_img
                        except Exception as e:
                            # If there's an error, just return the original frame
                            pass
                        
                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                
                # RTC Configuration for STUN servers
                rtc_config = RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                )
                
                # Create the webrtc streamer
                ctx = webrtc_streamer(
                    key="opencv-live-video",
                    video_processor_factory=OpenCVVideoProcessor,
                    rtc_configuration=rtc_config,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                
                # Update the processor's pipeline when it's running
                if ctx.video_processor:
                    ctx.video_processor.pipeline = st.session_state.effect_pipeline
                
                st.caption("💡 Tip: Add effects in the left sidebar and they'll be applied to the live video in real-time!")
                
                # Add a capture button to grab a frame from the live video
                if st.button("📸 Capture Current Frame", type="primary"):
                    if ctx.video_processor and hasattr(ctx, 'video_receiver'):
                        st.info("Frame capture from live video requires stopping the stream first.")
            else:
                st.warning("⚠️ Live video requires `streamlit-webrtc`. Install it with:")
                st.code("pip install streamlit-webrtc av", language="bash")
                st.info("Falling back to single photo mode...")
                
                # Fallback to camera input
                camera_photo = st.camera_input("Take a photo", key="webcam_camera_input_fallback")
                
                if camera_photo is not None:
                    file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    camera_photo.seek(0)
                    
                    if frame is not None:
                        st.session_state.webcam_frame = frame.copy()
                        st.session_state.original_image = frame.copy()
                        st.session_state.current_image = frame.copy()
                        st.session_state.uploaded_file_name = f"webcam_capture_{datetime.now().strftime('%H%M%S')}"
                        st.session_state.is_video = False
                        st.success(f"✅ Photo captured! Size: {frame.shape[1]}x{frame.shape[0]}")
    
    elif image_source == "URL":
        st.markdown("**🔗 Load from URL**")
        image_url = st.text_input("Enter image URL", value=st.session_state.image_url, placeholder="https://example.com/image.jpg")
        st.session_state.image_url = image_url
        
        if st.button("🔽 Load Image", type="primary") and image_url:
            try:
                import urllib.request
                with urllib.request.urlopen(image_url, timeout=10) as response:
                    image_data = response.read()
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        st.session_state.original_image = img.copy()
                        st.session_state.current_image = img.copy()
                        st.session_state.uploaded_file_name = f"url_image_{datetime.now().strftime('%H%M%S')}"
                        st.session_state.is_video = False
                        st.success("✅ Image loaded from URL!")
                    else:
                        st.error("Could not decode image from URL")
            except Exception as e:
                st.error(f"URL error: {str(e)}")
    
    elif image_source == "Sample Images":
        st.markdown("**🖼️ Sample Images**")
        sample_options = {
            "Lena (Classic)": "lena",
            "Baboon": "baboon",
            "Peppers": "peppers",
            "Cameraman": "cameraman",
            "Gradient Test": "gradient",
            "Color Bars": "colorbars",
            "Checkerboard": "checkerboard"
        }
        sample_choice = st.selectbox("Select sample image", list(sample_options.keys()))
        
        if st.button("📥 Load Sample", type="primary"):
            sample_type = sample_options[sample_choice]
            h, w = 512, 512
            
            if sample_type == "gradient":
                img = np.zeros((h, w, 3), dtype=np.uint8)
                for i in range(w):
                    img[:, i] = [int(255 * i / w)] * 3
            elif sample_type == "colorbars":
                img = np.zeros((h, w, 3), dtype=np.uint8)
                colors = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0)]
                bar_width = w // len(colors)
                for i, color in enumerate(colors):
                    img[:, i*bar_width:(i+1)*bar_width] = color[::-1]  # RGB to BGR
            elif sample_type == "checkerboard":
                img = np.zeros((h, w, 3), dtype=np.uint8)
                block_size = 32
                for y in range(0, h, block_size):
                    for x in range(0, w, block_size):
                        if (x // block_size + y // block_size) % 2 == 0:
                            img[y:y+block_size, x:x+block_size] = [255, 255, 255]
            else:
                # Generate synthetic versions of classic test images
                img = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
                cv2.putText(img, sample_choice, (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            st.session_state.original_image = img.copy()
            st.session_state.current_image = img.copy()
            st.session_state.uploaded_file_name = f"sample_{sample_type}"
            st.session_state.is_video = False
            st.success(f"✅ Loaded {sample_choice}!")
    
    elif image_source == "Solid Color":
        st.markdown("**🎨 Generate Solid Color Image**")
        color_cols = st.columns([1, 1, 1])
        with color_cols[0]:
            color_r = st.slider("Red", 0, 255, 128)
        with color_cols[1]:
            color_g = st.slider("Green", 0, 255, 128)
        with color_cols[2]:
            color_b = st.slider("Blue", 0, 255, 128)
        
        size_cols = st.columns(2)
        with size_cols[0]:
            img_width = st.number_input("Width", 64, 4096, 512)
        with size_cols[1]:
            img_height = st.number_input("Height", 64, 4096, 512)
        
        if st.button("🎨 Generate", type="primary"):
            img = np.full((img_height, img_width, 3), [color_b, color_g, color_r], dtype=np.uint8)
            st.session_state.original_image = img.copy()
            st.session_state.current_image = img.copy()
            st.session_state.uploaded_file_name = f"solid_color_{color_r}_{color_g}_{color_b}"
            st.session_state.is_video = False
            st.success("✅ Solid color image generated!")
    
    elif image_source == "Gradient":
        st.markdown("**🌈 Generate Gradient Image**")
        gradient_type = st.selectbox("Gradient Type", ["Horizontal", "Vertical", "Diagonal", "Radial"])
        
        grad_cols = st.columns(2)
        with grad_cols[0]:
            start_color = st.color_picker("Start Color", "#000000")
        with grad_cols[1]:
            end_color = st.color_picker("End Color", "#FFFFFF")
        
        size_cols = st.columns(2)
        with size_cols[0]:
            img_width = st.number_input("Width", 64, 4096, 512, key="grad_w")
        with size_cols[1]:
            img_height = st.number_input("Height", 64, 4096, 512, key="grad_h")
        
        if st.button("🌈 Generate Gradient", type="primary"):
            # Parse colors
            start_rgb = tuple(int(start_color[i:i+2], 16) for i in (1, 3, 5))
            end_rgb = tuple(int(end_color[i:i+2], 16) for i in (1, 3, 5))
            
            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            
            if gradient_type == "Horizontal":
                for x in range(img_width):
                    t = x / max(1, img_width - 1)
                    color = [int(start_rgb[2] + t * (end_rgb[2] - start_rgb[2])),
                             int(start_rgb[1] + t * (end_rgb[1] - start_rgb[1])),
                             int(start_rgb[0] + t * (end_rgb[0] - start_rgb[0]))]
                    img[:, x] = color
            elif gradient_type == "Vertical":
                for y in range(img_height):
                    t = y / max(1, img_height - 1)
                    color = [int(start_rgb[2] + t * (end_rgb[2] - start_rgb[2])),
                             int(start_rgb[1] + t * (end_rgb[1] - start_rgb[1])),
                             int(start_rgb[0] + t * (end_rgb[0] - start_rgb[0]))]
                    img[y, :] = color
            elif gradient_type == "Diagonal":
                for y in range(img_height):
                    for x in range(img_width):
                        t = (x + y) / max(1, img_width + img_height - 2)
                        color = [int(start_rgb[2] + t * (end_rgb[2] - start_rgb[2])),
                                 int(start_rgb[1] + t * (end_rgb[1] - start_rgb[1])),
                                 int(start_rgb[0] + t * (end_rgb[0] - start_rgb[0]))]
                        img[y, x] = color
            else:  # Radial
                cx, cy = img_width // 2, img_height // 2
                max_dist = np.sqrt(cx**2 + cy**2)
                for y in range(img_height):
                    for x in range(img_width):
                        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                        t = min(1.0, dist / max_dist)
                        color = [int(start_rgb[2] + t * (end_rgb[2] - start_rgb[2])),
                                 int(start_rgb[1] + t * (end_rgb[1] - start_rgb[1])),
                                 int(start_rgb[0] + t * (end_rgb[0] - start_rgb[0]))]
                        img[y, x] = color
            
            st.session_state.original_image = img.copy()
            st.session_state.current_image = img.copy()
            st.session_state.uploaded_file_name = f"gradient_{gradient_type.lower()}"
            st.session_state.is_video = False
            st.success("✅ Gradient image generated!")
    
    elif image_source == "Noise Pattern":
        st.markdown("**📊 Generate Noise Pattern**")
        noise_type = st.selectbox("Noise Type", ["Gaussian", "Salt & Pepper", "Uniform", "Perlin-like"])
        
        size_cols = st.columns(2)
        with size_cols[0]:
            img_width = st.number_input("Width", 64, 4096, 512, key="noise_w")
        with size_cols[1]:
            img_height = st.number_input("Height", 64, 4096, 512, key="noise_h")
        
        is_color = st.checkbox("Color Noise", value=False)
        
        if st.button("📊 Generate Noise", type="primary"):
            if noise_type == "Gaussian":
                if is_color:
                    img = np.random.normal(128, 50, (img_height, img_width, 3)).astype(np.uint8)
                else:
                    gray = np.random.normal(128, 50, (img_height, img_width)).astype(np.uint8)
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            elif noise_type == "Salt & Pepper":
                img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 128
                prob = 0.05
                salt = np.random.random((img_height, img_width)) < prob
                pepper = np.random.random((img_height, img_width)) < prob
                img[salt] = [255, 255, 255]
                img[pepper] = [0, 0, 0]
            elif noise_type == "Uniform":
                if is_color:
                    img = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
                else:
                    gray = np.random.randint(0, 256, (img_height, img_width), dtype=np.uint8)
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:  # Perlin-like (simplified)
                scale = 50
                x = np.linspace(0, img_width / scale, img_width)
                y = np.linspace(0, img_height / scale, img_height)
                xx, yy = np.meshgrid(x, y)
                noise = np.sin(xx * 2) * np.cos(yy * 2) + np.sin(xx * 3 + yy * 3) * 0.5
                noise = ((noise + 2) / 4 * 255).astype(np.uint8)
                img = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
            
            st.session_state.original_image = img.copy()
            st.session_state.current_image = img.copy()
            st.session_state.uploaded_file_name = f"noise_{noise_type.lower().replace(' ', '_')}"
            st.session_state.is_video = False
            st.success(f"✅ {noise_type} noise pattern generated!")
    
    # Handle uploaded file (only for Upload File source)
    if image_source == "Upload File" and uploaded_file is not None:
        file_name = uploaded_file.name
        file_ext = file_name.lower().split('.')[-1]
        is_video = file_ext in ['mp4', 'avi', 'mov', 'mkv']
        
        if st.session_state.uploaded_file_name != file_name:
            st.session_state.uploaded_file_name = file_name
            st.session_state.is_video = is_video
            st.session_state.use_as_input = False
            
            if is_video:
                st.session_state.video_file = uploaded_file
                frame, total_frames = load_video_frame(uploaded_file, 0)
                if frame is not None:
                    st.session_state.original_image = frame.copy()
                    st.session_state.current_image = frame.copy()
                    st.session_state.video_frame = 0
            else:
                new_image = load_image_from_file(uploaded_file)
                if new_image is not None:
                    st.session_state.original_image = new_image.copy()
                    st.session_state.current_image = new_image.copy()
    
    if st.session_state.current_image is not None:
        if st.session_state.is_video and uploaded_file is not None:
            video_info = get_video_info(uploaded_file)
            st.session_state.video_total_frames = video_info['total_frames']
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea22, #764ba222); padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem;">
                <strong>🎬 Video Info:</strong> {video_info['width']}x{video_info['height']} | {video_info['fps']:.1f} FPS | {video_info['duration']:.1f}s | {video_info['total_frames']} frames
            </div>
            """, unsafe_allow_html=True)
            
            # Video playback controls - all buttons on same line
            play_cols = st.columns([1, 1, 1, 1, 1])
            
            with play_cols[0]:
                if st.button("⏮️", key="video_start", help="Go to start"):
                    st.session_state.video_frame = 0
                    st.session_state.video_playing = False
                    frame, _ = load_video_frame(uploaded_file, 0)
                    if frame is not None:
                        st.session_state.current_image = frame.copy()
                        if not st.session_state.use_as_input:
                            st.session_state.original_image = frame.copy()
                    st.rerun()
            
            with play_cols[1]:
                if st.button("⏪", key="video_prev", help="Previous frame"):
                    if st.session_state.video_frame > 0:
                        st.session_state.video_frame -= 1
                        st.session_state.video_playing = False
                        frame, _ = load_video_frame(uploaded_file, st.session_state.video_frame)
                        if frame is not None:
                            st.session_state.current_image = frame.copy()
                            if not st.session_state.use_as_input:
                                st.session_state.original_image = frame.copy()
                        st.rerun()
            
            with play_cols[2]:
                play_label = "⏸️" if st.session_state.video_playing else "▶️"
                if st.button(play_label, key="video_play", type="primary" if not st.session_state.video_playing else "secondary"):
                    st.session_state.video_playing = not st.session_state.video_playing
                    if st.session_state.video_playing:
                        # Load current frame when starting to play
                        frame, _ = load_video_frame(uploaded_file, st.session_state.video_frame)
                        if frame is not None:
                            st.session_state.current_image = frame.copy()
                    st.rerun()
            
            with play_cols[3]:
                if st.button("⏩", key="video_next", help="Next frame"):
                    if st.session_state.video_frame < video_info['total_frames'] - 1:
                        st.session_state.video_frame += 1
                        st.session_state.video_playing = False
                        frame, _ = load_video_frame(uploaded_file, st.session_state.video_frame)
                        if frame is not None:
                            st.session_state.current_image = frame.copy()
                            if not st.session_state.use_as_input:
                                st.session_state.original_image = frame.copy()
                        st.rerun()
            
            with play_cols[4]:
                if st.button("⏭️", key="video_end", help="Go to end"):
                    st.session_state.video_frame = video_info['total_frames'] - 1
                    st.session_state.video_playing = False
                    frame, _ = load_video_frame(uploaded_file, st.session_state.video_frame)
                    if frame is not None:
                        st.session_state.current_image = frame.copy()
                        if not st.session_state.use_as_input:
                            st.session_state.original_image = frame.copy()
                    st.rerun()
            
            # Frame slider (only update if not playing)
            if not st.session_state.video_playing:
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
                    st.rerun()
            else:
                # Show frame progress when playing
                st.progress(st.session_state.video_frame / max(1, video_info['total_frames'] - 1))
                st.caption(f"Frame {st.session_state.video_frame + 1} / {video_info['total_frames']}")
                
                # Load next frame for playback (will rerun at end of page)
                if st.session_state.video_frame < video_info['total_frames'] - 1:
                    st.session_state.video_frame += 1
                    frame, _ = load_video_frame(uploaded_file, st.session_state.video_frame)
                    if frame is not None:
                        st.session_state.current_image = frame.copy()
                        if not st.session_state.use_as_input:
                            st.session_state.original_image = frame.copy()
                else:
                    # Reached end of video
                    st.session_state.video_playing = False
        
        if st.session_state.effect_pipeline:
            processed = apply_effect_pipeline(
                st.session_state.current_image,
                st.session_state.effect_pipeline
            )
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #28a74522, #20c99722); padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #28a745;">
                <strong>🔗 Pipeline ({len(st.session_state.effect_pipeline)}):</strong> {get_pipeline_summary(st.session_state.effect_pipeline)}
            </div>
            """, unsafe_allow_html=True)
        else:
            processed = st.session_state.current_image.copy()
        
        # Check if current effect requires interactive input
        requires_mask = method_selected and selected_category and registry.effect_requires_mask(selected_category)
        requires_points = method_selected and selected_category and registry.effect_requires_points(selected_category)
        
        # Interactive mode selection
        if requires_mask or requires_points:
            st.markdown("---")
            if requires_mask:
                st.markdown("### 🖌️ Draw Mask")
                st.markdown("Draw on the image below to create a mask for inpainting.")
                render_mask_canvas(processed, key="main_mask_canvas", height=350)
                
                # Update session state with binary options
                st.session_state.mask_binary = st.session_state.get("main_mask_canvas_binary", True)
                st.session_state.mask_threshold = st.session_state.get("main_mask_canvas_threshold", 127)
                
                # Show current mask preview if exists
                if st.session_state.mask_data is not None:
                    mask_preview_col1, mask_preview_col2 = st.columns(2)
                    with mask_preview_col1:
                        st.markdown("**Current Mask:**")
                        mask_display = st.session_state.mask_data
                        if len(mask_display.shape) == 3:
                            mask_display = cv2.cvtColor(mask_display, cv2.COLOR_BGR2GRAY)
                        
                        # Show binary version if enabled
                        if st.session_state.get('mask_binary', True):
                            threshold = st.session_state.get('mask_threshold', 127)
                            _, mask_display = cv2.threshold(mask_display.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
                        
                        st.image(mask_display, caption="Mask (white = area to process)", width=200)
                    
                    with mask_preview_col2:
                        if st.button("🗑️ Clear Mask", key="clear_mask_btn"):
                            st.session_state.mask_data = None
                            st.rerun()
                
                # Manual mask input as fallback
                with st.expander("📤 Or upload a mask image"):
                    mask_file = st.file_uploader("Upload mask (white = area to process)", 
                                                  type=["png", "jpg", "jpeg"], 
                                                  key="mask_upload")
                    if mask_file is not None:
                        mask_bytes = np.asarray(bytearray(mask_file.read()), dtype=np.uint8)
                        mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
                        if mask_img is not None:
                            # Resize mask to match image
                            h, w = processed.shape[:2]
                            mask_img = cv2.resize(mask_img, (w, h))
                            st.session_state.mask_data = mask_img
                            st.success("✅ Mask loaded! Binary conversion will be applied based on settings above.")
                
                # Quick mask creation buttons
                with st.expander("🎨 Quick Mask Creation"):
                    st.markdown("Create a simple mask by drawing a shape:")
                    shape_col1, shape_col2 = st.columns(2)
                    with shape_col1:
                        mask_shape = st.selectbox("Shape", ["Rectangle", "Circle", "Ellipse"], key="mask_shape")
                    with shape_col2:
                        mask_size = st.slider("Size %", 10, 90, 30, key="mask_size_pct")
                    
                    if st.button("Create Mask", key="create_quick_mask"):
                        h, w = processed.shape[:2]
                        quick_mask = np.zeros((h, w), dtype=np.uint8)
                        center_x, center_y = w // 2, h // 2
                        size_w = int(w * mask_size / 100)
                        size_h = int(h * mask_size / 100)
                        
                        if mask_shape == "Rectangle":
                            x1, y1 = center_x - size_w // 2, center_y - size_h // 2
                            x2, y2 = center_x + size_w // 2, center_y + size_h // 2
                            cv2.rectangle(quick_mask, (x1, y1), (x2, y2), 255, -1)
                        elif mask_shape == "Circle":
                            radius = min(size_w, size_h) // 2
                            cv2.circle(quick_mask, (center_x, center_y), radius, 255, -1)
                        elif mask_shape == "Ellipse":
                            cv2.ellipse(quick_mask, (center_x, center_y), (size_w // 2, size_h // 2), 0, 0, 360, 255, -1)
                        
                        st.session_state.mask_data = quick_mask
                        st.success("✅ Quick mask created!")
                        st.rerun()
            
            if requires_points:
                num_pts = registry.get_num_points_required(selected_category)
                point_labels = registry.get_point_labels(selected_category, selected_method)
                
                # Determine actual number of points needed based on method
                actual_num_pts = num_pts
                if selected_method in ["Perspective Correction", "Bird's Eye View"]:
                    actual_num_pts = 4
                    point_labels = point_labels[:4] if len(point_labels) >= 4 else ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                elif selected_method == "Affine Transform 3-Point":
                    actual_num_pts = 6
                elif selected_method == "Perspective Transform 4-Point":
                    actual_num_pts = 8
                elif selected_method in ["Affine Shear", "Affine Scale & Rotate", "Affine Translation", "Perspective Tilt"]:
                    # These don't need points
                    actual_num_pts = 0
                
                if actual_num_pts > 0:
                    st.markdown(f"### 📍 Select {actual_num_pts} Points")
                    st.markdown(f"Click on the image to select points for the transformation.")
                    render_point_selector(processed, actual_num_pts, point_labels[:actual_num_pts], 
                                         key="main_point_canvas", height=350)
                    
                    # Manual point input as fallback
                    with st.expander("⌨️ Or enter points manually"):
                        manual_points = []
                        cols = st.columns(min(actual_num_pts, 4))
                        h, w = processed.shape[:2]
                        for i in range(actual_num_pts):
                            col = cols[i % len(cols)]
                            with col:
                                label = point_labels[i] if i < len(point_labels) else f"Point {i+1}"
                                st.markdown(f"**{label}**")
                                px = st.number_input(f"X", 0, w-1, w//2, key=f"manual_pt_{i}_x")
                                py = st.number_input(f"Y", 0, h-1, h//2, key=f"manual_pt_{i}_y")
                                manual_points.append([px, py])
                        
                        if st.button("Apply Manual Points", key="apply_manual_pts"):
                            st.session_state.selected_points = manual_points
                            st.success(f"Set {len(manual_points)} points!")
            
            st.markdown("---")
        
        if method_selected and selected_method:
            preview_with_current = apply_opencv_method(
                processed,
                selected_category,
                selected_method,
                param_values
            )
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffc10722, #ff980022); padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid #ff9800;">
                <strong>👁️ Preview:</strong> {selected_method} <span style="font-size: 11px; color: #666;">(adjust sliders to see changes instantly)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            preview_with_current = processed
        
        st.session_state.processed_image = preview_with_current
        
        view_col1, view_col2 = st.columns([3, 1])
        with view_col2:
            compare_mode = st.checkbox("🔀 Compare Slider", value=st.session_state.compare_mode, key="compare_toggle")
            st.session_state.compare_mode = compare_mode
        
        if st.session_state.compare_mode:
            st.markdown("**🔀 Drag the slider on the image to compare**")
            
            input_rgb = cv2.cvtColor(st.session_state.current_image, cv2.COLOR_BGR2RGB)
            if len(preview_with_current.shape) == 2:
                processed_display = cv2.cvtColor(preview_with_current, cv2.COLOR_GRAY2RGB)
            else:
                processed_display = cv2.cvtColor(preview_with_current, cv2.COLOR_BGR2RGB)
            
            _, input_buffer = cv2.imencode('.jpg', cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, processed_buffer = cv2.imencode('.jpg', cv2.cvtColor(processed_display, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            input_b64 = base64.b64encode(input_buffer).decode()
            processed_b64 = base64.b64encode(processed_buffer).decode()
            
            img_height, img_width = input_rgb.shape[:2]
            aspect_ratio = img_height / img_width
            container_height = int(600 * aspect_ratio)
            
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
                        
                        if (percentage > 0) {{
                            overlayImg.style.width = (rect.width) + 'px';
                        }}
                    }}
                    
                    bgImg.onload = function() {{
                        overlayImg.style.width = container.offsetWidth + 'px';
                    }};
                    
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
                    
                    container.addEventListener('click', function(e) {{
                        updateSlider(e.clientX);
                    }});
                </script>
            </body>
            </html>
            '''
            
            components.html(html_code, height=container_height + 20)
        else:
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("**📥 Original / Input**")
                input_rgb = cv2.cvtColor(st.session_state.current_image, cv2.COLOR_BGR2RGB)
                st.image(input_rgb)
            
            with img_col2:
                st.markdown("**📤 Preview**")
                if len(preview_with_current.shape) == 2:
                    st.image(preview_with_current)
                else:
                    processed_rgb = cv2.cvtColor(preview_with_current, cv2.COLOR_BGR2RGB)
                    st.image(processed_rgb)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        
        with btn_col1:
            if st.button("🔄 Use as Input", type="primary"):
                st.session_state.current_image = preview_with_current.copy()
                st.session_state.use_as_input = True
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
        
        if st.session_state.use_as_input:
            st.info("🔗 Using processed output as input - add more effects to the pipeline!")
    else:
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
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        for i, item in enumerate(st.session_state.history[:20]):
            with st.container():
                if os.path.exists(item["filepath"]):
                    full_img = cv2.imread(item["filepath"])
                    if full_img is not None:
                        full_img_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
                        h, w = full_img_rgb.shape[:2]
                        max_preview = 300
                        if w > h:
                            preview_w = max_preview
                            preview_h = int(h * max_preview / w)
                        else:
                            preview_h = max_preview
                            preview_w = int(w * max_preview / h)
                        preview_img = cv2.resize(full_img_rgb, (preview_w, preview_h))
                        
                        _, preview_buffer = cv2.imencode('.png', cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
                        preview_b64 = base64.b64encode(preview_buffer).decode()
                        
                        thumbnail_rgb = cv2.cvtColor(item["thumbnail"], cv2.COLOR_BGR2RGB)
                        _, thumb_buffer = cv2.imencode('.png', cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2BGR))
                        thumb_b64 = base64.b64encode(thumb_buffer).decode()
                        
                        st.markdown(f"""
                        <div class="thumbnail-tooltip" style="text-align: center; margin-bottom: 8px;">
                            <img src="data:image/png;base64,{thumb_b64}" style="max-width: 80px; max-height: 60px; border-radius: 6px; cursor: pointer; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div class="tooltip-image">
                                <img src="data:image/png;base64,{preview_b64}">
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    thumbnail_rgb = cv2.cvtColor(item["thumbnail"], cv2.COLOR_BGR2RGB)
                    st.image(thumbnail_rgb, width=80)
                
                st.markdown(f"<p style='font-size: 12px; margin: 2px 0; font-weight: 600;'>{item['method']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 10px; color: #888; margin: 0;'>{item['timestamp'][:8]}</p>", unsafe_allow_html=True)
                
                pipeline = item.get('pipeline', [])
                if pipeline and len(pipeline) > 0:
                    st.markdown(f"<p style='font-size: 9px; color: #28a745; margin: 2px 0; background: #e8f5e9; padding: 2px 6px; border-radius: 4px; display: inline-block;'>🔗 {len(pipeline)} effect(s)</p>", unsafe_allow_html=True)
                
                elif item.get('params') and len(item['params']) > 0:
                    params_str = " | ".join([f"{k}={v}" for k, v in item['params'].items()])
                    st.markdown(f"<p style='font-size: 9px; color: #667eea; margin: 2px 0; background: #f0f2ff; padding: 2px 6px; border-radius: 4px; display: inline-block;'>{params_str}</p>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📥", key=f"load_{i}", help="Load this image"):
                        loaded_img = cv2.imread(item["filepath"])
                        if loaded_img is not None:
                            st.session_state.current_image = loaded_img.copy()
                            st.session_state.use_as_input = True
                
                with col2:
                    if pipeline and len(pipeline) > 0:
                        if st.button("🔗", key=f"pipeline_{i}", help="Load this pipeline"):
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

# Code Generation Panel
if st.session_state.show_code_panel:
    with code_col:
        st.markdown("### 💻 Code Generator")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        if not st.session_state.effect_pipeline:
            st.info("Add effects to your pipeline to see the generated code!")
        else:
            st.markdown(f"**📝 Pipeline Code ({len(st.session_state.effect_pipeline)} effects)**")
            
            for i, effect in enumerate(st.session_state.effect_pipeline):
                code_info = generate_opencv_code(effect['method'], effect['params'], effect.get('category'))
                
                st.markdown(f"""
                <div class="code-step">
                    <div class="code-step-header">
                        <span class="code-step-number">{i + 1}</span>
                        <span class="code-step-method">{effect['method']}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                if code_info['param_info'] and code_info['param_info'][0]['params']:
                    params_display = code_info['param_info'][0]['params']
                    params_html = ", ".join([f"<span class='code-param'>{k}</span>=<span class='code-value'>{v}</span>" for k, v in params_display.items()])
                    st.markdown(f"""
                    <div style="font-size: 10px; color: #999; margin-bottom: 6px; padding-left: 28px;">
                        {params_html}
                    </div>
                    """, unsafe_allow_html=True)
                
                code_text = "\\n".join(code_info['code_lines'])
                st.markdown(f"""
                    <div class="code-block">{code_text}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            with st.expander("📄 Complete Python Script", expanded=False):
                full_code = generate_full_pipeline_code(st.session_state.effect_pipeline)
                st.code(full_code, language="python")
                
                st.download_button(
                    "⬇️ Download Script",
                    data=full_code,
                    file_name="opencv_pipeline.py",
                    mime="text/plain",
                    key="download_full_code"
                )
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("**📋 Quick Reference**")
            
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
                st.dataframe(df, hide_index=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>Built with ❤️ using Streamlit , OpenCV By Rajilesh Panoli</p>
    <p style="font-size: 0.8rem;">Stack multiple effects, reorder, and export your pipelines!</p>
    <p style="font-size: 0.7rem; color: #999;">Effects are loaded dynamically from the <code>effects/</code> folder - add or remove as needed!</p>
</div>
""", unsafe_allow_html=True)

# Always save pipeline to localStorage at the end of each render
save_pipeline_to_storage()

# Video auto-play: rerun at the very end after all UI is rendered
if st.session_state.get('video_playing', False) and st.session_state.get('is_video', False):
    import time
    time.sleep(0.05)  # Small delay for ~20 FPS playback
    st.rerun()
