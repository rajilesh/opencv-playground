"""
Interactive Drawing Components for Streamlit

This module provides reusable UI components for:
- Drawing masks on images (for inpainting, masking effects)
- Selecting points on images (for affine transforms, homography)

These components can be used by any effect that requires user interaction.
"""

import streamlit as st
import numpy as np
import base64
import cv2
from typing import Tuple, List, Optional


def get_mask_drawing_component(image: np.ndarray, 
                                canvas_key: str = "mask_canvas",
                                brush_size: int = 15,
                                height: int = 400) -> Optional[np.ndarray]:
    """
    Creates an interactive mask drawing component on an image.
    
    Args:
        image: The base image to draw on (BGR format)
        canvas_key: Unique key for the canvas session state
        brush_size: Initial brush size
        height: Height of the canvas
        
    Returns:
        Binary mask as numpy array (white = masked area), or None if no mask drawn
    """
    import streamlit.components.v1 as components
    
    # Convert image to base64
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Calculate dimensions
    aspect_ratio = w / h
    canvas_width = int(height * aspect_ratio)
    
    # Resize image for display
    display_img = cv2.resize(img_rgb, (canvas_width, height))
    _, buffer = cv2.imencode('.png', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
    img_b64 = base64.b64encode(buffer).decode()
    
    # Initialize session state for mask data
    mask_data_key = f"{canvas_key}_data"
    if mask_data_key not in st.session_state:
        st.session_state[mask_data_key] = None
    
    # Create the HTML/JS canvas component
    html_code = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            .container {{
                position: relative;
                display: inline-block;
            }}
            canvas {{
                border-radius: 8px;
                cursor: crosshair;
            }}
            #baseCanvas {{
                position: absolute;
                top: 0;
                left: 0;
            }}
            #drawCanvas {{
                position: relative;
                z-index: 10;
            }}
            .controls {{
                display: flex;
                gap: 10px;
                margin-bottom: 10px;
                align-items: center;
                flex-wrap: wrap;
            }}
            .btn {{
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.2s;
            }}
            .btn-primary {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
            }}
            .btn-secondary {{
                background: #f0f0f0;
                color: #333;
            }}
            .btn:hover {{
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }}
            .slider-container {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            label {{
                font-size: 12px;
                color: #666;
            }}
            input[type="range"] {{
                width: 80px;
            }}
            .status {{
                font-size: 11px;
                color: #667eea;
                margin-left: auto;
            }}
        </style>
    </head>
    <body>
        <div class="controls">
            <div class="slider-container">
                <label>Brush Size:</label>
                <input type="range" id="brushSize" min="5" max="50" value="{brush_size}">
                <span id="brushSizeVal">{brush_size}</span>
            </div>
            <button class="btn btn-secondary" onclick="clearCanvas()">🗑️ Clear</button>
            <button class="btn btn-primary" onclick="saveMask()">💾 Apply Mask</button>
            <span class="status" id="status"></span>
        </div>
        
        <div class="container">
            <canvas id="baseCanvas" width="{canvas_width}" height="{height}"></canvas>
            <canvas id="drawCanvas" width="{canvas_width}" height="{height}"></canvas>
        </div>
        
        <script>
            const baseCanvas = document.getElementById('baseCanvas');
            const drawCanvas = document.getElementById('drawCanvas');
            const baseCtx = baseCanvas.getContext('2d');
            const drawCtx = drawCanvas.getContext('2d');
            const brushSizeInput = document.getElementById('brushSize');
            const brushSizeVal = document.getElementById('brushSizeVal');
            const status = document.getElementById('status');
            
            let isDrawing = false;
            let lastX = 0;
            let lastY = 0;
            
            // Load background image
            const img = new Image();
            img.onload = function() {{
                baseCtx.drawImage(img, 0, 0, {canvas_width}, {height});
            }};
            img.src = 'data:image/png;base64,{img_b64}';
            
            // Set up drawing canvas with semi-transparent fill
            drawCtx.fillStyle = 'rgba(255, 0, 0, 0)';
            drawCtx.fillRect(0, 0, {canvas_width}, {height});
            
            brushSizeInput.addEventListener('input', function() {{
                brushSizeVal.textContent = this.value;
            }});
            
            function startDrawing(e) {{
                isDrawing = true;
                [lastX, lastY] = getPos(e);
            }}
            
            function draw(e) {{
                if (!isDrawing) return;
                
                const [x, y] = getPos(e);
                const brushSize = parseInt(brushSizeInput.value);
                
                drawCtx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
                drawCtx.lineWidth = brushSize;
                drawCtx.lineCap = 'round';
                drawCtx.lineJoin = 'round';
                
                drawCtx.beginPath();
                drawCtx.moveTo(lastX, lastY);
                drawCtx.lineTo(x, y);
                drawCtx.stroke();
                
                [lastX, lastY] = [x, y];
            }}
            
            function stopDrawing() {{
                isDrawing = false;
            }}
            
            function getPos(e) {{
                const rect = drawCanvas.getBoundingClientRect();
                const clientX = e.touches ? e.touches[0].clientX : e.clientX;
                const clientY = e.touches ? e.touches[0].clientY : e.clientY;
                return [clientX - rect.left, clientY - rect.top];
            }}
            
            function clearCanvas() {{
                drawCtx.clearRect(0, 0, {canvas_width}, {height});
                status.textContent = 'Cleared';
            }}
            
            function saveMask() {{
                // Get the mask data as base64
                const maskData = drawCanvas.toDataURL('image/png');
                
                // Send to Streamlit
                window.parent.postMessage({{
                    type: 'mask_data',
                    key: '{canvas_key}',
                    data: maskData,
                    width: {w},
                    height: {h},
                    canvasWidth: {canvas_width},
                    canvasHeight: {height}
                }}, '*');
                
                status.textContent = '✓ Mask applied!';
            }}
            
            // Event listeners
            drawCanvas.addEventListener('mousedown', startDrawing);
            drawCanvas.addEventListener('mousemove', draw);
            drawCanvas.addEventListener('mouseup', stopDrawing);
            drawCanvas.addEventListener('mouseout', stopDrawing);
            
            drawCanvas.addEventListener('touchstart', startDrawing);
            drawCanvas.addEventListener('touchmove', draw);
            drawCanvas.addEventListener('touchend', stopDrawing);
        </script>
    </body>
    </html>
    '''
    
    components.html(html_code, height=height + 60)
    
    return st.session_state.get(mask_data_key)


def get_point_selection_component(image: np.ndarray,
                                   num_points: int,
                                   canvas_key: str = "point_canvas",
                                   height: int = 400,
                                   point_labels: List[str] = None) -> List[Tuple[int, int]]:
    """
    Creates an interactive point selection component on an image.
    
    Args:
        image: The base image to select points on (BGR format)
        num_points: Number of points to select
        canvas_key: Unique key for the canvas session state
        height: Height of the canvas
        point_labels: Optional labels for each point (e.g., ["Top-left", "Top-right", ...])
        
    Returns:
        List of (x, y) tuples representing selected points (in original image coordinates)
    """
    import streamlit.components.v1 as components
    
    # Convert image to base64
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    # Calculate dimensions
    aspect_ratio = w / h
    canvas_width = int(height * aspect_ratio)
    
    # Resize image for display
    display_img = cv2.resize(img_rgb, (canvas_width, height))
    _, buffer = cv2.imencode('.png', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
    img_b64 = base64.b64encode(buffer).decode()
    
    # Scale factors
    scale_x = w / canvas_width
    scale_y = h / height
    
    # Initialize session state
    points_key = f"{canvas_key}_points"
    if points_key not in st.session_state:
        st.session_state[points_key] = []
    
    # Generate point labels
    if point_labels is None:
        point_labels = [f"Point {i+1}" for i in range(num_points)]
    
    labels_js = str(point_labels).replace("'", '"')
    
    # Colors for different point groups (source vs destination)
    colors = ["#667eea", "#28a745", "#ffc107", "#dc3545", "#17a2b8", "#6c757d", "#e83e8c", "#fd7e14"]
    
    html_code = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            .container {{
                position: relative;
                display: inline-block;
            }}
            canvas {{
                border-radius: 8px;
                cursor: crosshair;
            }}
            .controls {{
                display: flex;
                gap: 10px;
                margin-bottom: 10px;
                align-items: center;
                flex-wrap: wrap;
            }}
            .btn {{
                padding: 8px 16px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.2s;
            }}
            .btn-primary {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
            }}
            .btn-secondary {{
                background: #f0f0f0;
                color: #333;
            }}
            .btn:hover {{
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }}
            .status {{
                font-size: 12px;
                color: #667eea;
                margin-left: auto;
            }}
            .instructions {{
                font-size: 11px;
                color: #666;
                margin-bottom: 8px;
                padding: 8px;
                background: #f8f9ff;
                border-radius: 6px;
                border-left: 3px solid #667eea;
            }}
            .point-list {{
                margin-top: 8px;
                font-size: 11px;
                max-height: 80px;
                overflow-y: auto;
            }}
            .point-item {{
                display: inline-block;
                padding: 2px 8px;
                margin: 2px;
                border-radius: 12px;
                font-weight: 500;
            }}
        </style>
    </head>
    <body>
        <div class="instructions" id="instructions">
            Click on the image to select {num_points} points. Currently selecting: <strong id="currentPoint">Point 1</strong>
        </div>
        
        <div class="controls">
            <button class="btn btn-secondary" onclick="clearPoints()">🗑️ Clear All</button>
            <button class="btn btn-secondary" onclick="undoPoint()">↩️ Undo</button>
            <button class="btn btn-primary" onclick="savePoints()">✓ Apply Points</button>
            <span class="status" id="status">0/{num_points} points</span>
        </div>
        
        <div class="container">
            <canvas id="canvas" width="{canvas_width}" height="{height}"></canvas>
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
            
            // Load background image
            const img = new Image();
            img.onload = function() {{
                ctx.drawImage(img, 0, 0, {canvas_width}, {height});
            }};
            img.src = 'data:image/png;base64,{img_b64}';
            
            function getColor(index) {{
                return colors[index % colors.length];
            }}
            
            function redraw() {{
                ctx.clearRect(0, 0, {canvas_width}, {height});
                ctx.drawImage(img, 0, 0, {canvas_width}, {height});
                
                // Draw points
                points.forEach((p, i) => {{
                    const color = getColor(i);
                    
                    // Draw point circle
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 8, 0, 2 * Math.PI);
                    ctx.fillStyle = color;
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    // Draw point number
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 10px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(i + 1, p.x, p.y);
                    
                    // Draw label
                    ctx.fillStyle = color;
                    ctx.font = '10px Arial';
                    ctx.textAlign = 'left';
                    ctx.fillText(labels[i] || ('Point ' + (i+1)), p.x + 12, p.y + 3);
                }});
                
                // Draw lines between consecutive points if applicable
                if (points.length > 1) {{
                    ctx.beginPath();
                    ctx.moveTo(points[0].x, points[0].y);
                    for (let i = 1; i < points.length; i++) {{
                        ctx.lineTo(points[i].x, points[i].y);
                    }}
                    if (points.length >= 4 && points.length % 4 === 0) {{
                        // Close polygon for sets of 4 points
                        for (let start = 0; start < points.length; start += 4) {{
                            if (start + 3 < points.length) {{
                                ctx.moveTo(points[start + 3].x, points[start + 3].y);
                                ctx.lineTo(points[start].x, points[start].y);
                            }}
                        }}
                    }} else if (points.length >= 3 && points.length % 3 === 0) {{
                        // Close triangle for sets of 3 points
                        for (let start = 0; start < points.length; start += 3) {{
                            if (start + 2 < points.length) {{
                                ctx.moveTo(points[start + 2].x, points[start + 2].y);
                                ctx.lineTo(points[start].x, points[start].y);
                            }}
                        }}
                    }}
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
                if (points.length < numPoints) {{
                    currentPoint.textContent = labels[points.length] || ('Point ' + (points.length + 1));
                }} else {{
                    currentPoint.textContent = 'All points selected!';
                }}
                
                // Update point list
                pointList.innerHTML = points.map((p, i) => {{
                    const origX = Math.round(p.x * scaleX);
                    const origY = Math.round(p.y * scaleY);
                    return '<span class="point-item" style="background:' + getColor(i) + '22; color:' + getColor(i) + ';">' + 
                           (labels[i] || ('P' + (i+1))) + ': (' + origX + ', ' + origY + ')</span>';
                }}).join('');
            }}
            
            canvas.addEventListener('click', function(e) {{
                if (points.length >= numPoints) {{
                    status.textContent = 'Max points reached! Clear to start over.';
                    return;
                }}
                
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                points.push({{ x: x, y: y }});
                redraw();
            }});
            
            function clearPoints() {{
                points = [];
                redraw();
            }}
            
            function undoPoint() {{
                if (points.length > 0) {{
                    points.pop();
                    redraw();
                }}
            }}
            
            function savePoints() {{
                // Convert to original image coordinates
                const origPoints = points.map(p => [
                    Math.round(p.x * scaleX),
                    Math.round(p.y * scaleY)
                ]);
                
                window.parent.postMessage({{
                    type: 'point_data',
                    key: '{canvas_key}',
                    points: origPoints
                }}, '*');
                
                status.textContent = '✓ Points applied!';
            }}
            
            updateStatus();
        </script>
    </body>
    </html>
    '''
    
    components.html(html_code, height=height + 120)
    
    return st.session_state.get(points_key, [])


def create_streamlit_canvas_for_mask(image: np.ndarray, key: str = "mask") -> Tuple[np.ndarray, int]:
    """
    A simpler mask drawing interface using Streamlit's native components.
    Returns (mask, brush_size)
    """
    h, w = image.shape[:2]
    
    col1, col2 = st.columns([3, 1])
    with col2:
        brush_size = st.slider("Brush Size", 5, 50, 15, key=f"{key}_brush")
        if st.button("Clear Mask", key=f"{key}_clear"):
            st.session_state[f"{key}_mask"] = np.zeros((h, w), dtype=np.uint8)
    
    # Initialize mask
    mask_key = f"{key}_mask"
    if mask_key not in st.session_state:
        st.session_state[mask_key] = np.zeros((h, w), dtype=np.uint8)
    
    return st.session_state[mask_key], brush_size


def create_streamlit_point_selector(image: np.ndarray, 
                                     num_points: int, 
                                     key: str = "points",
                                     point_labels: List[str] = None) -> List[Tuple[int, int]]:
    """
    A simpler point selection interface using Streamlit's native number inputs.
    """
    h, w = image.shape[:2]
    
    points_key = f"{key}_points"
    if points_key not in st.session_state:
        st.session_state[points_key] = []
    
    if point_labels is None:
        point_labels = [f"Point {i+1}" for i in range(num_points)]
    
    st.markdown(f"**Select {num_points} points:**")
    
    points = []
    cols = st.columns(min(num_points, 4))
    
    for i in range(num_points):
        col = cols[i % len(cols)]
        with col:
            st.markdown(f"**{point_labels[i]}**" if i < len(point_labels) else f"**Point {i+1}**")
            x = st.number_input(f"X", 0, w-1, w//2, key=f"{key}_p{i}_x")
            y = st.number_input(f"Y", 0, h-1, h//2, key=f"{key}_p{i}_y")
            points.append((x, y))
    
    if st.button("Apply Points", key=f"{key}_apply"):
        st.session_state[points_key] = points
    
    return st.session_state.get(points_key, [])
