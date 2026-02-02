"""
Homography Effects

This module provides homography/perspective transformation effects:
- Perspective Transform (4-point mapping)
- Perspective Warp
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class HomographyEffect(BaseEffect):
    """Homography/Perspective transformation effects for images"""
    
    @property
    def category_name(self) -> str:
        return "Homography"
    
    @property
    def category_icon(self) -> str:
        return "🔲"
    
    @property
    def requires_points(self) -> bool:
        """This effect requires points to be selected"""
        return True
    
    @property
    def num_points_required(self) -> int:
        """Homography needs 4 source + 4 destination points"""
        return 8  # 4 source, 4 destination
    
    def get_point_labels(self, method_name: str) -> list:
        """Return labels for the points"""
        if method_name == "Perspective Transform 4-Point":
            return [
                "Src Top-Left", "Src Top-Right", "Src Bottom-Right", "Src Bottom-Left",
                "Dst Top-Left", "Dst Top-Right", "Dst Bottom-Right", "Dst Bottom-Left"
            ]
        elif method_name in ["Perspective Correction", "Bird's Eye View"]:
            return ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        return []
    
    def get_methods(self):
        return {
            "Perspective Transform 4-Point": EffectMethod(
                name="Perspective Transform 4-Point",
                description="Transform image using 4 point pairs. Select 4 source corners, then 4 destination corners.",
                function="cv2.warpPerspective",
                params=[]
            ),
            "Perspective Correction": EffectMethod(
                name="Perspective Correction",
                description="Correct perspective by selecting 4 corners of a rectangular object. Points will map to a rectangle.",
                function="cv2.warpPerspective",
                params=[
                    EffectParam("output_width", "slider", "Output Width", 400, 100, 1000, 10),
                    EffectParam("output_height", "slider", "Output Height", 300, 100, 1000, 10)
                ]
            ),
            "Bird's Eye View": EffectMethod(
                name="Bird's Eye View",
                description="Create a top-down view by specifying 4 ground points",
                function="cv2.warpPerspective",
                params=[
                    EffectParam("scale", "slider", "Scale", 1.0, 0.5, 3.0, 0.1)
                ]
            ),
            "Perspective Tilt": EffectMethod(
                name="Perspective Tilt",
                description="Apply perspective tilt without manual point selection",
                function="cv2.warpPerspective",
                params=[
                    EffectParam("tilt_x", "slider", "Tilt X", 0.0, -0.5, 0.5, 0.01),
                    EffectParam("tilt_y", "slider", "Tilt Y", 0.0, -0.5, 0.5, 0.01),
                    EffectParam("perspective", "slider", "Perspective Strength", 0.0, 0.0, 0.002, 0.0001)
                ]
            )
        }
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        
        if method_name == "Perspective Transform 4-Point":
            points = params.get("_points", [])
            if len(points) < 8:
                return img
            
            src_pts = np.float32(points[:4])
            dst_pts = np.float32(points[4:8])
            
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            result = cv2.warpPerspective(img, matrix, (w, h))
            
        elif method_name == "Perspective Correction":
            points = params.get("_points", [])
            if len(points) < 4:
                return img
            
            src_pts = np.float32(points[:4])
            
            out_w = int(params.get("output_width", 400))
            out_h = int(params.get("output_height", 300))
            
            # Map to a rectangle
            dst_pts = np.float32([
                [0, 0],
                [out_w - 1, 0],
                [out_w - 1, out_h - 1],
                [0, out_h - 1]
            ])
            
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            result = cv2.warpPerspective(img, matrix, (out_w, out_h))
            
        elif method_name == "Bird's Eye View":
            points = params.get("_points", [])
            if len(points) < 4:
                return img
            
            src_pts = np.float32(points[:4])
            scale = params.get("scale", 1.0)
            
            # Calculate bounding rectangle of source points
            x_coords = [p[0] for p in src_pts]
            y_coords = [p[1] for p in src_pts]
            
            width = int((max(x_coords) - min(x_coords)) * scale)
            height = int((max(y_coords) - min(y_coords)) * scale)
            
            dst_pts = np.float32([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ])
            
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            result = cv2.warpPerspective(img, matrix, (width, height))
            
        elif method_name == "Perspective Tilt":
            tilt_x = params.get("tilt_x", 0.0)
            tilt_y = params.get("tilt_y", 0.0)
            perspective = params.get("perspective", 0.0)
            
            # Create perspective transformation matrix
            # Standard perspective matrix with tilt
            matrix = np.float32([
                [1 + tilt_x, tilt_y, 0],
                [tilt_y, 1 + tilt_x, 0],
                [perspective, perspective, 1]
            ])
            
            # Adjust to keep image centered
            src_pts = np.float32([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ])
            
            # Apply transformation to corners to calculate offset
            dst_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), matrix)
            dst_pts = dst_pts.reshape(-1, 2)
            
            # Center the result
            min_x = min(dst_pts[:, 0])
            min_y = min(dst_pts[:, 1])
            matrix[0, 2] = -min_x
            matrix[1, 2] = -min_y
            
            result = cv2.warpPerspective(img, matrix, (w, h))
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        if method_name == "Perspective Transform 4-Point":
            code_lines.append("# Define 4 source and 4 destination points")
            code_lines.append("src_pts = np.float32([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])")
            code_lines.append("dst_pts = np.float32([[x1',y1'], [x2',y2'], [x3',y3'], [x4',y4']])")
            code_lines.append("matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)")
            code_lines.append("h, w = img.shape[:2]")
            code_lines.append("result = cv2.warpPerspective(img, matrix, (w, h))")
            param_info.append({"function": "cv2.getPerspectiveTransform", "params": {"points": "4 pairs"}})
            
        elif method_name == "Perspective Correction":
            out_w = int(params.get("output_width", 400))
            out_h = int(params.get("output_height", 300))
            code_lines.append("# Select 4 corners of the object to correct")
            code_lines.append("src_pts = np.float32([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])")
            code_lines.append(f"dst_pts = np.float32([[0,0], [{out_w-1},0], [{out_w-1},{out_h-1}], [0,{out_h-1}]])")
            code_lines.append("matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)")
            code_lines.append(f"result = cv2.warpPerspective(img, matrix, ({out_w}, {out_h}))")
            param_info.append({"function": "cv2.warpPerspective", "params": {"output_size": f"{out_w}x{out_h}"}})
            
        elif method_name == "Perspective Tilt":
            tilt_x = params.get("tilt_x", 0.0)
            tilt_y = params.get("tilt_y", 0.0)
            code_lines.append(f"# Perspective tilt transformation")
            code_lines.append(f"matrix = np.float32([[1+{tilt_x}, {tilt_y}, 0], [{tilt_y}, 1+{tilt_x}, 0], [0, 0, 1]])")
            code_lines.append("h, w = img.shape[:2]")
            code_lines.append("result = cv2.warpPerspective(img, matrix, (w, h))")
            param_info.append({"function": "cv2.warpPerspective", "params": {"tilt_x": tilt_x, "tilt_y": tilt_y}})
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
