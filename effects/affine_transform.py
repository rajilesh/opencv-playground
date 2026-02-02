"""
Affine Transform Effects

This module provides affine transformation effects:
- Affine Transform (3-point mapping)
- Warp Affine with custom matrix
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class AffineTransformEffect(BaseEffect):
    """Affine transformation effects for images"""
    
    @property
    def category_name(self) -> str:
        return "Affine Transform"
    
    @property
    def category_icon(self) -> str:
        return "🔀"
    
    @property
    def requires_points(self) -> bool:
        """This effect requires points to be selected"""
        return True
    
    @property
    def num_points_required(self) -> int:
        """Affine transform needs 3 source + 3 destination points"""
        return 6  # 3 source, 3 destination
    
    def get_point_labels(self, method_name: str) -> list:
        """Return labels for the 6 points (3 source + 3 destination)"""
        if method_name == "Affine Transform 3-Point":
            return [
                "Src Point 1", "Src Point 2", "Src Point 3",
                "Dst Point 1", "Dst Point 2", "Dst Point 3"
            ]
        return []
    
    def get_methods(self):
        return {
            "Affine Transform 3-Point": EffectMethod(
                name="Affine Transform 3-Point",
                description="Transform image using 3 point pairs (source → destination). Select 3 source points, then 3 destination points.",
                function="cv2.warpAffine",
                params=[]
            ),
            "Affine Shear": EffectMethod(
                name="Affine Shear",
                description="Apply shear transformation",
                function="cv2.warpAffine",
                params=[
                    EffectParam("shear_x", "slider", "Shear X", 0.0, -1.0, 1.0, 0.05),
                    EffectParam("shear_y", "slider", "Shear Y", 0.0, -1.0, 1.0, 0.05)
                ]
            ),
            "Affine Scale & Rotate": EffectMethod(
                name="Affine Scale & Rotate",
                description="Combined scale and rotation transform",
                function="cv2.warpAffine",
                params=[
                    EffectParam("scale", "slider", "Scale", 1.0, 0.1, 3.0, 0.1),
                    EffectParam("angle", "slider", "Angle (degrees)", 0, -180, 180, 1),
                    EffectParam("center_x", "slider", "Center X (%)", 50, 0, 100, 1),
                    EffectParam("center_y", "slider", "Center Y (%)", 50, 0, 100, 1)
                ]
            ),
            "Affine Translation": EffectMethod(
                name="Affine Translation",
                description="Translate/shift the image",
                function="cv2.warpAffine",
                params=[
                    EffectParam("tx", "slider", "Translate X (px)", 0, -500, 500, 1),
                    EffectParam("ty", "slider", "Translate Y (px)", 0, -500, 500, 1)
                ]
            )
        }
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        
        if method_name == "Affine Transform 3-Point":
            # Get points from params (selected by user)
            points = params.get("_points", [])
            if len(points) < 6:
                # Not enough points, return original
                return img
            
            # First 3 points are source, last 3 are destination
            src_pts = np.float32(points[:3])
            dst_pts = np.float32(points[3:6])
            
            # Get affine transformation matrix
            matrix = cv2.getAffineTransform(src_pts, dst_pts)
            result = cv2.warpAffine(img, matrix, (w, h))
            
        elif method_name == "Affine Shear":
            shear_x = params.get("shear_x", 0.0)
            shear_y = params.get("shear_y", 0.0)
            
            # Shear matrix: [1, shear_x, 0], [shear_y, 1, 0]
            matrix = np.float32([
                [1, shear_x, 0],
                [shear_y, 1, 0]
            ])
            
            # Calculate new dimensions to fit sheared image
            new_w = int(w + abs(shear_x) * h)
            new_h = int(h + abs(shear_y) * w)
            
            # Adjust translation to keep image centered
            if shear_x < 0:
                matrix[0, 2] = -shear_x * h
            if shear_y < 0:
                matrix[1, 2] = -shear_y * w
            
            result = cv2.warpAffine(img, matrix, (new_w, new_h))
            
        elif method_name == "Affine Scale & Rotate":
            scale = params.get("scale", 1.0)
            angle = params.get("angle", 0)
            center_x_pct = params.get("center_x", 50)
            center_y_pct = params.get("center_y", 50)
            
            center_x = int(w * center_x_pct / 100)
            center_y = int(h * center_y_pct / 100)
            
            matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
            result = cv2.warpAffine(img, matrix, (w, h))
            
        elif method_name == "Affine Translation":
            tx = params.get("tx", 0)
            ty = params.get("ty", 0)
            
            matrix = np.float32([
                [1, 0, tx],
                [0, 1, ty]
            ])
            result = cv2.warpAffine(img, matrix, (w, h))
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        if method_name == "Affine Transform 3-Point":
            code_lines.append("# Define source and destination points (3 pairs)")
            code_lines.append("src_pts = np.float32([[x1, y1], [x2, y2], [x3, y3]])")
            code_lines.append("dst_pts = np.float32([[x1', y1'], [x2', y2'], [x3', y3']])")
            code_lines.append("matrix = cv2.getAffineTransform(src_pts, dst_pts)")
            code_lines.append("h, w = img.shape[:2]")
            code_lines.append("result = cv2.warpAffine(img, matrix, (w, h))")
            param_info.append({"function": "cv2.getAffineTransform + cv2.warpAffine", "params": {"points": "3 pairs"}})
            
        elif method_name == "Affine Shear":
            shear_x = params.get("shear_x", 0.0)
            shear_y = params.get("shear_y", 0.0)
            code_lines.append(f"matrix = np.float32([[1, {shear_x}, 0], [{shear_y}, 1, 0]])")
            code_lines.append("h, w = img.shape[:2]")
            code_lines.append("result = cv2.warpAffine(img, matrix, (w, h))")
            param_info.append({"function": "cv2.warpAffine", "params": {"shear_x": shear_x, "shear_y": shear_y}})
            
        elif method_name == "Affine Scale & Rotate":
            scale = params.get("scale", 1.0)
            angle = params.get("angle", 0)
            code_lines.append("h, w = img.shape[:2]")
            code_lines.append("center = (w // 2, h // 2)")
            code_lines.append(f"matrix = cv2.getRotationMatrix2D(center, {angle}, {scale})")
            code_lines.append("result = cv2.warpAffine(img, matrix, (w, h))")
            param_info.append({"function": "cv2.warpAffine", "params": {"scale": scale, "angle": angle}})
            
        elif method_name == "Affine Translation":
            tx = params.get("tx", 0)
            ty = params.get("ty", 0)
            code_lines.append(f"matrix = np.float32([[1, 0, {tx}], [0, 1, {ty}]])")
            code_lines.append("h, w = img.shape[:2]")
            code_lines.append("result = cv2.warpAffine(img, matrix, (w, h))")
            param_info.append({"function": "cv2.warpAffine", "params": {"tx": tx, "ty": ty}})
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
