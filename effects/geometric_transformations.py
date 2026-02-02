"""
Geometric Transformations Effects

This module provides geometric transformation effects including:
- Resize
- Rotate
- Flip Horizontal
- Flip Vertical
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class GeometricTransformationsEffect(BaseEffect):
    """Geometric transformation effects for images"""
    
    @property
    def category_name(self) -> str:
        return "Geometric Transformations"
    
    @property
    def category_icon(self) -> str:
        return "📐"
    
    def get_methods(self):
        return {
            "Resize": EffectMethod(
                name="Resize",
                description="Resize the image",
                function="cv2.resize",
                params=[
                    EffectParam("scale", "slider", "Scale Factor", 1.0, 0.1, 3.0, 0.1),
                    EffectParam("interpolation", "dropdown", "Interpolation", "Linear", options=["Nearest", "Linear", "Area", "Cubic", "Lanczos"])
                ]
            ),
            "Rotate": EffectMethod(
                name="Rotate",
                description="Rotate the image",
                function="cv2.rotate",
                params=[
                    EffectParam("angle", "slider", "Angle (degrees)", 0, -180, 180, 1)
                ]
            ),
            "Flip Horizontal": EffectMethod(
                name="Flip Horizontal",
                description="Flip image horizontally",
                function="cv2.flip",
                params=[]
            ),
            "Flip Vertical": EffectMethod(
                name="Flip Vertical",
                description="Flip image vertically",
                function="cv2.flip",
                params=[]
            )
        }
    
    def _get_interpolation(self, interp_name: str):
        """Get OpenCV interpolation flag from string name"""
        interps = {
            "Nearest": cv2.INTER_NEAREST,
            "Linear": cv2.INTER_LINEAR,
            "Area": cv2.INTER_AREA,
            "Cubic": cv2.INTER_CUBIC,
            "Lanczos": cv2.INTER_LANCZOS4
        }
        return interps.get(interp_name, cv2.INTER_LINEAR)
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        
        if method_name == "Resize":
            scale = params.get("scale", 1.0)
            interp = self._get_interpolation(params.get("interpolation", "Linear"))
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
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        if method_name == "Resize":
            scale = params.get("scale", 1.0)
            interp = params.get("interpolation", "Linear")
            interp_map = {"Nearest": "INTER_NEAREST", "Linear": "INTER_LINEAR", "Area": "INTER_AREA", "Cubic": "INTER_CUBIC", "Lanczos": "INTER_LANCZOS4"}
            code_lines.append("height, width = img.shape[:2]")
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
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
