"""
Contour Detection Effects

This module provides contour detection effects including:
- Find Contours
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class ContourDetectionEffect(BaseEffect):
    """Contour detection effects for images"""
    
    @property
    def category_name(self) -> str:
        return "Contour Detection"
    
    @property
    def category_icon(self) -> str:
        return "🔍"
    
    def get_methods(self):
        return {
            "Find Contours": EffectMethod(
                name="Find Contours",
                description="Detect and draw contours",
                function="cv2.findContours",
                params=[
                    EffectParam("mode", "dropdown", "Retrieval Mode", "External", options=["External", "List", "Tree", "Component"]),
                    EffectParam("thickness", "slider", "Line Thickness", 2, 1, 10, 1),
                    EffectParam("threshold", "slider", "Threshold", 127, 0, 255, 1)
                ]
            )
        }
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        
        if method_name == "Find Contours":
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
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        if method_name == "Find Contours":
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
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
