"""
Morphological Operations Effects

This module provides morphological operation effects including:
- Erosion
- Dilation
- Opening
- Closing
- Gradient
- Top Hat
- Black Hat
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class MorphologicalEffect(BaseEffect):
    """Morphological operation effects for images"""
    
    @property
    def category_name(self) -> str:
        return "Morphological Operations"
    
    @property
    def category_icon(self) -> str:
        return "🔲"
    
    def get_methods(self):
        return {
            "Erosion": EffectMethod(
                name="Erosion",
                description="Erode the image - shrinks bright regions",
                function="cv2.erode",
                params=[
                    EffectParam("kernel_size", "slider", "Kernel Size", 5, 1, 21, 1),
                    EffectParam("iterations", "slider", "Iterations", 1, 1, 10, 1),
                    EffectParam("kernel_shape", "dropdown", "Kernel Shape", "Rectangle", options=["Rectangle", "Ellipse", "Cross"])
                ]
            ),
            "Dilation": EffectMethod(
                name="Dilation",
                description="Dilate the image - expands bright regions",
                function="cv2.dilate",
                params=[
                    EffectParam("kernel_size", "slider", "Kernel Size", 5, 1, 21, 1),
                    EffectParam("iterations", "slider", "Iterations", 1, 1, 10, 1),
                    EffectParam("kernel_shape", "dropdown", "Kernel Shape", "Rectangle", options=["Rectangle", "Ellipse", "Cross"])
                ]
            ),
            "Opening": EffectMethod(
                name="Opening",
                description="Opening - erosion followed by dilation",
                function="cv2.morphologyEx",
                params=[
                    EffectParam("kernel_size", "slider", "Kernel Size", 5, 1, 21, 1),
                    EffectParam("kernel_shape", "dropdown", "Kernel Shape", "Rectangle", options=["Rectangle", "Ellipse", "Cross"])
                ]
            ),
            "Closing": EffectMethod(
                name="Closing",
                description="Closing - dilation followed by erosion",
                function="cv2.morphologyEx",
                params=[
                    EffectParam("kernel_size", "slider", "Kernel Size", 5, 1, 21, 1),
                    EffectParam("kernel_shape", "dropdown", "Kernel Shape", "Rectangle", options=["Rectangle", "Ellipse", "Cross"])
                ]
            ),
            "Gradient": EffectMethod(
                name="Gradient",
                description="Morphological gradient - difference between dilation and erosion",
                function="cv2.morphologyEx",
                params=[
                    EffectParam("kernel_size", "slider", "Kernel Size", 5, 1, 21, 1),
                    EffectParam("kernel_shape", "dropdown", "Kernel Shape", "Rectangle", options=["Rectangle", "Ellipse", "Cross"])
                ]
            ),
            "Top Hat": EffectMethod(
                name="Top Hat",
                description="Top hat - difference between image and opening",
                function="cv2.morphologyEx",
                params=[
                    EffectParam("kernel_size", "slider", "Kernel Size", 5, 1, 21, 1),
                    EffectParam("kernel_shape", "dropdown", "Kernel Shape", "Rectangle", options=["Rectangle", "Ellipse", "Cross"])
                ]
            ),
            "Black Hat": EffectMethod(
                name="Black Hat",
                description="Black hat - difference between closing and image",
                function="cv2.morphologyEx",
                params=[
                    EffectParam("kernel_size", "slider", "Kernel Size", 5, 1, 21, 1),
                    EffectParam("kernel_shape", "dropdown", "Kernel Shape", "Rectangle", options=["Rectangle", "Ellipse", "Cross"])
                ]
            )
        }
    
    def _get_kernel_shape(self, shape_name: str):
        """Get OpenCV kernel shape from string name"""
        shapes = {
            "Rectangle": cv2.MORPH_RECT,
            "Ellipse": cv2.MORPH_ELLIPSE,
            "Cross": cv2.MORPH_CROSS
        }
        return shapes.get(shape_name, cv2.MORPH_RECT)
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        
        ksize = int(params.get("kernel_size", 5))
        shape = self._get_kernel_shape(params.get("kernel_shape", "Rectangle"))
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
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
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
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
