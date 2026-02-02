"""
Blurring & Smoothing Effects

This module provides various blur and smoothing effects including:
- Gaussian Blur
- Median Blur
- Bilateral Filter
- Box Filter
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class BlurringEffect(BaseEffect):
    """Blurring and smoothing effects for images"""
    
    @property
    def category_name(self) -> str:
        return "Blurring & Smoothing"
    
    @property
    def category_icon(self) -> str:
        return "💫"
    
    def get_methods(self):
        return {
            "GaussianBlur": EffectMethod(
                name="GaussianBlur",
                description="Apply Gaussian blur to smooth the image",
                function="cv2.GaussianBlur",
                params=[
                    EffectParam("kernel_size", "slider", "Kernel Size", 5, 1, 31, 2),
                    EffectParam("sigma", "slider", "Sigma", 0.0, 0.0, 10.0, 0.1)
                ]
            ),
            "MedianBlur": EffectMethod(
                name="MedianBlur",
                description="Apply median blur - great for noise reduction",
                function="cv2.medianBlur",
                params=[
                    EffectParam("kernel_size", "slider", "Kernel Size", 5, 1, 31, 2)
                ]
            ),
            "BilateralFilter": EffectMethod(
                name="BilateralFilter",
                description="Apply bilateral filter - preserves edges while smoothing",
                function="cv2.bilateralFilter",
                params=[
                    EffectParam("d", "slider", "Diameter", 9, 1, 15, 1),
                    EffectParam("sigmaColor", "slider", "Sigma Color", 75, 10, 200, 5),
                    EffectParam("sigmaSpace", "slider", "Sigma Space", 75, 10, 200, 5)
                ]
            ),
            "BoxFilter": EffectMethod(
                name="BoxFilter",
                description="Apply box filter (averaging blur)",
                function="cv2.boxFilter",
                params=[
                    EffectParam("kernel_size", "slider", "Kernel Size", 5, 1, 31, 1)
                ]
            )
        }
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        
        if method_name == "GaussianBlur":
            ksize = self._ensure_odd(int(params.get("kernel_size", 5)))
            sigma = params.get("sigma", 0)
            result = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        elif method_name == "MedianBlur":
            ksize = self._ensure_odd(int(params.get("kernel_size", 5)))
            result = cv2.medianBlur(img, ksize)
        elif method_name == "BilateralFilter":
            d = int(params.get("d", 9))
            sigmaColor = params.get("sigmaColor", 75)
            sigmaSpace = params.get("sigmaSpace", 75)
            result = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
        elif method_name == "BoxFilter":
            ksize = int(params.get("kernel_size", 5))
            result = cv2.boxFilter(img, -1, (ksize, ksize))
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        if method_name == "GaussianBlur":
            ksize = self._ensure_odd(int(params.get("kernel_size", 5)))
            sigma = params.get("sigma", 0)
            code_lines.append(f"result = cv2.GaussianBlur(img, ({ksize}, {ksize}), {sigma})")
            param_info.append({"function": "cv2.GaussianBlur", "params": {"ksize": f"({ksize}, {ksize})", "sigmaX": sigma}})
        elif method_name == "MedianBlur":
            ksize = self._ensure_odd(int(params.get("kernel_size", 5)))
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
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
