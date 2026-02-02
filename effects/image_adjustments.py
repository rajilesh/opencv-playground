"""
Image Adjustments Effects

This module provides image adjustment effects including:
- Brightness & Contrast
- Histogram Equalization
- CLAHE
- Gamma Correction
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class ImageAdjustmentsEffect(BaseEffect):
    """Image adjustment effects"""
    
    @property
    def category_name(self) -> str:
        return "Image Adjustments"
    
    @property
    def category_icon(self) -> str:
        return "🎚️"
    
    def get_methods(self):
        return {
            "Brightness & Contrast": EffectMethod(
                name="Brightness & Contrast",
                description="Adjust brightness and contrast",
                function="cv2.convertScaleAbs",
                params=[
                    EffectParam("alpha", "slider", "Contrast (Alpha)", 1.0, 0.0, 3.0, 0.1),
                    EffectParam("beta", "slider", "Brightness (Beta)", 0, -100, 100, 1)
                ]
            ),
            "Histogram Equalization": EffectMethod(
                name="Histogram Equalization",
                description="Enhance contrast using histogram equalization",
                function="cv2.equalizeHist",
                params=[]
            ),
            "CLAHE": EffectMethod(
                name="CLAHE",
                description="Contrast Limited Adaptive Histogram Equalization",
                function="cv2.createCLAHE",
                params=[
                    EffectParam("clipLimit", "slider", "Clip Limit", 2.0, 1.0, 10.0, 0.5),
                    EffectParam("tileGridSize", "slider", "Tile Grid Size", 8, 2, 16, 1)
                ]
            ),
            "Gamma Correction": EffectMethod(
                name="Gamma Correction",
                description="Apply gamma correction",
                function="gamma",
                params=[
                    EffectParam("gamma", "slider", "Gamma", 1.0, 0.1, 3.0, 0.1)
                ]
            )
        }
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        
        if method_name == "Brightness & Contrast":
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
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        if method_name == "Brightness & Contrast":
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
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
