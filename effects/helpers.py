"""
Helper Effects

This module provides helper effects including:
- Normalize
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class HelpersEffect(BaseEffect):
    """Helper effects for images"""
    
    @property
    def category_name(self) -> str:
        return "Helpers"
    
    @property
    def category_icon(self) -> str:
        return "🔧"
    
    def get_methods(self):
        return {
            "Normalize": EffectMethod(
                name="Normalize",
                description="Normalize pixel values to a range (enhances contrast)",
                function="cv2.normalize",
                params=[
                    EffectParam("alpha", "slider", "Min Value (Alpha)", 0, 0, 255, 1),
                    EffectParam("beta", "slider", "Max Value (Beta)", 255, 0, 255, 1),
                    EffectParam("norm_type", "dropdown", "Norm Type", "MINMAX", options=["MINMAX", "INF", "L1", "L2"])
                ]
            )
        }
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        
        if method_name == "Normalize":
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
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        if method_name == "Normalize":
            alpha = int(params.get("alpha", 0))
            beta = int(params.get("beta", 255))
            norm_type = params.get("norm_type", "MINMAX")
            code_lines.append(f"result = cv2.normalize(img, None, {alpha}, {beta}, cv2.NORM_{norm_type})")
            param_info.append({"function": "cv2.normalize", "params": {"alpha": alpha, "beta": beta, "norm_type": norm_type}})
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
