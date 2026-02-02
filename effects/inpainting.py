"""
Inpainting Effects

This module provides inpainting effects that can use a drawable mask:
- Inpaint NS (Navier-Stokes)
- Inpaint Telea
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class InpaintingEffect(BaseEffect):
    """Inpainting effects for image restoration"""
    
    @property
    def category_name(self) -> str:
        return "Inpainting"
    
    @property
    def category_icon(self) -> str:
        return "🖌️"
    
    @property
    def requires_mask(self) -> bool:
        """This effect requires a mask to be drawn"""
        return True
    
    def get_methods(self):
        return {
            "Inpaint NS": EffectMethod(
                name="Inpaint NS",
                description="Inpaint using Navier-Stokes method - good for small regions",
                function="cv2.inpaint",
                params=[
                    EffectParam("radius", "slider", "Inpaint Radius", 3, 1, 20, 1)
                ]
            ),
            "Inpaint Telea": EffectMethod(
                name="Inpaint Telea",
                description="Inpaint using Telea's method - fast marching method",
                function="cv2.inpaint",
                params=[
                    EffectParam("radius", "slider", "Inpaint Radius", 3, 1, 20, 1)
                ]
            )
        }
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        
        # Get the mask from params (drawn by user)
        mask = params.get("_mask")
        if mask is None:
            # No mask provided, return original
            return img
        
        # Ensure mask is proper format (single channel, uint8)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)
        
        # Threshold to binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        radius = int(params.get("radius", 3))
        
        if method_name == "Inpaint NS":
            result = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
        elif method_name == "Inpaint Telea":
            result = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        radius = int(params.get("radius", 3))
        
        if method_name == "Inpaint NS":
            code_lines.append("# mask should be a binary image where white pixels indicate areas to inpaint")
            code_lines.append(f"result = cv2.inpaint(img, mask, {radius}, cv2.INPAINT_NS)")
            param_info.append({"function": "cv2.inpaint", "params": {"radius": radius, "method": "INPAINT_NS"}})
        elif method_name == "Inpaint Telea":
            code_lines.append("# mask should be a binary image where white pixels indicate areas to inpaint")
            code_lines.append(f"result = cv2.inpaint(img, mask, {radius}, cv2.INPAINT_TELEA)")
            param_info.append({"function": "cv2.inpaint", "params": {"radius": radius, "method": "INPAINT_TELEA"}})
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
