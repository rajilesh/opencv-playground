"""
Noise Effects

This module provides noise-related effects including:
- Add Gaussian Noise
- Add Salt & Pepper Noise
- Denoise (fastNlMeans)
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class NoiseEffect(BaseEffect):
    """Noise-related effects for images"""
    
    @property
    def category_name(self) -> str:
        return "Noise"
    
    @property
    def category_icon(self) -> str:
        return "📡"
    
    def get_methods(self):
        return {
            "Add Gaussian Noise": EffectMethod(
                name="Add Gaussian Noise",
                description="Add Gaussian noise to image",
                function="noise_gaussian",
                params=[
                    EffectParam("mean", "slider", "Mean", 0, 0, 50, 1),
                    EffectParam("sigma", "slider", "Sigma", 25, 1, 100, 1)
                ]
            ),
            "Add Salt & Pepper Noise": EffectMethod(
                name="Add Salt & Pepper Noise",
                description="Add salt and pepper noise",
                function="noise_sp",
                params=[
                    EffectParam("amount", "slider", "Amount", 0.05, 0.0, 0.5, 0.01)
                ]
            ),
            "Denoise (fastNlMeans)": EffectMethod(
                name="Denoise (fastNlMeans)",
                description="Remove noise using Non-local Means",
                function="cv2.fastNlMeansDenoisingColored",
                params=[
                    EffectParam("h", "slider", "Filter Strength", 10, 1, 20, 1),
                    EffectParam("hColor", "slider", "Color Filter Strength", 10, 1, 20, 1)
                ]
            )
        }
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        
        if method_name == "Add Gaussian Noise":
            mean = params.get("mean", 0)
            sigma = params.get("sigma", 25)
            noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
            result = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif method_name == "Add Salt & Pepper Noise":
            amount = params.get("amount", 0.05)
            result = img.copy()
            num_salt = np.ceil(amount * img.size * 0.5)
            num_pepper = np.ceil(amount * img.size * 0.5)
            
            # Salt
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
            result[coords[0], coords[1], :] = 255
            
            # Pepper
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
            result[coords[0], coords[1], :] = 0
        elif method_name == "Denoise (fastNlMeans)":
            h = int(params.get("h", 10))
            hColor = int(params.get("hColor", 10))
            result = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, 7, 21)
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        if method_name == "Add Gaussian Noise":
            mean = params.get("mean", 0)
            sigma = params.get("sigma", 25)
            code_lines.append(f"noise = np.random.normal({mean}, {sigma}, img.shape).astype(np.float32)")
            code_lines.append("result = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)")
            param_info.append({"function": "gaussian_noise", "params": {"mean": mean, "sigma": sigma}})
        elif method_name == "Add Salt & Pepper Noise":
            amount = params.get("amount", 0.05)
            code_lines.append("result = img.copy()")
            code_lines.append(f"num_salt = int({amount} * img.size * 0.5)")
            code_lines.append("coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]")
            code_lines.append("result[coords[0], coords[1], :] = 255  # Salt")
            code_lines.append("coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]")
            code_lines.append("result[coords[0], coords[1], :] = 0  # Pepper")
            param_info.append({"function": "salt_pepper_noise", "params": {"amount": amount}})
        elif method_name == "Denoise (fastNlMeans)":
            h = int(params.get("h", 10))
            hColor = int(params.get("hColor", 10))
            code_lines.append(f"result = cv2.fastNlMeansDenoisingColored(img, None, {h}, {hColor}, 7, 21)")
            param_info.append({"function": "cv2.fastNlMeansDenoisingColored", "params": {"h": h, "hColor": hColor}})
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
