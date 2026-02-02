"""
Edge Detection Effects

This module provides edge detection effects including:
- Canny Edge Detection
- Sobel X
- Sobel Y
- Laplacian
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class EdgeDetectionEffect(BaseEffect):
    """Edge detection effects for images"""
    
    @property
    def category_name(self) -> str:
        return "Edge Detection"
    
    @property
    def category_icon(self) -> str:
        return "📐"
    
    def get_methods(self):
        return {
            "Canny": EffectMethod(
                name="Canny",
                description="Canny edge detection algorithm",
                function="cv2.Canny",
                params=[
                    EffectParam("threshold1", "slider", "Threshold 1", 100, 0, 255, 1),
                    EffectParam("threshold2", "slider", "Threshold 2", 200, 0, 255, 1)
                ]
            ),
            "Sobel X": EffectMethod(
                name="Sobel X",
                description="Sobel edge detection in X direction",
                function="cv2.Sobel",
                params=[
                    EffectParam("ksize", "slider", "Kernel Size", 3, 1, 7, 2)
                ]
            ),
            "Sobel Y": EffectMethod(
                name="Sobel Y",
                description="Sobel edge detection in Y direction",
                function="cv2.Sobel",
                params=[
                    EffectParam("ksize", "slider", "Kernel Size", 3, 1, 7, 2)
                ]
            ),
            "Laplacian": EffectMethod(
                name="Laplacian",
                description="Laplacian edge detection (auto depth)",
                function="cv2.Laplacian",
                params=[
                    EffectParam("ksize", "slider", "Kernel Size", 3, 1, 7, 2)
                ]
            ),
            "Laplacian CV_8U": EffectMethod(
                name="Laplacian CV_8U",
                description="Laplacian with 8-bit unsigned depth (may lose negative gradients)",
                function="cv2.Laplacian",
                params=[
                    EffectParam("ksize", "slider", "Kernel Size", 3, 1, 7, 2)
                ]
            ),
            "Laplacian CV_16S": EffectMethod(
                name="Laplacian CV_16S",
                description="Laplacian with 16-bit signed depth (preserves negative gradients)",
                function="cv2.Laplacian",
                params=[
                    EffectParam("ksize", "slider", "Kernel Size", 3, 1, 7, 2)
                ]
            ),
            "Laplacian CV_16U": EffectMethod(
                name="Laplacian CV_16U",
                description="Laplacian with 16-bit unsigned depth",
                function="cv2.Laplacian",
                params=[
                    EffectParam("ksize", "slider", "Kernel Size", 3, 1, 7, 2)
                ]
            ),
            "Laplacian CV_32F": EffectMethod(
                name="Laplacian CV_32F",
                description="Laplacian with 32-bit float depth (high precision)",
                function="cv2.Laplacian",
                params=[
                    EffectParam("ksize", "slider", "Kernel Size", 3, 1, 7, 2),
                    EffectParam("scale", "slider", "Scale", 1.0, 0.1, 5.0, 0.1),
                    EffectParam("delta", "slider", "Delta", 0.0, -100.0, 100.0, 1.0)
                ]
            ),
            "Laplacian CV_64F": EffectMethod(
                name="Laplacian CV_64F",
                description="Laplacian with 64-bit float depth (highest precision)",
                function="cv2.Laplacian",
                params=[
                    EffectParam("ksize", "slider", "Kernel Size", 3, 1, 7, 2),
                    EffectParam("scale", "slider", "Scale", 1.0, 0.1, 5.0, 0.1),
                    EffectParam("delta", "slider", "Delta", 0.0, -100.0, 100.0, 1.0)
                ]
            )
        }
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if method_name == "Canny":
            result = cv2.Canny(gray, params.get("threshold1", 100), params.get("threshold2", 200))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Sobel X":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            result = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            result = np.uint8(np.absolute(result))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Sobel Y":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            result = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            result = np.uint8(np.absolute(result))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Laplacian":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            result = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
            result = np.uint8(np.absolute(result))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Laplacian CV_8U":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            result = cv2.Laplacian(gray, cv2.CV_8U, ksize=ksize)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Laplacian CV_16S":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            result = cv2.Laplacian(gray, cv2.CV_16S, ksize=ksize)
            result = cv2.convertScaleAbs(result)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Laplacian CV_16U":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            result = cv2.Laplacian(gray, cv2.CV_16U, ksize=ksize)
            result = np.uint8(np.clip(result / 256, 0, 255))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Laplacian CV_32F":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            scale = float(params.get("scale", 1.0))
            delta = float(params.get("delta", 0.0))
            result = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize, scale=scale, delta=delta)
            result = cv2.convertScaleAbs(result)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Laplacian CV_64F":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            scale = float(params.get("scale", 1.0))
            delta = float(params.get("delta", 0.0))
            result = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)
            result = cv2.convertScaleAbs(result)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        if method_name == "Canny":
            t1 = params.get("threshold1", 100)
            t2 = params.get("threshold2", 200)
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"edges = cv2.Canny(gray, {t1}, {t2})")
            code_lines.append("result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.Canny", "params": {"threshold1": t1, "threshold2": t2}})
        elif method_name == "Sobel X":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize={ksize})")
            code_lines.append("result = np.uint8(np.absolute(sobel))")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.Sobel", "params": {"ddepth": "cv2.CV_64F", "dx": 1, "dy": 0, "ksize": ksize}})
        elif method_name == "Sobel Y":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize={ksize})")
            code_lines.append("result = np.uint8(np.absolute(sobel))")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.Sobel", "params": {"ddepth": "cv2.CV_64F", "dx": 0, "dy": 1, "ksize": ksize}})
        elif method_name == "Laplacian":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize={ksize})")
            code_lines.append("result = np.uint8(np.absolute(laplacian))")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.Laplacian", "params": {"ddepth": "cv2.CV_64F", "ksize": ksize}})
        elif method_name == "Laplacian CV_8U":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize={ksize})")
            code_lines.append("result = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.Laplacian", "params": {"ddepth": "cv2.CV_8U", "ksize": ksize}})
        elif method_name == "Laplacian CV_16S":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize={ksize})")
            code_lines.append("result = cv2.convertScaleAbs(laplacian)")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.Laplacian", "params": {"ddepth": "cv2.CV_16S", "ksize": ksize}})
        elif method_name == "Laplacian CV_16U":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"laplacian = cv2.Laplacian(gray, cv2.CV_16U, ksize={ksize})")
            code_lines.append("result = np.uint8(np.clip(laplacian / 256, 0, 255))")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.Laplacian", "params": {"ddepth": "cv2.CV_16U", "ksize": ksize}})
        elif method_name == "Laplacian CV_32F":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            scale = float(params.get("scale", 1.0))
            delta = float(params.get("delta", 0.0))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize={ksize}, scale={scale}, delta={delta})")
            code_lines.append("result = cv2.convertScaleAbs(laplacian)")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.Laplacian", "params": {"ddepth": "cv2.CV_32F", "ksize": ksize, "scale": scale, "delta": delta}})
        elif method_name == "Laplacian CV_64F":
            ksize = self._ensure_odd(int(params.get("ksize", 3)))
            scale = float(params.get("scale", 1.0))
            delta = float(params.get("delta", 0.0))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize={ksize}, scale={scale}, delta={delta})")
            code_lines.append("result = cv2.convertScaleAbs(laplacian)")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.Laplacian", "params": {"ddepth": "cv2.CV_64F", "ksize": ksize, "scale": scale, "delta": delta}})
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
