"""
Thresholding Effects

This module provides thresholding effects including:
- Binary Threshold
- Binary Inverse Threshold
- Truncate Threshold
- To Zero Threshold
- To Zero Inverse Threshold
- Adaptive Threshold Mean
- Adaptive Threshold Gaussian
- Adaptive Threshold Mean Inverse
- Adaptive Threshold Gaussian Inverse
- Otsu's Threshold
- Otsu's Threshold + Gaussian Blur
- Triangle Threshold
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class ThresholdingEffect(BaseEffect):
    """Thresholding effects for images"""
    
    @property
    def category_name(self) -> str:
        return "Thresholding"
    
    @property
    def category_icon(self) -> str:
        return "⚫"
    
    def get_methods(self):
        return {
            "Binary Threshold": EffectMethod(
                name="Binary Threshold",
                description="Apply binary thresholding (THRESH_BINARY)",
                function="cv2.threshold",
                params=[
                    EffectParam("thresh", "slider", "Threshold", 127, 0, 255, 1),
                    EffectParam("maxval", "slider", "Max Value", 255, 0, 255, 1)
                ]
            ),
            "Binary Inverse Threshold": EffectMethod(
                name="Binary Inverse Threshold",
                description="Apply inverse binary thresholding (THRESH_BINARY_INV)",
                function="cv2.threshold",
                params=[
                    EffectParam("thresh", "slider", "Threshold", 127, 0, 255, 1),
                    EffectParam("maxval", "slider", "Max Value", 255, 0, 255, 1)
                ]
            ),
            "Truncate Threshold": EffectMethod(
                name="Truncate Threshold",
                description="Truncate values above threshold (THRESH_TRUNC)",
                function="cv2.threshold",
                params=[
                    EffectParam("thresh", "slider", "Threshold", 127, 0, 255, 1),
                    EffectParam("maxval", "slider", "Max Value", 255, 0, 255, 1)
                ]
            ),
            "To Zero Threshold": EffectMethod(
                name="To Zero Threshold",
                description="Set to zero below threshold (THRESH_TOZERO)",
                function="cv2.threshold",
                params=[
                    EffectParam("thresh", "slider", "Threshold", 127, 0, 255, 1),
                    EffectParam("maxval", "slider", "Max Value", 255, 0, 255, 1)
                ]
            ),
            "To Zero Inverse Threshold": EffectMethod(
                name="To Zero Inverse Threshold",
                description="Set to zero above threshold (THRESH_TOZERO_INV)",
                function="cv2.threshold",
                params=[
                    EffectParam("thresh", "slider", "Threshold", 127, 0, 255, 1),
                    EffectParam("maxval", "slider", "Max Value", 255, 0, 255, 1)
                ]
            ),
            "Adaptive Threshold Mean": EffectMethod(
                name="Adaptive Threshold Mean",
                description="Adaptive thresholding using mean (ADAPTIVE_THRESH_MEAN_C)",
                function="cv2.adaptiveThreshold",
                params=[
                    EffectParam("maxval", "slider", "Max Value", 255, 0, 255, 1),
                    EffectParam("block_size", "slider", "Block Size", 11, 3, 51, 2),
                    EffectParam("C", "slider", "Constant C", 2, -20, 20, 1)
                ]
            ),
            "Adaptive Threshold Gaussian": EffectMethod(
                name="Adaptive Threshold Gaussian",
                description="Adaptive thresholding using Gaussian (ADAPTIVE_THRESH_GAUSSIAN_C)",
                function="cv2.adaptiveThreshold",
                params=[
                    EffectParam("maxval", "slider", "Max Value", 255, 0, 255, 1),
                    EffectParam("block_size", "slider", "Block Size", 11, 3, 51, 2),
                    EffectParam("C", "slider", "Constant C", 2, -20, 20, 1)
                ]
            ),
            "Adaptive Threshold Mean Inverse": EffectMethod(
                name="Adaptive Threshold Mean Inverse",
                description="Adaptive thresholding using mean with inverse (THRESH_BINARY_INV)",
                function="cv2.adaptiveThreshold",
                params=[
                    EffectParam("maxval", "slider", "Max Value", 255, 0, 255, 1),
                    EffectParam("block_size", "slider", "Block Size", 11, 3, 51, 2),
                    EffectParam("C", "slider", "Constant C", 2, -20, 20, 1)
                ]
            ),
            "Adaptive Threshold Gaussian Inverse": EffectMethod(
                name="Adaptive Threshold Gaussian Inverse",
                description="Adaptive thresholding using Gaussian with inverse (THRESH_BINARY_INV)",
                function="cv2.adaptiveThreshold",
                params=[
                    EffectParam("maxval", "slider", "Max Value", 255, 0, 255, 1),
                    EffectParam("block_size", "slider", "Block Size", 11, 3, 51, 2),
                    EffectParam("C", "slider", "Constant C", 2, -20, 20, 1)
                ]
            ),
            "Otsu's Threshold": EffectMethod(
                name="Otsu's Threshold",
                description="Automatic thresholding using Otsu's method",
                function="cv2.threshold",
                params=[
                    EffectParam("maxval", "slider", "Max Value", 255, 0, 255, 1)
                ]
            ),
            "Otsu's Threshold + Gaussian": EffectMethod(
                name="Otsu's Threshold + Gaussian",
                description="Otsu's threshold with Gaussian blur preprocessing",
                function="cv2.threshold",
                params=[
                    EffectParam("maxval", "slider", "Max Value", 255, 0, 255, 1),
                    EffectParam("blur_ksize", "slider", "Blur Kernel Size", 5, 3, 31, 2)
                ]
            ),
            "Triangle Threshold": EffectMethod(
                name="Triangle Threshold",
                description="Automatic thresholding using Triangle method",
                function="cv2.threshold",
                params=[
                    EffectParam("maxval", "slider", "Max Value", 255, 0, 255, 1)
                ]
            )
        }
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if method_name == "Binary Threshold":
            _, result = cv2.threshold(gray, params.get("thresh", 127), params.get("maxval", 255), cv2.THRESH_BINARY)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Binary Inverse Threshold":
            _, result = cv2.threshold(gray, params.get("thresh", 127), params.get("maxval", 255), cv2.THRESH_BINARY_INV)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Truncate Threshold":
            _, result = cv2.threshold(gray, params.get("thresh", 127), params.get("maxval", 255), cv2.THRESH_TRUNC)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "To Zero Threshold":
            _, result = cv2.threshold(gray, params.get("thresh", 127), params.get("maxval", 255), cv2.THRESH_TOZERO)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "To Zero Inverse Threshold":
            _, result = cv2.threshold(gray, params.get("thresh", 127), params.get("maxval", 255), cv2.THRESH_TOZERO_INV)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Adaptive Threshold Mean":
            block_size = self._ensure_odd(int(params.get("block_size", 11)))
            maxval = int(params.get("maxval", 255))
            C = int(params.get("C", 2))
            result = cv2.adaptiveThreshold(gray, maxval, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Adaptive Threshold Gaussian":
            block_size = self._ensure_odd(int(params.get("block_size", 11)))
            maxval = int(params.get("maxval", 255))
            C = int(params.get("C", 2))
            result = cv2.adaptiveThreshold(gray, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Adaptive Threshold Mean Inverse":
            block_size = self._ensure_odd(int(params.get("block_size", 11)))
            maxval = int(params.get("maxval", 255))
            C = int(params.get("C", 2))
            result = cv2.adaptiveThreshold(gray, maxval, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, C)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Adaptive Threshold Gaussian Inverse":
            block_size = self._ensure_odd(int(params.get("block_size", 11)))
            maxval = int(params.get("maxval", 255))
            C = int(params.get("C", 2))
            result = cv2.adaptiveThreshold(gray, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Otsu's Threshold":
            maxval = int(params.get("maxval", 255))
            _, result = cv2.threshold(gray, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Otsu's Threshold + Gaussian":
            maxval = int(params.get("maxval", 255))
            blur_ksize = self._ensure_odd(int(params.get("blur_ksize", 5)))
            blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
            _, result = cv2.threshold(blurred, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Triangle Threshold":
            maxval = int(params.get("maxval", 255))
            _, result = cv2.threshold(gray, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        if method_name == "Binary Threshold":
            thresh = params.get("thresh", 127)
            maxval = params.get("maxval", 255)
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"_, result = cv2.threshold(gray, {thresh}, {maxval}, cv2.THRESH_BINARY)")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.threshold", "params": {"thresh": thresh, "maxval": maxval, "type": "cv2.THRESH_BINARY"}})
        elif method_name == "Binary Inverse Threshold":
            thresh = params.get("thresh", 127)
            maxval = params.get("maxval", 255)
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"_, result = cv2.threshold(gray, {thresh}, {maxval}, cv2.THRESH_BINARY_INV)")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.threshold", "params": {"thresh": thresh, "maxval": maxval, "type": "cv2.THRESH_BINARY_INV"}})
        elif method_name == "Truncate Threshold":
            thresh = params.get("thresh", 127)
            maxval = params.get("maxval", 255)
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"_, result = cv2.threshold(gray, {thresh}, {maxval}, cv2.THRESH_TRUNC)")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.threshold", "params": {"thresh": thresh, "maxval": maxval, "type": "cv2.THRESH_TRUNC"}})
        elif method_name == "To Zero Threshold":
            thresh = params.get("thresh", 127)
            maxval = params.get("maxval", 255)
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"_, result = cv2.threshold(gray, {thresh}, {maxval}, cv2.THRESH_TOZERO)")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.threshold", "params": {"thresh": thresh, "maxval": maxval, "type": "cv2.THRESH_TOZERO"}})
        elif method_name == "To Zero Inverse Threshold":
            thresh = params.get("thresh", 127)
            maxval = params.get("maxval", 255)
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"_, result = cv2.threshold(gray, {thresh}, {maxval}, cv2.THRESH_TOZERO_INV)")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.threshold", "params": {"thresh": thresh, "maxval": maxval, "type": "cv2.THRESH_TOZERO_INV"}})
        elif method_name == "Adaptive Threshold Mean":
            block_size = self._ensure_odd(int(params.get("block_size", 11)))
            maxval = int(params.get("maxval", 255))
            C = int(params.get("C", 2))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"result = cv2.adaptiveThreshold(gray, {maxval}, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, {block_size}, {C})")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.adaptiveThreshold", "params": {"maxValue": maxval, "adaptiveMethod": "ADAPTIVE_THRESH_MEAN_C", "thresholdType": "THRESH_BINARY", "blockSize": block_size, "C": C}})
        elif method_name == "Adaptive Threshold Gaussian":
            block_size = self._ensure_odd(int(params.get("block_size", 11)))
            maxval = int(params.get("maxval", 255))
            C = int(params.get("C", 2))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"result = cv2.adaptiveThreshold(gray, {maxval}, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, {block_size}, {C})")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.adaptiveThreshold", "params": {"maxValue": maxval, "adaptiveMethod": "ADAPTIVE_THRESH_GAUSSIAN_C", "thresholdType": "THRESH_BINARY", "blockSize": block_size, "C": C}})
        elif method_name == "Adaptive Threshold Mean Inverse":
            block_size = self._ensure_odd(int(params.get("block_size", 11)))
            maxval = int(params.get("maxval", 255))
            C = int(params.get("C", 2))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"result = cv2.adaptiveThreshold(gray, {maxval}, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, {block_size}, {C})")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.adaptiveThreshold", "params": {"maxValue": maxval, "adaptiveMethod": "ADAPTIVE_THRESH_MEAN_C", "thresholdType": "THRESH_BINARY_INV", "blockSize": block_size, "C": C}})
        elif method_name == "Adaptive Threshold Gaussian Inverse":
            block_size = self._ensure_odd(int(params.get("block_size", 11)))
            maxval = int(params.get("maxval", 255))
            C = int(params.get("C", 2))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"result = cv2.adaptiveThreshold(gray, {maxval}, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, {block_size}, {C})")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.adaptiveThreshold", "params": {"maxValue": maxval, "adaptiveMethod": "ADAPTIVE_THRESH_GAUSSIAN_C", "thresholdType": "THRESH_BINARY_INV", "blockSize": block_size, "C": C}})
        elif method_name == "Otsu's Threshold":
            maxval = int(params.get("maxval", 255))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"_, result = cv2.threshold(gray, 0, {maxval}, cv2.THRESH_BINARY + cv2.THRESH_OTSU)")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.threshold", "params": {"maxval": maxval, "type": "THRESH_BINARY + THRESH_OTSU"}})
        elif method_name == "Otsu's Threshold + Gaussian":
            maxval = int(params.get("maxval", 255))
            blur_ksize = self._ensure_odd(int(params.get("blur_ksize", 5)))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"blurred = cv2.GaussianBlur(gray, ({blur_ksize}, {blur_ksize}), 0)")
            code_lines.append(f"_, result = cv2.threshold(blurred, 0, {maxval}, cv2.THRESH_BINARY + cv2.THRESH_OTSU)")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.threshold", "params": {"maxval": maxval, "blur_ksize": blur_ksize, "type": "THRESH_BINARY + THRESH_OTSU"}})
        elif method_name == "Triangle Threshold":
            maxval = int(params.get("maxval", 255))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"_, result = cv2.threshold(gray, 0, {maxval}, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.threshold", "params": {"maxval": maxval, "type": "THRESH_BINARY + THRESH_TRIANGLE"}})
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
