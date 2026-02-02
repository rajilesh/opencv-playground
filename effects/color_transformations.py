"""
Color Transformations Effect

This module provides color transformation effects including:
- BGR to Grayscale / Grayscale to BGR
- BGR to HSV / HSV to BGR
- BGR to LAB / LAB to BGR
- BGR to RGB / RGB to BGR
- BGR to YCrCb / YCrCb to BGR
- BGR to HLS / HLS to BGR
- BGR to XYZ / XYZ to BGR
- BGR to LUV / LUV to BGR
- Color inversion
- Channel extraction and manipulation
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class ColorTransformationsEffect(BaseEffect):
    """Color transformation effects for images"""
    
    @property
    def category_name(self) -> str:
        return "Color Transformations"
    
    @property
    def category_icon(self) -> str:
        return "🎨"
    
    def get_methods(self):
        return {
            # Grayscale conversions
            "BGR to Grayscale": EffectMethod(
                name="BGR to Grayscale",
                description="Convert BGR image to grayscale (single channel)",
                function="cv2.cvtColor",
                params=[
                    EffectParam("output_3ch", "checkbox", "Output as 3-channel", True)
                ]
            ),
            "Grayscale to BGR": EffectMethod(
                name="Grayscale to BGR",
                description="Convert grayscale image to BGR (3 channels)",
                function="cv2.cvtColor",
                params=[]
            ),
            
            # HSV conversions
            "BGR to HSV": EffectMethod(
                name="BGR to HSV",
                description="Convert BGR to HSV (Hue, Saturation, Value)",
                function="cv2.cvtColor",
                params=[]
            ),
            "HSV to BGR": EffectMethod(
                name="HSV to BGR",
                description="Convert HSV back to BGR",
                function="cv2.cvtColor",
                params=[]
            ),
            
            # RGB conversions
            "BGR to RGB": EffectMethod(
                name="BGR to RGB",
                description="Convert BGR to RGB color order",
                function="cv2.cvtColor",
                params=[]
            ),
            "RGB to BGR": EffectMethod(
                name="RGB to BGR",
                description="Convert RGB back to BGR color order",
                function="cv2.cvtColor",
                params=[]
            ),
            
            # LAB conversions
            "BGR to LAB": EffectMethod(
                name="BGR to LAB",
                description="Convert BGR to LAB color space (L*a*b*)",
                function="cv2.cvtColor",
                params=[]
            ),
            "LAB to BGR": EffectMethod(
                name="LAB to BGR",
                description="Convert LAB back to BGR",
                function="cv2.cvtColor",
                params=[]
            ),
            
            # YCrCb conversions
            "BGR to YCrCb": EffectMethod(
                name="BGR to YCrCb",
                description="Convert BGR to YCrCb (luma and chroma)",
                function="cv2.cvtColor",
                params=[]
            ),
            "YCrCb to BGR": EffectMethod(
                name="YCrCb to BGR",
                description="Convert YCrCb back to BGR",
                function="cv2.cvtColor",
                params=[]
            ),
            
            # HLS conversions
            "BGR to HLS": EffectMethod(
                name="BGR to HLS",
                description="Convert BGR to HLS (Hue, Lightness, Saturation)",
                function="cv2.cvtColor",
                params=[]
            ),
            "HLS to BGR": EffectMethod(
                name="HLS to BGR",
                description="Convert HLS back to BGR",
                function="cv2.cvtColor",
                params=[]
            ),
            
            # XYZ conversions
            "BGR to XYZ": EffectMethod(
                name="BGR to XYZ",
                description="Convert BGR to CIE XYZ color space",
                function="cv2.cvtColor",
                params=[]
            ),
            "XYZ to BGR": EffectMethod(
                name="XYZ to BGR",
                description="Convert XYZ back to BGR",
                function="cv2.cvtColor",
                params=[]
            ),
            
            # LUV conversions
            "BGR to LUV": EffectMethod(
                name="BGR to LUV",
                description="Convert BGR to CIE LUV color space",
                function="cv2.cvtColor",
                params=[]
            ),
            "LUV to BGR": EffectMethod(
                name="LUV to BGR",
                description="Convert LUV back to BGR",
                function="cv2.cvtColor",
                params=[]
            ),
            
            # YUV conversions
            "BGR to YUV": EffectMethod(
                name="BGR to YUV",
                description="Convert BGR to YUV color space",
                function="cv2.cvtColor",
                params=[]
            ),
            "YUV to BGR": EffectMethod(
                name="YUV to BGR",
                description="Convert YUV back to BGR",
                function="cv2.cvtColor",
                params=[]
            ),
            
            # Generic color conversion
            "Custom Color Conversion": EffectMethod(
                name="Custom Color Conversion",
                description="Apply any OpenCV color conversion",
                function="cv2.cvtColor",
                params=[
                    EffectParam("conversion", "selectbox", "Color Conversion", "BGR2GRAY", None, None, None,
                               ["BGR2GRAY", "GRAY2BGR", "BGR2RGB", "RGB2BGR", 
                                "BGR2HSV", "HSV2BGR", "BGR2HSV_FULL", "HSV2BGR_FULL",
                                "BGR2HLS", "HLS2BGR", "BGR2HLS_FULL", "HLS2BGR_FULL",
                                "BGR2LAB", "LAB2BGR", "BGR2LUV", "LUV2BGR",
                                "BGR2XYZ", "XYZ2BGR", "BGR2YCrCb", "YCrCb2BGR",
                                "BGR2YUV", "YUV2BGR", "BGR2YUV_I420", "BGR2YUV_IYUV",
                                "RGBA2BGRA", "BGRA2RGBA", "BGR2BGRA", "BGRA2BGR",
                                "RGB2RGBA", "RGBA2RGB", "BGR2RGBA", "RGBA2BGR"]),
                    EffectParam("output_3ch", "checkbox", "Ensure 3-channel output", False)
                ]
            ),
            
            # Channel operations
            "Extract Channel": EffectMethod(
                name="Extract Channel",
                description="Extract a single channel from the image",
                function="cv2.extractChannel",
                params=[
                    EffectParam("channel", "selectbox", "Channel", "0 (Blue)", None, None, None,
                               ["0 (Blue)", "1 (Green)", "2 (Red)"]),
                    EffectParam("output_3ch", "checkbox", "Output as 3-channel", True)
                ]
            ),
            "Merge Channels": EffectMethod(
                name="Merge Channels",
                description="Swap or rearrange color channels",
                function="cv2.merge",
                params=[
                    EffectParam("arrangement", "selectbox", "Channel Arrangement", "BGR (Original)", None, None, None,
                               ["BGR (Original)", "RGB", "GBR", "GRB", "RBG", "BRG",
                                "BBB (Blue only)", "GGG (Green only)", "RRR (Red only)"])
                ]
            ),
            
            # Color inversion
            "Invert Colors": EffectMethod(
                name="Invert Colors",
                description="Invert all colors in the image (negative)",
                function="cv2.bitwise_not",
                params=[]
            ),
            
            # Sepia effect
            "Sepia Tone": EffectMethod(
                name="Sepia Tone",
                description="Apply sepia tone effect",
                function="sepia",
                params=[
                    EffectParam("intensity", "slider", "Intensity", 100, 0, 100, 5)
                ]
            ),
            
            # Color balance
            "Adjust Color Balance": EffectMethod(
                name="Adjust Color Balance",
                description="Adjust individual color channel levels",
                function="color_balance",
                params=[
                    EffectParam("blue", "slider", "Blue", 100, 0, 200, 1),
                    EffectParam("green", "slider", "Green", 100, 0, 200, 1),
                    EffectParam("red", "slider", "Red", 100, 0, 200, 1)
                ]
            ),
            
            # Saturation adjustment
            "Adjust Saturation": EffectMethod(
                name="Adjust Saturation",
                description="Adjust color saturation",
                function="saturation",
                params=[
                    EffectParam("saturation", "slider", "Saturation", 100, 0, 200, 1)
                ]
            ),
            
            # Hue shift
            "Shift Hue": EffectMethod(
                name="Shift Hue",
                description="Shift hue values",
                function="hue_shift",
                params=[
                    EffectParam("shift", "slider", "Hue Shift", 0, -180, 180, 1)
                ]
            )
        }
    
    def _get_conversion_code(self, name: str):
        """Get OpenCV color conversion code"""
        conversions = {
            "BGR2GRAY": cv2.COLOR_BGR2GRAY,
            "GRAY2BGR": cv2.COLOR_GRAY2BGR,
            "BGR2RGB": cv2.COLOR_BGR2RGB,
            "RGB2BGR": cv2.COLOR_RGB2BGR,
            "BGR2HSV": cv2.COLOR_BGR2HSV,
            "HSV2BGR": cv2.COLOR_HSV2BGR,
            "BGR2HSV_FULL": cv2.COLOR_BGR2HSV_FULL,
            "HSV2BGR_FULL": cv2.COLOR_HSV2BGR_FULL,
            "BGR2HLS": cv2.COLOR_BGR2HLS,
            "HLS2BGR": cv2.COLOR_HLS2BGR,
            "BGR2HLS_FULL": cv2.COLOR_BGR2HLS_FULL,
            "HLS2BGR_FULL": cv2.COLOR_HLS2BGR_FULL,
            "BGR2LAB": cv2.COLOR_BGR2LAB,
            "LAB2BGR": cv2.COLOR_LAB2BGR,
            "BGR2LUV": cv2.COLOR_BGR2LUV,
            "LUV2BGR": cv2.COLOR_LUV2BGR,
            "BGR2XYZ": cv2.COLOR_BGR2XYZ,
            "XYZ2BGR": cv2.COLOR_XYZ2BGR,
            "BGR2YCrCb": cv2.COLOR_BGR2YCrCb,
            "YCrCb2BGR": cv2.COLOR_YCrCb2BGR,
            "BGR2YUV": cv2.COLOR_BGR2YUV,
            "YUV2BGR": cv2.COLOR_YUV2BGR,
            "BGR2YUV_I420": cv2.COLOR_BGR2YUV_I420,
            "BGR2YUV_IYUV": cv2.COLOR_BGR2YUV_IYUV,
            "RGBA2BGRA": cv2.COLOR_RGBA2BGRA,
            "BGRA2RGBA": cv2.COLOR_BGRA2RGBA,
            "BGR2BGRA": cv2.COLOR_BGR2BGRA,
            "BGRA2BGR": cv2.COLOR_BGRA2BGR,
            "RGB2RGBA": cv2.COLOR_RGB2RGBA,
            "RGBA2RGB": cv2.COLOR_RGBA2RGB,
            "BGR2RGBA": cv2.COLOR_BGR2RGBA,
            "RGBA2BGR": cv2.COLOR_RGBA2BGR,
        }
        return conversions.get(name, None)
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        
        # Handle grayscale input - convert to BGR first if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        if method_name == "BGR to Grayscale":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if params.get("output_3ch", True):
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        elif method_name == "Grayscale to BGR":
            if len(img.shape) == 2:
                result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                # Already 3 channel, convert to gray first then back
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        elif method_name == "BGR to HSV":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        elif method_name == "HSV to BGR":
            result = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        
        elif method_name == "BGR to RGB":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        elif method_name == "RGB to BGR":
            result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        elif method_name == "BGR to LAB":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        elif method_name == "LAB to BGR":
            result = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        
        elif method_name == "BGR to YCrCb":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        elif method_name == "YCrCb to BGR":
            result = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        
        elif method_name == "BGR to HLS":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        
        elif method_name == "HLS to BGR":
            result = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)
        
        elif method_name == "BGR to XYZ":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
        
        elif method_name == "XYZ to BGR":
            result = cv2.cvtColor(img, cv2.COLOR_XYZ2BGR)
        
        elif method_name == "BGR to LUV":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        
        elif method_name == "LUV to BGR":
            result = cv2.cvtColor(img, cv2.COLOR_LUV2BGR)
        
        elif method_name == "BGR to YUV":
            result = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        
        elif method_name == "YUV to BGR":
            result = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        
        elif method_name == "Custom Color Conversion":
            conversion = params.get("conversion", "BGR2GRAY")
            code = self._get_conversion_code(conversion)
            if code is not None:
                try:
                    result = cv2.cvtColor(img, code)
                    if params.get("output_3ch", False) and len(result.shape) == 2:
                        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                except cv2.error:
                    result = img
            else:
                result = img
        
        elif method_name == "Extract Channel":
            channel_str = params.get("channel", "0 (Blue)")
            channel = int(channel_str.split()[0])
            if len(img.shape) == 3 and img.shape[2] >= channel + 1:
                result = img[:, :, channel]
                if params.get("output_3ch", True):
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            else:
                result = img
        
        elif method_name == "Merge Channels":
            arrangement = params.get("arrangement", "BGR (Original)")
            if len(img.shape) == 3:
                b, g, r = cv2.split(img)
                if arrangement == "BGR (Original)":
                    result = img
                elif arrangement == "RGB":
                    result = cv2.merge([r, g, b])
                elif arrangement == "GBR":
                    result = cv2.merge([g, b, r])
                elif arrangement == "GRB":
                    result = cv2.merge([g, r, b])
                elif arrangement == "RBG":
                    result = cv2.merge([r, b, g])
                elif arrangement == "BRG":
                    result = cv2.merge([b, r, g])
                elif arrangement == "BBB (Blue only)":
                    result = cv2.merge([b, b, b])
                elif arrangement == "GGG (Green only)":
                    result = cv2.merge([g, g, g])
                elif arrangement == "RRR (Red only)":
                    result = cv2.merge([r, r, r])
                else:
                    result = img
            else:
                result = img
        
        elif method_name == "Invert Colors":
            result = cv2.bitwise_not(img)
        
        elif method_name == "Sepia Tone":
            intensity = params.get("intensity", 100) / 100.0
            # Sepia kernel
            kernel = np.array([
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]
            ])
            sepia = cv2.transform(img, kernel)
            sepia = np.clip(sepia, 0, 255).astype(np.uint8)
            result = cv2.addWeighted(img, 1 - intensity, sepia, intensity, 0)
        
        elif method_name == "Adjust Color Balance":
            blue = params.get("blue", 100) / 100.0
            green = params.get("green", 100) / 100.0
            red = params.get("red", 100) / 100.0
            
            result = img.astype(np.float32)
            result[:, :, 0] = np.clip(result[:, :, 0] * blue, 0, 255)
            result[:, :, 1] = np.clip(result[:, :, 1] * green, 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * red, 0, 255)
            result = result.astype(np.uint8)
        
        elif method_name == "Adjust Saturation":
            saturation = params.get("saturation", 100) / 100.0
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            hsv = hsv.astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        elif method_name == "Shift Hue":
            shift = int(params.get("shift", 0))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + shift) % 180
            hsv = hsv.astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        if method_name == "BGR to Grayscale":
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            if params.get("output_3ch", True):
                code_lines.append("result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)")
            else:
                code_lines.append("result = gray")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_BGR2GRAY"}})
        
        elif method_name == "Grayscale to BGR":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_GRAY2BGR"}})
        
        elif method_name == "BGR to HSV":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_BGR2HSV"}})
        
        elif method_name == "HSV to BGR":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_HSV2BGR"}})
        
        elif method_name == "BGR to RGB":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_BGR2RGB"}})
        
        elif method_name == "RGB to BGR":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_RGB2BGR"}})
        
        elif method_name == "BGR to LAB":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_BGR2LAB"}})
        
        elif method_name == "LAB to BGR":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_LAB2BGR"}})
        
        elif method_name == "BGR to YCrCb":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_BGR2YCrCb"}})
        
        elif method_name == "YCrCb to BGR":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_YCrCb2BGR"}})
        
        elif method_name == "BGR to HLS":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_BGR2HLS"}})
        
        elif method_name == "HLS to BGR":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_HLS2BGR"}})
        
        elif method_name == "BGR to XYZ":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_BGR2XYZ"}})
        
        elif method_name == "XYZ to BGR":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_XYZ2BGR)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_XYZ2BGR"}})
        
        elif method_name == "BGR to LUV":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_BGR2LUV"}})
        
        elif method_name == "LUV to BGR":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_LUV2BGR)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_LUV2BGR"}})
        
        elif method_name == "BGR to YUV":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_BGR2YUV"}})
        
        elif method_name == "YUV to BGR":
            code_lines.append("result = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": "cv2.COLOR_YUV2BGR"}})
        
        elif method_name == "Custom Color Conversion":
            conversion = params.get("conversion", "BGR2GRAY")
            code_lines.append(f"result = cv2.cvtColor(img, cv2.COLOR_{conversion})")
            if params.get("output_3ch", False):
                code_lines.append("if len(result.shape) == 2:")
                code_lines.append("    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "cv2.cvtColor", "params": {"code": f"cv2.COLOR_{conversion}"}})
        
        elif method_name == "Extract Channel":
            channel_str = params.get("channel", "0 (Blue)")
            channel = int(channel_str.split()[0])
            code_lines.append(f"result = img[:, :, {channel}]")
            if params.get("output_3ch", True):
                code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "channel extraction", "params": {"channel": channel}})
        
        elif method_name == "Merge Channels":
            arrangement = params.get("arrangement", "BGR (Original)")
            code_lines.append("b, g, r = cv2.split(img)")
            if arrangement == "RGB":
                code_lines.append("result = cv2.merge([r, g, b])")
            elif arrangement == "GBR":
                code_lines.append("result = cv2.merge([g, b, r])")
            elif arrangement == "GRB":
                code_lines.append("result = cv2.merge([g, r, b])")
            elif arrangement == "RBG":
                code_lines.append("result = cv2.merge([r, b, g])")
            elif arrangement == "BRG":
                code_lines.append("result = cv2.merge([b, r, g])")
            elif "Blue only" in arrangement:
                code_lines.append("result = cv2.merge([b, b, b])")
            elif "Green only" in arrangement:
                code_lines.append("result = cv2.merge([g, g, g])")
            elif "Red only" in arrangement:
                code_lines.append("result = cv2.merge([r, r, r])")
            else:
                code_lines.append("result = img")
            param_info.append({"function": "cv2.merge", "params": {"arrangement": arrangement}})
        
        elif method_name == "Invert Colors":
            code_lines.append("result = cv2.bitwise_not(img)")
            param_info.append({"function": "cv2.bitwise_not", "params": {}})
        
        elif method_name == "Sepia Tone":
            intensity = params.get("intensity", 100) / 100.0
            code_lines.append("# Sepia tone kernel")
            code_lines.append("kernel = np.array([")
            code_lines.append("    [0.272, 0.534, 0.131],")
            code_lines.append("    [0.349, 0.686, 0.168],")
            code_lines.append("    [0.393, 0.769, 0.189]")
            code_lines.append("])")
            code_lines.append("sepia = cv2.transform(img, kernel)")
            code_lines.append("sepia = np.clip(sepia, 0, 255).astype(np.uint8)")
            code_lines.append(f"result = cv2.addWeighted(img, {1 - intensity:.2f}, sepia, {intensity:.2f}, 0)")
            param_info.append({"function": "cv2.transform", "params": {"intensity": intensity}})
        
        elif method_name == "Adjust Color Balance":
            blue = params.get("blue", 100) / 100.0
            green = params.get("green", 100) / 100.0
            red = params.get("red", 100) / 100.0
            code_lines.append("result = img.astype(np.float32)")
            code_lines.append(f"result[:, :, 0] = np.clip(result[:, :, 0] * {blue:.2f}, 0, 255)  # Blue")
            code_lines.append(f"result[:, :, 1] = np.clip(result[:, :, 1] * {green:.2f}, 0, 255)  # Green")
            code_lines.append(f"result[:, :, 2] = np.clip(result[:, :, 2] * {red:.2f}, 0, 255)  # Red")
            code_lines.append("result = result.astype(np.uint8)")
            param_info.append({"function": "color_balance", "params": {"blue": blue, "green": green, "red": red}})
        
        elif method_name == "Adjust Saturation":
            saturation = params.get("saturation", 100) / 100.0
            code_lines.append("hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)")
            code_lines.append(f"hsv[:, :, 1] = np.clip(hsv[:, :, 1] * {saturation:.2f}, 0, 255)")
            code_lines.append("hsv = hsv.astype(np.uint8)")
            code_lines.append("result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)")
            param_info.append({"function": "saturation", "params": {"saturation": saturation}})
        
        elif method_name == "Shift Hue":
            shift = int(params.get("shift", 0))
            code_lines.append("hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)")
            code_lines.append(f"hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + {shift}) % 180")
            code_lines.append("hsv = hsv.astype(np.uint8)")
            code_lines.append("result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)")
            param_info.append({"function": "hue_shift", "params": {"shift": shift}})
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
