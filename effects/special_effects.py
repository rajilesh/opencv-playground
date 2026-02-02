"""
Special Effects

This module provides special artistic effects including:
- Sharpen
- Emboss
- Sketch Effect
- Cartoon Effect
- Sepia
- Vignette
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam
from effects import utils


class SpecialEffectsEffect(BaseEffect):
    """Special artistic effects for images"""
    
    @property
    def category_name(self) -> str:
        return "Special Effects"
    
    @property
    def category_icon(self) -> str:
        return "✨"
    
    def get_methods(self):
        return {
            "Sharpen": EffectMethod(
                name="Sharpen",
                description="Sharpen the image",
                function="cv2.filter2D",
                params=[
                    EffectParam("strength", "slider", "Strength", 1.0, 0.0, 2.0, 0.1)
                ]
            ),
            "Emboss": EffectMethod(
                name="Emboss",
                description="Apply emboss effect",
                function="cv2.filter2D",
                params=[]
            ),
            "Sketch Effect": EffectMethod(
                name="Sketch Effect",
                description="Convert image to pencil sketch",
                function="sketch",
                params=[
                    EffectParam("blur_sigma", "slider", "Blur Sigma", 21, 1, 100, 2)
                ]
            ),
            "Cartoon Effect": EffectMethod(
                name="Cartoon Effect",
                description="Apply cartoon effect to image",
                function="cartoon",
                params=[
                    EffectParam("num_colors", "slider", "Number of Colors", 9, 2, 20, 1),
                    EffectParam("blur_value", "slider", "Blur Value", 7, 1, 21, 2)
                ]
            ),
            "Cartoon Effect (Bilateral)": EffectMethod(
                name="Cartoon Effect (Bilateral)",
                description="Cartoon effect using bilateral filtering with iterations",
                function="cartoon_bilateral",
                params=[
                    EffectParam("iterations", "slider", "Bilateral Iterations", 5, 1, 10, 1),
                    EffectParam("d", "slider", "Filter Diameter", 9, 3, 15, 2),
                    EffectParam("sigma_color", "slider", "Sigma Color", 75, 10, 150, 5),
                    EffectParam("sigma_space", "slider", "Sigma Space", 75, 10, 150, 5),
                    EffectParam("edge_block_size", "slider", "Edge Block Size", 9, 3, 15, 2),
                    EffectParam("edge_c", "slider", "Edge C", 2, 1, 10, 1)
                ]
            ),
            "Cartoon Effect (Laplacian)": EffectMethod(
                name="Cartoon Effect (Laplacian)",
                description="Cartoon effect using Laplacian edge detection",
                function="cartoon_laplacian",
                params=[
                    EffectParam("blur_ksize", "slider", "Blur Kernel Size", 5, 3, 15, 2),
                    EffectParam("laplacian_ksize", "slider", "Laplacian Kernel Size", 5, 1, 7, 2),
                    EffectParam("edge_threshold", "slider", "Edge Threshold", 50, 10, 150, 5),
                    EffectParam("bilateral_iterations", "slider", "Bilateral Iterations", 5, 1, 15, 1),
                    EffectParam("d", "slider", "Filter Diameter", 9, 3, 15, 2),
                    EffectParam("sigma_color", "slider", "Sigma Color", 75, 10, 150, 5),
                    EffectParam("sigma_space", "slider", "Sigma Space", 75, 10, 150, 5)
                ]
            ),
            "Sepia": EffectMethod(
                name="Sepia",
                description="Apply sepia tone effect",
                function="sepia",
                params=[]
            ),
            "Vignette": EffectMethod(
                name="Vignette",
                description="Add vignette effect",
                function="vignette",
                params=[
                    EffectParam("strength", "slider", "Strength", 0.5, 0.0, 1.0, 0.05)
                ]
            ),
            "Smooth Colors (Bilateral)": EffectMethod(
                name="Smooth Colors (Bilateral)",
                description="Smooth colors using iterative bilateral filtering",
                function="smooth_bilateral",
                params=[
                    EffectParam("iterations", "slider", "Iterations", 5, 1, 15, 1),
                    EffectParam("d", "slider", "Filter Diameter", 9, 3, 15, 2),
                    EffectParam("sigma_color", "slider", "Sigma Color", 75, 10, 150, 5),
                    EffectParam("sigma_space", "slider", "Sigma Space", 75, 10, 150, 5)
                ]
            )
        }
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        
        if method_name == "Sharpen":
            strength = params.get("strength", 1.0)
            kernel = np.array([[-1, -1, -1],
                              [-1, 9 + strength, -1],
                              [-1, -1, -1]])
            result = cv2.filter2D(img, -1, kernel)
        elif method_name == "Emboss":
            kernel = np.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])
            result = cv2.filter2D(img, -1, kernel)
        elif method_name == "Sketch Effect":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inv_gray = 255 - gray
            blur_sigma = self._ensure_odd(int(params.get("blur_sigma", 21)))
            blur = cv2.GaussianBlur(inv_gray, (blur_sigma, blur_sigma), 0)
            result = cv2.divide(gray, 255 - blur, scale=256)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif method_name == "Cartoon Effect":
            num_colors = int(params.get("num_colors", 9))
            blur_val = self._ensure_odd(int(params.get("blur_value", 7)))
            
            # Use reusable utilities
            edges = utils.adaptive_edges(img, blur_ksize=blur_val, block_size=9, c=9)
            quantized = utils.quantize_colors(img, num_colors=num_colors)
            result = utils.combine_with_edges(quantized, edges)
        elif method_name == "Cartoon Effect (Bilateral)":
            iterations = int(params.get("iterations", 5))
            d = int(params.get("d", 9))
            sigma_color = int(params.get("sigma_color", 75))
            sigma_space = int(params.get("sigma_space", 75))
            edge_block_size = self._ensure_odd(int(params.get("edge_block_size", 9)))
            edge_c = int(params.get("edge_c", 2))
            
            # Use reusable utilities
            edges = utils.adaptive_edges(img, blur_ksize=5, block_size=edge_block_size, c=edge_c)
            color = utils.bilateral_smooth(img, iterations=iterations, d=d, 
                                           sigma_color=sigma_color, sigma_space=sigma_space)
            result = utils.combine_with_edges(color, edges)
        elif method_name == "Cartoon Effect (Laplacian)":
            blur_ksize = self._ensure_odd(int(params.get("blur_ksize", 5)))
            laplacian_ksize = self._ensure_odd(int(params.get("laplacian_ksize", 5)))
            edge_threshold = int(params.get("edge_threshold", 50))
            iterations = int(params.get("bilateral_iterations", 5))
            d = int(params.get("d", 9))
            sigma_color = int(params.get("sigma_color", 75))
            sigma_space = int(params.get("sigma_space", 75))
            
            # Use reusable utilities
            edges = utils.laplacian_edges(img, blur_ksize=blur_ksize, laplacian_ksize=laplacian_ksize,
                                          threshold=edge_threshold, invert=True)
            color = utils.bilateral_smooth(img, iterations=iterations, d=d,
                                           sigma_color=sigma_color, sigma_space=sigma_space)
            result = utils.combine_with_edges(color, edges)
        elif method_name == "Sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            result = cv2.transform(img, kernel)
            result = np.clip(result, 0, 255).astype(np.uint8)
        elif method_name == "Vignette":
            strength = params.get("strength", 0.5)
            rows, cols = img.shape[:2]
            X = np.arange(0, cols)
            Y = np.arange(0, rows)
            X, Y = np.meshgrid(X, Y)
            centerX, centerY = cols / 2, rows / 2
            mask = np.sqrt((X - centerX) ** 2 + (Y - centerY) ** 2)
            mask = mask / mask.max()
            mask = 1 - mask * strength
            mask = np.dstack([mask] * 3)
            result = (img * mask).astype(np.uint8)
        elif method_name == "Smooth Colors (Bilateral)":
            iterations = int(params.get("iterations", 5))
            d = int(params.get("d", 9))
            sigma_color = int(params.get("sigma_color", 75))
            sigma_space = int(params.get("sigma_space", 75))
            
            # Use reusable utility
            result = utils.bilateral_smooth(img, iterations=iterations, d=d,
                                            sigma_color=sigma_color, sigma_space=sigma_space)
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        if method_name == "Sharpen":
            strength = params.get("strength", 1.0)
            code_lines.append(f"kernel = np.array([[-1, -1, -1], [-1, 9 + {strength}, -1], [-1, -1, -1]])")
            code_lines.append("result = cv2.filter2D(img, -1, kernel)")
            param_info.append({"function": "cv2.filter2D", "params": {"strength": strength}})
        elif method_name == "Emboss":
            code_lines.append("kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])")
            code_lines.append("result = cv2.filter2D(img, -1, kernel)")
            param_info.append({"function": "cv2.filter2D", "params": {"effect": "emboss"}})
        elif method_name == "Sketch Effect":
            blur_sigma = self._ensure_odd(int(params.get("blur_sigma", 21)))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append("inv_gray = 255 - gray")
            code_lines.append(f"blur = cv2.GaussianBlur(inv_gray, ({blur_sigma}, {blur_sigma}), 0)")
            code_lines.append("result = cv2.divide(gray, 255 - blur, scale=256)")
            code_lines.append("result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)")
            param_info.append({"function": "sketch_effect", "params": {"blur_sigma": blur_sigma}})
        elif method_name == "Cartoon Effect":
            num_colors = int(params.get("num_colors", 9))
            blur_val = self._ensure_odd(int(params.get("blur_value", 7)))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"gray = cv2.medianBlur(gray, {blur_val})")
            code_lines.append("edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)")
            code_lines.append("data = np.float32(img).reshape((-1, 3))")
            code_lines.append("criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)")
            code_lines.append(f"_, labels, centers = cv2.kmeans(data, {num_colors}, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)")
            code_lines.append("centers = np.uint8(centers)")
            code_lines.append("quantized = centers[labels.flatten()].reshape(img.shape)")
            code_lines.append("result = cv2.bitwise_and(quantized, quantized, mask=edges)")
            param_info.append({"function": "cartoon_effect", "params": {"num_colors": num_colors, "blur_value": blur_val}})
        elif method_name == "Cartoon Effect (Bilateral)":
            iterations = int(params.get("iterations", 5))
            d = int(params.get("d", 9))
            sigma_color = int(params.get("sigma_color", 75))
            sigma_space = int(params.get("sigma_space", 75))
            edge_block_size = self._ensure_odd(int(params.get("edge_block_size", 9)))
            edge_c = int(params.get("edge_c", 2))
            code_lines.append("# Edge detection")
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append("gray = cv2.medianBlur(gray, 5)")
            code_lines.append(f"edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, {edge_block_size}, {edge_c})")
            code_lines.append("")
            code_lines.append("# Smooth colors using bilateral filtering")
            code_lines.append("color = img.copy()")
            code_lines.append(f"for _ in range({iterations}):")
            code_lines.append(f"    color = cv2.bilateralFilter(color, d={d}, sigmaColor={sigma_color}, sigmaSpace={sigma_space})")
            code_lines.append("")
            code_lines.append("# Combine edges with color image")
            code_lines.append("result = cv2.bitwise_and(color, color, mask=edges)")
            param_info.append({"function": "cartoon_bilateral", "params": {"iterations": iterations, "d": d, "sigma_color": sigma_color, "sigma_space": sigma_space}})
        elif method_name == "Cartoon Effect (Laplacian)":
            blur_ksize = self._ensure_odd(int(params.get("blur_ksize", 5)))
            laplacian_ksize = self._ensure_odd(int(params.get("laplacian_ksize", 5)))
            edge_threshold = int(params.get("edge_threshold", 50))
            iterations = int(params.get("bilateral_iterations", 5))
            d = int(params.get("d", 9))
            sigma_color = int(params.get("sigma_color", 75))
            sigma_space = int(params.get("sigma_space", 75))
            code_lines.append("# Prepare grayscale and blur")
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"gray_blur = cv2.GaussianBlur(gray, ({blur_ksize}, {blur_ksize}), 0)")
            code_lines.append("")
            code_lines.append("# Laplacian edge detection")
            code_lines.append(f"edges = cv2.Laplacian(gray_blur, cv2.CV_8U, ksize={laplacian_ksize})")
            code_lines.append("")
            code_lines.append("# Invert edges (white background, black edges)")
            code_lines.append(f"edges = cv2.threshold(edges, {edge_threshold}, 255, cv2.THRESH_BINARY_INV)[1]")
            code_lines.append("")
            code_lines.append("# Smooth colors using bilateral filtering")
            code_lines.append("color = img.copy()")
            code_lines.append(f"for _ in range({iterations}):")
            code_lines.append(f"    color = cv2.bilateralFilter(color, d={d}, sigmaColor={sigma_color}, sigmaSpace={sigma_space})")
            code_lines.append("")
            code_lines.append("# Combine edges with color image")
            code_lines.append("result = cv2.bitwise_and(color, color, mask=edges)")
            param_info.append({"function": "cartoon_laplacian", "params": {"blur_ksize": blur_ksize, "laplacian_ksize": laplacian_ksize, "edge_threshold": edge_threshold, "iterations": iterations}})
        elif method_name == "Sepia":
            code_lines.append("kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])")
            code_lines.append("result = cv2.transform(img, kernel)")
            code_lines.append("result = np.clip(result, 0, 255).astype(np.uint8)")
            param_info.append({"function": "cv2.transform", "params": {"effect": "sepia"}})
        elif method_name == "Vignette":
            strength = params.get("strength", 0.5)
            code_lines.append("rows, cols = img.shape[:2]")
            code_lines.append("X, Y = np.meshgrid(np.arange(cols), np.arange(rows))")
            code_lines.append("centerX, centerY = cols / 2, rows / 2")
            code_lines.append("mask = np.sqrt((X - centerX) ** 2 + (Y - centerY) ** 2)")
            code_lines.append("mask = mask / mask.max()")
            code_lines.append(f"mask = 1 - mask * {strength}")
            code_lines.append("mask = np.dstack([mask] * 3)")
            code_lines.append("result = (img * mask).astype(np.uint8)")
            param_info.append({"function": "vignette_effect", "params": {"strength": strength}})
        elif method_name == "Smooth Colors (Bilateral)":
            iterations = int(params.get("iterations", 5))
            d = int(params.get("d", 9))
            sigma_color = int(params.get("sigma_color", 75))
            sigma_space = int(params.get("sigma_space", 75))
            code_lines.append("# Smooth colors using bilateral filtering")
            code_lines.append("color = img.copy()")
            code_lines.append(f"for _ in range({iterations}):")
            code_lines.append(f"    color = cv2.bilateralFilter(color, d={d}, sigmaColor={sigma_color}, sigmaSpace={sigma_space})")
            code_lines.append("result = color")
            param_info.append({"function": "cv2.bilateralFilter", "params": {"iterations": iterations, "d": d, "sigmaColor": sigma_color, "sigmaSpace": sigma_space}})
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
