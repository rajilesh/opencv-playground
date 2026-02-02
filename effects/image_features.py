"""
Image Features Effects

This module provides feature detection and matching effects:
- SIFT Features
- ORB Features
- Harris Corner Detection
- Shi-Tomasi Corners (Good Features to Track)
- FAST Features
- Feature Matching (between two images)
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class ImageFeaturesEffect(BaseEffect):
    """Image feature detection effects"""
    
    @property
    def category_name(self) -> str:
        return "Image Features"
    
    @property
    def category_icon(self) -> str:
        return "🎯"
    
    def get_methods(self):
        return {
            "SIFT Features": EffectMethod(
                name="SIFT Features",
                description="Detect SIFT (Scale-Invariant Feature Transform) keypoints",
                function="cv2.SIFT_create",
                params=[
                    EffectParam("nfeatures", "slider", "Max Features", 500, 10, 2000, 10),
                    EffectParam("draw_rich", "dropdown", "Draw Style", "Rich", options=["Simple", "Rich"]),
                    EffectParam("color_r", "slider", "Color R", 0, 0, 255, 1),
                    EffectParam("color_g", "slider", "Color G", 255, 0, 255, 1),
                    EffectParam("color_b", "slider", "Color B", 0, 0, 255, 1)
                ]
            ),
            "ORB Features": EffectMethod(
                name="ORB Features",
                description="Detect ORB (Oriented FAST and Rotated BRIEF) keypoints - faster than SIFT",
                function="cv2.ORB_create",
                params=[
                    EffectParam("nfeatures", "slider", "Max Features", 500, 10, 2000, 10),
                    EffectParam("scale_factor", "slider", "Scale Factor", 1.2, 1.1, 2.0, 0.1),
                    EffectParam("draw_rich", "dropdown", "Draw Style", "Rich", options=["Simple", "Rich"]),
                    EffectParam("color_r", "slider", "Color R", 255, 0, 255, 1),
                    EffectParam("color_g", "slider", "Color G", 0, 0, 255, 1),
                    EffectParam("color_b", "slider", "Color B", 255, 0, 255, 1)
                ]
            ),
            "Harris Corners": EffectMethod(
                name="Harris Corners",
                description="Detect corners using Harris corner detection",
                function="cv2.cornerHarris",
                params=[
                    EffectParam("block_size", "slider", "Block Size", 2, 2, 10, 1),
                    EffectParam("ksize", "slider", "Sobel Kernel Size", 3, 3, 7, 2),
                    EffectParam("k", "slider", "Harris Parameter", 0.04, 0.01, 0.1, 0.01),
                    EffectParam("threshold", "slider", "Threshold (%)", 1, 0, 10, 1),
                    EffectParam("color_r", "slider", "Color R", 255, 0, 255, 1),
                    EffectParam("color_g", "slider", "Color G", 0, 0, 255, 1),
                    EffectParam("color_b", "slider", "Color B", 0, 0, 255, 1)
                ]
            ),
            "Shi-Tomasi Corners": EffectMethod(
                name="Shi-Tomasi Corners",
                description="Detect corners using Shi-Tomasi method (Good Features to Track)",
                function="cv2.goodFeaturesToTrack",
                params=[
                    EffectParam("max_corners", "slider", "Max Corners", 100, 10, 500, 10),
                    EffectParam("quality", "slider", "Quality Level", 0.01, 0.001, 0.1, 0.001),
                    EffectParam("min_distance", "slider", "Min Distance", 10, 1, 50, 1),
                    EffectParam("radius", "slider", "Draw Radius", 5, 1, 20, 1),
                    EffectParam("color_r", "slider", "Color R", 0, 0, 255, 1),
                    EffectParam("color_g", "slider", "Color G", 255, 0, 255, 1),
                    EffectParam("color_b", "slider", "Color B", 255, 0, 255, 1)
                ]
            ),
            "FAST Features": EffectMethod(
                name="FAST Features",
                description="Detect FAST (Features from Accelerated Segment Test) keypoints",
                function="cv2.FastFeatureDetector_create",
                params=[
                    EffectParam("threshold", "slider", "Threshold", 25, 1, 100, 1),
                    EffectParam("nonmax", "dropdown", "Non-max Suppression", "Yes", options=["Yes", "No"]),
                    EffectParam("draw_rich", "dropdown", "Draw Style", "Simple", options=["Simple", "Rich"]),
                    EffectParam("color_r", "slider", "Color R", 0, 0, 255, 1),
                    EffectParam("color_g", "slider", "Color G", 255, 0, 255, 1),
                    EffectParam("color_b", "slider", "Color B", 0, 0, 255, 1)
                ]
            ),
            "AKAZE Features": EffectMethod(
                name="AKAZE Features",
                description="Detect AKAZE keypoints - accelerated KAZE",
                function="cv2.AKAZE_create",
                params=[
                    EffectParam("threshold", "slider", "Threshold", 0.001, 0.0001, 0.01, 0.0001),
                    EffectParam("draw_rich", "dropdown", "Draw Style", "Rich", options=["Simple", "Rich"]),
                    EffectParam("color_r", "slider", "Color R", 255, 0, 255, 1),
                    EffectParam("color_g", "slider", "Color G", 255, 0, 255, 1),
                    EffectParam("color_b", "slider", "Color B", 0, 0, 255, 1)
                ]
            ),
            "Blob Detection": EffectMethod(
                name="Blob Detection",
                description="Detect blobs using SimpleBlobDetector",
                function="cv2.SimpleBlobDetector_create",
                params=[
                    EffectParam("min_area", "slider", "Min Area", 100, 10, 1000, 10),
                    EffectParam("max_area", "slider", "Max Area", 5000, 500, 50000, 100),
                    EffectParam("min_circularity", "slider", "Min Circularity", 0.1, 0.0, 1.0, 0.1),
                    EffectParam("color_r", "slider", "Color R", 255, 0, 255, 1),
                    EffectParam("color_g", "slider", "Color G", 0, 0, 255, 1),
                    EffectParam("color_b", "slider", "Color B", 255, 0, 255, 1)
                ]
            )
        }
    
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get color from params
        color = (
            int(params.get("color_b", 0)),
            int(params.get("color_g", 255)),
            int(params.get("color_r", 0))
        )
        
        if method_name == "SIFT Features":
            nfeatures = int(params.get("nfeatures", 500))
            draw_style = params.get("draw_rich", "Rich")
            
            sift = cv2.SIFT_create(nFeatures=nfeatures)
            keypoints = sift.detect(gray, None)
            
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if draw_style == "Rich" else cv2.DRAW_MATCHES_FLAGS_DEFAULT
            result = cv2.drawKeypoints(img, keypoints, None, color=color, flags=flags)
            
        elif method_name == "ORB Features":
            nfeatures = int(params.get("nfeatures", 500))
            scale_factor = float(params.get("scale_factor", 1.2))
            draw_style = params.get("draw_rich", "Rich")
            
            orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=scale_factor)
            keypoints = orb.detect(gray, None)
            
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if draw_style == "Rich" else cv2.DRAW_MATCHES_FLAGS_DEFAULT
            result = cv2.drawKeypoints(img, keypoints, None, color=color, flags=flags)
            
        elif method_name == "Harris Corners":
            block_size = int(params.get("block_size", 2))
            ksize = int(params.get("ksize", 3))
            k = float(params.get("k", 0.04))
            threshold_pct = float(params.get("threshold", 1)) / 100
            
            gray_float = np.float32(gray)
            harris = cv2.cornerHarris(gray_float, block_size, ksize, k)
            harris = cv2.dilate(harris, None)
            
            result = img.copy()
            result[harris > threshold_pct * harris.max()] = [color[2], color[1], color[0]]
            
        elif method_name == "Shi-Tomasi Corners":
            max_corners = int(params.get("max_corners", 100))
            quality = float(params.get("quality", 0.01))
            min_distance = int(params.get("min_distance", 10))
            radius = int(params.get("radius", 5))
            
            corners = cv2.goodFeaturesToTrack(gray, max_corners, quality, min_distance)
            
            result = img.copy()
            if corners is not None:
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(result, (int(x), int(y)), radius, color, -1)
                    
        elif method_name == "FAST Features":
            threshold = int(params.get("threshold", 25))
            nonmax = params.get("nonmax", "Yes") == "Yes"
            draw_style = params.get("draw_rich", "Simple")
            
            fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax)
            keypoints = fast.detect(gray, None)
            
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if draw_style == "Rich" else cv2.DRAW_MATCHES_FLAGS_DEFAULT
            result = cv2.drawKeypoints(img, keypoints, None, color=color, flags=flags)
            
        elif method_name == "AKAZE Features":
            threshold = float(params.get("threshold", 0.001))
            draw_style = params.get("draw_rich", "Rich")
            
            akaze = cv2.AKAZE_create(threshold=threshold)
            keypoints = akaze.detect(gray, None)
            
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if draw_style == "Rich" else cv2.DRAW_MATCHES_FLAGS_DEFAULT
            result = cv2.drawKeypoints(img, keypoints, None, color=color, flags=flags)
            
        elif method_name == "Blob Detection":
            min_area = int(params.get("min_area", 100))
            max_area = int(params.get("max_area", 5000))
            min_circularity = float(params.get("min_circularity", 0.1))
            
            # Setup blob detector parameters
            blob_params = cv2.SimpleBlobDetector_Params()
            blob_params.filterByArea = True
            blob_params.minArea = min_area
            blob_params.maxArea = max_area
            blob_params.filterByCircularity = True
            blob_params.minCircularity = min_circularity
            blob_params.filterByConvexity = False
            blob_params.filterByInertia = False
            
            detector = cv2.SimpleBlobDetector_create(blob_params)
            keypoints = detector.detect(gray)
            
            result = cv2.drawKeypoints(img, keypoints, None, color=color, 
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            result = img
        
        return result
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        color = (int(params.get("color_b", 0)), int(params.get("color_g", 255)), int(params.get("color_r", 0)))
        
        if method_name == "SIFT Features":
            nfeatures = int(params.get("nfeatures", 500))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"sift = cv2.SIFT_create(nFeatures={nfeatures})")
            code_lines.append("keypoints = sift.detect(gray, None)")
            code_lines.append(f"result = cv2.drawKeypoints(img, keypoints, None, color={color}, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)")
            param_info.append({"function": "cv2.SIFT_create", "params": {"nFeatures": nfeatures}})
            
        elif method_name == "ORB Features":
            nfeatures = int(params.get("nfeatures", 500))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"orb = cv2.ORB_create(nfeatures={nfeatures})")
            code_lines.append("keypoints = orb.detect(gray, None)")
            code_lines.append(f"result = cv2.drawKeypoints(img, keypoints, None, color={color})")
            param_info.append({"function": "cv2.ORB_create", "params": {"nfeatures": nfeatures}})
            
        elif method_name == "Harris Corners":
            block_size = int(params.get("block_size", 2))
            ksize = int(params.get("ksize", 3))
            k = float(params.get("k", 0.04))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append("gray_float = np.float32(gray)")
            code_lines.append(f"harris = cv2.cornerHarris(gray_float, {block_size}, {ksize}, {k})")
            code_lines.append("result = img.copy()")
            code_lines.append(f"result[harris > 0.01 * harris.max()] = {list(color)}")
            param_info.append({"function": "cv2.cornerHarris", "params": {"blockSize": block_size, "ksize": ksize, "k": k}})
            
        elif method_name == "Shi-Tomasi Corners":
            max_corners = int(params.get("max_corners", 100))
            quality = float(params.get("quality", 0.01))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"corners = cv2.goodFeaturesToTrack(gray, {max_corners}, {quality}, 10)")
            code_lines.append("result = img.copy()")
            code_lines.append("for corner in corners:")
            code_lines.append("    x, y = corner.ravel()")
            code_lines.append(f"    cv2.circle(result, (int(x), int(y)), 5, {color}, -1)")
            param_info.append({"function": "cv2.goodFeaturesToTrack", "params": {"maxCorners": max_corners, "qualityLevel": quality}})
            
        elif method_name == "FAST Features":
            threshold = int(params.get("threshold", 25))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"fast = cv2.FastFeatureDetector_create(threshold={threshold})")
            code_lines.append("keypoints = fast.detect(gray, None)")
            code_lines.append(f"result = cv2.drawKeypoints(img, keypoints, None, color={color})")
            param_info.append({"function": "cv2.FastFeatureDetector_create", "params": {"threshold": threshold}})
            
        elif method_name == "AKAZE Features":
            threshold = float(params.get("threshold", 0.001))
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            code_lines.append(f"akaze = cv2.AKAZE_create(threshold={threshold})")
            code_lines.append("keypoints = akaze.detect(gray, None)")
            code_lines.append(f"result = cv2.drawKeypoints(img, keypoints, None, color={color})")
            param_info.append({"function": "cv2.AKAZE_create", "params": {"threshold": threshold}})
            
        elif method_name == "Blob Detection":
            min_area = int(params.get("min_area", 100))
            max_area = int(params.get("max_area", 5000))
            code_lines.append("params = cv2.SimpleBlobDetector_Params()")
            code_lines.append(f"params.minArea = {min_area}")
            code_lines.append(f"params.maxArea = {max_area}")
            code_lines.append("detector = cv2.SimpleBlobDetector_create(params)")
            code_lines.append("keypoints = detector.detect(gray)")
            code_lines.append(f"result = cv2.drawKeypoints(img, keypoints, None, color={color})")
            param_info.append({"function": "cv2.SimpleBlobDetector", "params": {"minArea": min_area, "maxArea": max_area}})
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
