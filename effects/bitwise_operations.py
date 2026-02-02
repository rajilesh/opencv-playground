"""
Bitwise Operations

This module provides bitwise operations including:
- Bitwise AND
- Bitwise OR
- Bitwise XOR
- Bitwise NOT
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect, EffectMethod, EffectParam


class BitwiseOperationsEffect(BaseEffect):
    """Bitwise operations for images"""
    
    @property
    def category_name(self) -> str:
        return "Bitwise Operations"
    
    @property
    def category_icon(self) -> str:
        return "🔢"
    
    def get_methods(self):
        return {
            "Bitwise NOT": EffectMethod(
                name="Bitwise NOT",
                description="Invert all bits in the image (creates negative)",
                function="cv2.bitwise_not",
                params=[
                    EffectParam("use_roi", "checkbox", "Use ROI Region", False),
                    EffectParam("roi_x", "slider", "ROI X (%)", 0, 0, 100, 1),
                    EffectParam("roi_y", "slider", "ROI Y (%)", 0, 0, 100, 1),
                    EffectParam("roi_width", "slider", "ROI Width (%)", 100, 1, 100, 1),
                    EffectParam("roi_height", "slider", "ROI Height (%)", 100, 1, 100, 1)
                ]
            ),
            "Bitwise AND (with mask)": EffectMethod(
                name="Bitwise AND (with mask)",
                description="Apply bitwise AND using a pipeline step as mask",
                function="cv2.bitwise_and",
                params=[
                    EffectParam("mask_step", "slider", "Mask from Step #", 1, 1, 20, 1),
                    EffectParam("image_step", "slider", "Image from Step # (0=current)", 0, 0, 20, 1),
                    EffectParam("invert_mask", "checkbox", "Invert Mask", False),
                    EffectParam("use_roi", "checkbox", "Use ROI Region", False),
                    EffectParam("roi_x", "slider", "ROI X (%)", 0, 0, 100, 1),
                    EffectParam("roi_y", "slider", "ROI Y (%)", 0, 0, 100, 1),
                    EffectParam("roi_width", "slider", "ROI Width (%)", 100, 1, 100, 1),
                    EffectParam("roi_height", "slider", "ROI Height (%)", 100, 1, 100, 1)
                ]
            ),
            "Bitwise OR (with mask)": EffectMethod(
                name="Bitwise OR (with mask)",
                description="Apply bitwise OR using a pipeline step as mask",
                function="cv2.bitwise_or",
                params=[
                    EffectParam("mask_step", "slider", "Mask from Step #", 1, 1, 20, 1),
                    EffectParam("image_step", "slider", "Image from Step # (0=current)", 0, 0, 20, 1),
                    EffectParam("invert_mask", "checkbox", "Invert Mask", False),
                    EffectParam("use_roi", "checkbox", "Use ROI Region", False),
                    EffectParam("roi_x", "slider", "ROI X (%)", 0, 0, 100, 1),
                    EffectParam("roi_y", "slider", "ROI Y (%)", 0, 0, 100, 1),
                    EffectParam("roi_width", "slider", "ROI Width (%)", 100, 1, 100, 1),
                    EffectParam("roi_height", "slider", "ROI Height (%)", 100, 1, 100, 1)
                ]
            ),
            "Bitwise XOR (with mask)": EffectMethod(
                name="Bitwise XOR (with mask)",
                description="Apply bitwise XOR using a pipeline step as mask",
                function="cv2.bitwise_xor",
                params=[
                    EffectParam("mask_step", "slider", "Mask from Step #", 1, 1, 20, 1),
                    EffectParam("image_step", "slider", "Image from Step # (0=current)", 0, 0, 20, 1),
                    EffectParam("invert_mask", "checkbox", "Invert Mask", False),
                    EffectParam("use_roi", "checkbox", "Use ROI Region", False),
                    EffectParam("roi_x", "slider", "ROI X (%)", 0, 0, 100, 1),
                    EffectParam("roi_y", "slider", "ROI Y (%)", 0, 0, 100, 1),
                    EffectParam("roi_width", "slider", "ROI Width (%)", 100, 1, 100, 1),
                    EffectParam("roi_height", "slider", "ROI Height (%)", 100, 1, 100, 1)
                ]
            ),
            "Bitwise AND (shape mask)": EffectMethod(
                name="Bitwise AND (shape mask)",
                description="Apply bitwise AND with a generated shape mask",
                function="cv2.bitwise_and",
                params=[
                    EffectParam("mask_type", "selectbox", "Mask Type", "circle", None, None, None, 
                               ["circle", "rectangle", "horizontal_gradient", "vertical_gradient"]),
                    EffectParam("mask_size", "slider", "Mask Size (%)", 50, 10, 100, 5),
                    EffectParam("invert_mask", "checkbox", "Invert Mask", False),
                    EffectParam("use_roi", "checkbox", "Use ROI Region", False),
                    EffectParam("roi_x", "slider", "ROI X (%)", 0, 0, 100, 1),
                    EffectParam("roi_y", "slider", "ROI Y (%)", 0, 0, 100, 1),
                    EffectParam("roi_width", "slider", "ROI Width (%)", 100, 1, 100, 1),
                    EffectParam("roi_height", "slider", "ROI Height (%)", 100, 1, 100, 1)
                ]
            ),
            "Bitwise OR (shape mask)": EffectMethod(
                name="Bitwise OR (shape mask)",
                description="Apply bitwise OR with a generated shape mask",
                function="cv2.bitwise_or",
                params=[
                    EffectParam("mask_type", "selectbox", "Mask Type", "circle", None, None, None,
                               ["circle", "rectangle", "horizontal_gradient", "vertical_gradient"]),
                    EffectParam("mask_size", "slider", "Mask Size (%)", 50, 10, 100, 5),
                    EffectParam("invert_mask", "checkbox", "Invert Mask", False),
                    EffectParam("use_roi", "checkbox", "Use ROI Region", False),
                    EffectParam("roi_x", "slider", "ROI X (%)", 0, 0, 100, 1),
                    EffectParam("roi_y", "slider", "ROI Y (%)", 0, 0, 100, 1),
                    EffectParam("roi_width", "slider", "ROI Width (%)", 100, 1, 100, 1),
                    EffectParam("roi_height", "slider", "ROI Height (%)", 100, 1, 100, 1)
                ]
            ),
            "Bitwise XOR (shape mask)": EffectMethod(
                name="Bitwise XOR (shape mask)",
                description="Apply bitwise XOR with a generated shape mask",
                function="cv2.bitwise_xor",
                params=[
                    EffectParam("mask_type", "selectbox", "Mask Type", "circle", None, None, None,
                               ["circle", "rectangle", "horizontal_gradient", "vertical_gradient"]),
                    EffectParam("mask_size", "slider", "Mask Size (%)", 50, 10, 100, 5),
                    EffectParam("invert_mask", "checkbox", "Invert Mask", False),
                    EffectParam("use_roi", "checkbox", "Use ROI Region", False),
                    EffectParam("roi_x", "slider", "ROI X (%)", 0, 0, 100, 1),
                    EffectParam("roi_y", "slider", "ROI Y (%)", 0, 0, 100, 1),
                    EffectParam("roi_width", "slider", "ROI Width (%)", 100, 1, 100, 1),
                    EffectParam("roi_height", "slider", "ROI Height (%)", 100, 1, 100, 1)
                ]
            ),
            "Bitwise AND (threshold mask)": EffectMethod(
                name="Bitwise AND (threshold mask)",
                description="Apply bitwise AND using a threshold-based mask",
                function="cv2.bitwise_and",
                params=[
                    EffectParam("threshold", "slider", "Threshold", 127, 0, 255, 1),
                    EffectParam("invert_mask", "checkbox", "Invert Mask", False),
                    EffectParam("use_roi", "checkbox", "Use ROI Region", False),
                    EffectParam("roi_x", "slider", "ROI X (%)", 0, 0, 100, 1),
                    EffectParam("roi_y", "slider", "ROI Y (%)", 0, 0, 100, 1),
                    EffectParam("roi_width", "slider", "ROI Width (%)", 100, 1, 100, 1),
                    EffectParam("roi_height", "slider", "ROI Height (%)", 100, 1, 100, 1)
                ]
            ),
            "Bitwise OR (threshold mask)": EffectMethod(
                name="Bitwise OR (threshold mask)",
                description="Apply bitwise OR using a threshold-based mask",
                function="cv2.bitwise_or",
                params=[
                    EffectParam("threshold", "slider", "Threshold", 127, 0, 255, 1),
                    EffectParam("invert_mask", "checkbox", "Invert Mask", False),
                    EffectParam("use_roi", "checkbox", "Use ROI Region", False),
                    EffectParam("roi_x", "slider", "ROI X (%)", 0, 0, 100, 1),
                    EffectParam("roi_y", "slider", "ROI Y (%)", 0, 0, 100, 1),
                    EffectParam("roi_width", "slider", "ROI Width (%)", 100, 1, 100, 1),
                    EffectParam("roi_height", "slider", "ROI Height (%)", 100, 1, 100, 1)
                ]
            ),
            "Bitwise XOR (threshold mask)": EffectMethod(
                name="Bitwise XOR (threshold mask)",
                description="Apply bitwise XOR using a threshold-based mask",
                function="cv2.bitwise_xor",
                params=[
                    EffectParam("threshold", "slider", "Threshold", 127, 0, 255, 1),
                    EffectParam("invert_mask", "checkbox", "Invert Mask", False),
                    EffectParam("use_roi", "checkbox", "Use ROI Region", False),
                    EffectParam("roi_x", "slider", "ROI X (%)", 0, 0, 100, 1),
                    EffectParam("roi_y", "slider", "ROI Y (%)", 0, 0, 100, 1),
                    EffectParam("roi_width", "slider", "ROI Width (%)", 100, 1, 100, 1),
                    EffectParam("roi_height", "slider", "ROI Height (%)", 100, 1, 100, 1)
                ]
            ),
            "Combine with Mask": EffectMethod(
                name="Combine with Mask",
                description="Combine color image with edge mask from pipeline steps",
                function="combine_pipeline",
                params=[
                    EffectParam("color_step", "slider", "Color Image from Step #", 2, 0, 20, 1),
                    EffectParam("mask_step", "slider", "Mask from Step #", 1, 1, 20, 1),
                    EffectParam("invert_mask", "checkbox", "Invert Mask", False),
                    EffectParam("use_roi", "checkbox", "Use ROI Region", False),
                    EffectParam("roi_x", "slider", "ROI X (%)", 0, 0, 100, 1),
                    EffectParam("roi_y", "slider", "ROI Y (%)", 0, 0, 100, 1),
                    EffectParam("roi_width", "slider", "ROI Width (%)", 100, 1, 100, 1),
                    EffectParam("roi_height", "slider", "ROI Height (%)", 100, 1, 100, 1)
                ]
            )
        }
    
    def _get_roi_bounds(self, shape, params: dict) -> tuple:
        """Get ROI bounds from parameters. Returns (x1, y1, x2, y2) or None if ROI not used"""
        use_roi = params.get("use_roi", False)
        if not use_roi:
            return None
        
        h, w = shape[:2]
        roi_x = int(params.get("roi_x", 0))
        roi_y = int(params.get("roi_y", 0))
        roi_width = int(params.get("roi_width", 100))
        roi_height = int(params.get("roi_height", 100))
        
        x1 = int(w * roi_x / 100)
        y1 = int(h * roi_y / 100)
        x2 = min(int(w * (roi_x + roi_width) / 100), w)
        y2 = min(int(h * (roi_y + roi_height) / 100), h)
        
        return (x1, y1, x2, y2)
    
    def _apply_with_roi(self, original: np.ndarray, processed: np.ndarray, roi_bounds: tuple) -> np.ndarray:
        """Apply processed result only within ROI, keeping original outside"""
        if roi_bounds is None:
            return processed
        
        x1, y1, x2, y2 = roi_bounds
        result = original.copy()
        
        # Handle dimension mismatch
        if len(processed.shape) == 2 and len(result.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        elif len(processed.shape) == 3 and len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
        result[y1:y2, x1:x2] = processed[y1:y2, x1:x2]
        return result
    
    def _create_shape_mask(self, shape, mask_type: str, size_percent: int) -> np.ndarray:
        """Create a mask based on the specified type and size"""
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if mask_type == "circle":
            center = (w // 2, h // 2)
            radius = int(min(w, h) * size_percent / 200)
            cv2.circle(mask, center, radius, 255, -1)
        elif mask_type == "rectangle":
            margin_x = int(w * (100 - size_percent) / 200)
            margin_y = int(h * (100 - size_percent) / 200)
            cv2.rectangle(mask, (margin_x, margin_y), (w - margin_x, h - margin_y), 255, -1)
        elif mask_type == "horizontal_gradient":
            for i in range(w):
                val = int(255 * i / w * size_percent / 100)
                mask[:, i] = val
        elif mask_type == "vertical_gradient":
            for i in range(h):
                val = int(255 * i / h * size_percent / 100)
                mask[i, :] = val
        
        return mask
    
    def _create_threshold_mask(self, image: np.ndarray, threshold: int, invert: bool) -> np.ndarray:
        """Create a mask based on threshold"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if invert:
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return mask
    
    def _get_pipeline_image(self, pipeline_results: dict, step: int, fallback: np.ndarray) -> np.ndarray:
        """Get image from pipeline step"""
        if step in pipeline_results:
            return pipeline_results[step].copy()
        return fallback.copy()
    
    def _get_pipeline_mask(self, pipeline_results: dict, mask_step: int, fallback: np.ndarray, invert: bool = False) -> np.ndarray:
        """Get mask from a pipeline step and ensure it's grayscale"""
        mask_img = self._get_pipeline_image(pipeline_results, mask_step, fallback)
        
        # Convert to grayscale if needed
        if len(mask_img.shape) == 3:
            mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        else:
            mask = mask_img.copy()
        
        if invert:
            mask = cv2.bitwise_not(mask)
        
        return mask

    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        img = image.copy()
        pipeline_results = params.get("_pipeline_results", {})
        roi_bounds = self._get_roi_bounds(img.shape, params)
        invert_mask = params.get("invert_mask", False)
        
        if method_name == "Bitwise NOT":
            processed = cv2.bitwise_not(img)
            result = self._apply_with_roi(img, processed, roi_bounds)
        
        elif method_name == "Bitwise AND (with mask)":
            mask_step = int(params.get("mask_step", 1))
            image_step = int(params.get("image_step", 0))
            
            # Get source image from specified step
            if image_step > 0:
                source_img = self._get_pipeline_image(pipeline_results, image_step, img)
            else:
                source_img = img.copy()
            
            # Get mask from specified step
            mask = self._get_pipeline_mask(pipeline_results, mask_step, img, invert_mask)
            
            # Apply bitwise AND
            processed = cv2.bitwise_and(source_img, source_img, mask=mask)
            result = self._apply_with_roi(source_img, processed, roi_bounds)
        
        elif method_name == "Bitwise OR (with mask)":
            mask_step = int(params.get("mask_step", 1))
            image_step = int(params.get("image_step", 0))
            
            if image_step > 0:
                source_img = self._get_pipeline_image(pipeline_results, image_step, img)
            else:
                source_img = img.copy()
            
            mask = self._get_pipeline_mask(pipeline_results, mask_step, img, invert_mask)
            
            white = np.ones_like(source_img) * 255
            masked_white = cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask))
            processed = cv2.bitwise_or(source_img, masked_white)
            result = self._apply_with_roi(source_img, processed, roi_bounds)
        
        elif method_name == "Bitwise XOR (with mask)":
            mask_step = int(params.get("mask_step", 1))
            image_step = int(params.get("image_step", 0))
            
            if image_step > 0:
                source_img = self._get_pipeline_image(pipeline_results, image_step, img)
            else:
                source_img = img.copy()
            
            mask = self._get_pipeline_mask(pipeline_results, mask_step, img, invert_mask)
            
            # Convert mask to match source channels
            if len(source_img.shape) == 3:
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_3ch = mask
            
            processed = cv2.bitwise_xor(source_img, mask_3ch)
            result = self._apply_with_roi(source_img, processed, roi_bounds)
        
        elif method_name == "Bitwise AND (shape mask)":
            mask_type = params.get("mask_type", "circle")
            mask_size = int(params.get("mask_size", 50))
            
            mask = self._create_shape_mask(img.shape, mask_type, mask_size)
            if invert_mask:
                mask = cv2.bitwise_not(mask)
            
            processed = cv2.bitwise_and(img, img, mask=mask)
            result = self._apply_with_roi(img, processed, roi_bounds)
        
        elif method_name == "Bitwise OR (shape mask)":
            mask_type = params.get("mask_type", "circle")
            mask_size = int(params.get("mask_size", 50))
            
            mask = self._create_shape_mask(img.shape, mask_type, mask_size)
            if invert_mask:
                mask = cv2.bitwise_not(mask)
            
            white = np.ones_like(img) * 255
            masked_white = cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask))
            processed = cv2.bitwise_or(img, masked_white)
            result = self._apply_with_roi(img, processed, roi_bounds)
        
        elif method_name == "Bitwise XOR (shape mask)":
            mask_type = params.get("mask_type", "circle")
            mask_size = int(params.get("mask_size", 50))
            
            mask = self._create_shape_mask(img.shape, mask_type, mask_size)
            if invert_mask:
                mask = cv2.bitwise_not(mask)
            
            if len(img.shape) == 3:
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_3ch = mask
            
            processed = cv2.bitwise_xor(img, mask_3ch)
            result = self._apply_with_roi(img, processed, roi_bounds)
        
        elif method_name == "Bitwise AND (threshold mask)":
            threshold = int(params.get("threshold", 127))
            
            mask = self._create_threshold_mask(img, threshold, invert_mask)
            processed = cv2.bitwise_and(img, img, mask=mask)
            result = self._apply_with_roi(img, processed, roi_bounds)
        
        elif method_name == "Bitwise OR (threshold mask)":
            threshold = int(params.get("threshold", 127))
            
            mask = self._create_threshold_mask(img, threshold, invert_mask)
            white = np.ones_like(img) * 255
            masked_white = cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask))
            processed = cv2.bitwise_or(img, masked_white)
            result = self._apply_with_roi(img, processed, roi_bounds)
        
        elif method_name == "Bitwise XOR (threshold mask)":
            threshold = int(params.get("threshold", 127))
            
            mask = self._create_threshold_mask(img, threshold, invert_mask)
            
            if len(img.shape) == 3:
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_3ch = mask
            
            processed = cv2.bitwise_xor(img, mask_3ch)
            result = self._apply_with_roi(img, processed, roi_bounds)
        
        elif method_name == "Combine with Mask":
            color_step = int(params.get("color_step", 2))
            mask_step = int(params.get("mask_step", 1))
            
            # Get color image from specified step
            if color_step > 0:
                color_img = self._get_pipeline_image(pipeline_results, color_step, img)
            else:
                color_img = img.copy()
            
            # Get mask from specified step
            mask = self._get_pipeline_mask(pipeline_results, mask_step, img, invert_mask)
            
            processed = cv2.bitwise_and(color_img, color_img, mask=mask)
            result = self._apply_with_roi(color_img, processed, roi_bounds)
        
        else:
            result = img
        
        return result
    
    def _generate_roi_code(self, params: dict) -> list:
        """Generate ROI code lines if ROI is enabled"""
        use_roi = params.get("use_roi", False)
        if not use_roi:
            return []
        
        roi_x = int(params.get("roi_x", 0))
        roi_y = int(params.get("roi_y", 0))
        roi_width = int(params.get("roi_width", 100))
        roi_height = int(params.get("roi_height", 100))
        
        lines = [
            "# Apply ROI region",
            "h, w = img.shape[:2]",
            f"x1 = int(w * {roi_x} / 100)",
            f"y1 = int(h * {roi_y} / 100)",
            f"x2 = min(int(w * ({roi_x} + {roi_width}) / 100), w)",
            f"y2 = min(int(h * ({roi_y} + {roi_height}) / 100), h)",
            "original = img.copy()",
        ]
        return lines
    
    def _generate_roi_apply_code(self, params: dict) -> list:
        """Generate code to apply ROI at the end"""
        use_roi = params.get("use_roi", False)
        if not use_roi:
            return []
        
        return [
            "# Apply result only within ROI",
            "final_result = original.copy()",
            "final_result[y1:y2, x1:x2] = result[y1:y2, x1:x2]",
            "result = final_result"
        ]
    
    def generate_code(self, method_name: str, params: dict) -> dict:
        code_lines = []
        param_info = []
        
        # Add ROI setup code if enabled
        roi_setup = self._generate_roi_code(params)
        code_lines.extend(roi_setup)
        
        if method_name == "Bitwise NOT":
            code_lines.append("result = cv2.bitwise_not(img)")
            param_info.append({"function": "cv2.bitwise_not", "params": {}})
        
        elif method_name == "Bitwise AND (with mask)":
            mask_step = int(params.get("mask_step", 1))
            image_step = int(params.get("image_step", 0))
            invert = params.get("invert_mask", False)
            code_lines.append(f"# Get mask from step {mask_step}")
            code_lines.append(f"mask_img = step_{mask_step}_result")
            code_lines.append("if len(mask_img.shape) == 3:")
            code_lines.append("    mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)")
            code_lines.append("else:")
            code_lines.append("    mask = mask_img.copy()")
            if invert:
                code_lines.append("mask = cv2.bitwise_not(mask)")
            source = f"step_{image_step}_result"  # Use step result for consistent pipeline access
            code_lines.append(f"result = cv2.bitwise_and({source}, {source}, mask=mask)")
            param_info.append({"function": "cv2.bitwise_and", "params": {"mask_step": mask_step, "image_step": image_step}})
        
        elif method_name == "Bitwise OR (with mask)":
            mask_step = int(params.get("mask_step", 1))
            image_step = int(params.get("image_step", 0))
            invert = params.get("invert_mask", False)
            code_lines.append(f"# Get mask from step {mask_step}")
            code_lines.append(f"mask_img = step_{mask_step}_result")
            code_lines.append("if len(mask_img.shape) == 3:")
            code_lines.append("    mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)")
            code_lines.append("else:")
            code_lines.append("    mask = mask_img.copy()")
            if invert:
                code_lines.append("mask = cv2.bitwise_not(mask)")
            source = f"step_{image_step}_result"  # Use step result for consistent pipeline access
            code_lines.append(f"white = np.ones_like({source}) * 255")
            code_lines.append("masked_white = cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask))")
            code_lines.append(f"result = cv2.bitwise_or({source}, masked_white)")
            param_info.append({"function": "cv2.bitwise_or", "params": {"mask_step": mask_step, "image_step": image_step}})
        
        elif method_name == "Bitwise XOR (with mask)":
            mask_step = int(params.get("mask_step", 1))
            image_step = int(params.get("image_step", 0))
            invert = params.get("invert_mask", False)
            code_lines.append(f"# Get mask from step {mask_step}")
            code_lines.append(f"mask_img = step_{mask_step}_result")
            code_lines.append("if len(mask_img.shape) == 3:")
            code_lines.append("    mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)")
            code_lines.append("else:")
            code_lines.append("    mask = mask_img.copy()")
            if invert:
                code_lines.append("mask = cv2.bitwise_not(mask)")
            source = f"step_{image_step}_result"  # Use step result for consistent pipeline access
            code_lines.append("mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)")
            code_lines.append(f"result = cv2.bitwise_xor({source}, mask_3ch)")
            param_info.append({"function": "cv2.bitwise_xor", "params": {"mask_step": mask_step, "image_step": image_step}})
        
        elif method_name == "Bitwise AND (shape mask)":
            mask_type = params.get("mask_type", "circle")
            mask_size = int(params.get("mask_size", 50))
            invert = params.get("invert_mask", False)
            code_lines.append("# Create shape mask")
            code_lines.append(f"mask = create_shape_mask(img.shape, '{mask_type}', {mask_size})")
            if invert:
                code_lines.append("mask = cv2.bitwise_not(mask)")
            code_lines.append("result = cv2.bitwise_and(img, img, mask=mask)")
            param_info.append({"function": "cv2.bitwise_and", "params": {"mask_type": mask_type, "mask_size": mask_size}})
        
        elif method_name == "Bitwise OR (shape mask)":
            mask_type = params.get("mask_type", "circle")
            mask_size = int(params.get("mask_size", 50))
            invert = params.get("invert_mask", False)
            code_lines.append("# Create shape mask and apply OR")
            code_lines.append(f"mask = create_shape_mask(img.shape, '{mask_type}', {mask_size})")
            if invert:
                code_lines.append("mask = cv2.bitwise_not(mask)")
            code_lines.append("white = np.ones_like(img) * 255")
            code_lines.append("masked_white = cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask))")
            code_lines.append("result = cv2.bitwise_or(img, masked_white)")
            param_info.append({"function": "cv2.bitwise_or", "params": {"mask_type": mask_type, "mask_size": mask_size}})
        
        elif method_name == "Bitwise XOR (shape mask)":
            mask_type = params.get("mask_type", "circle")
            mask_size = int(params.get("mask_size", 50))
            invert = params.get("invert_mask", False)
            code_lines.append("# Create shape mask and apply XOR")
            code_lines.append(f"mask = create_shape_mask(img.shape, '{mask_type}', {mask_size})")
            if invert:
                code_lines.append("mask = cv2.bitwise_not(mask)")
            code_lines.append("mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)")
            code_lines.append("result = cv2.bitwise_xor(img, mask_3ch)")
            param_info.append({"function": "cv2.bitwise_xor", "params": {"mask_type": mask_type, "mask_size": mask_size}})
        
        elif method_name == "Bitwise AND (threshold mask)":
            threshold = int(params.get("threshold", 127))
            invert = params.get("invert_mask", False)
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            thresh_type = "cv2.THRESH_BINARY_INV" if invert else "cv2.THRESH_BINARY"
            code_lines.append(f"_, mask = cv2.threshold(gray, {threshold}, 255, {thresh_type})")
            code_lines.append("result = cv2.bitwise_and(img, img, mask=mask)")
            param_info.append({"function": "cv2.bitwise_and", "params": {"threshold": threshold, "invert": invert}})
        
        elif method_name == "Bitwise OR (threshold mask)":
            threshold = int(params.get("threshold", 127))
            invert = params.get("invert_mask", False)
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            thresh_type = "cv2.THRESH_BINARY_INV" if invert else "cv2.THRESH_BINARY"
            code_lines.append(f"_, mask = cv2.threshold(gray, {threshold}, 255, {thresh_type})")
            code_lines.append("white = np.ones_like(img) * 255")
            code_lines.append("masked_white = cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask))")
            code_lines.append("result = cv2.bitwise_or(img, masked_white)")
            param_info.append({"function": "cv2.bitwise_or", "params": {"threshold": threshold, "invert": invert}})
        
        elif method_name == "Bitwise XOR (threshold mask)":
            threshold = int(params.get("threshold", 127))
            invert = params.get("invert_mask", False)
            code_lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
            thresh_type = "cv2.THRESH_BINARY_INV" if invert else "cv2.THRESH_BINARY"
            code_lines.append(f"_, mask = cv2.threshold(gray, {threshold}, 255, {thresh_type})")
            code_lines.append("mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)")
            code_lines.append("result = cv2.bitwise_xor(img, mask_3ch)")
            param_info.append({"function": "cv2.bitwise_xor", "params": {"threshold": threshold, "invert": invert}})
        
        elif method_name == "Combine with Mask":
            color_step = int(params.get("color_step", 2))
            mask_step = int(params.get("mask_step", 1))
            invert = params.get("invert_mask", False)
            code_lines.append(f"# Combine color from step {color_step} with mask from step {mask_step}")
            color_source = f"step_{color_step}_result"  # Use step result for consistent pipeline access
            code_lines.append(f"mask_img = step_{mask_step}_result")
            code_lines.append("if len(mask_img.shape) == 3:")
            code_lines.append("    mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)")
            code_lines.append("else:")
            code_lines.append("    mask = mask_img.copy()")
            if invert:
                code_lines.append("mask = cv2.bitwise_not(mask)")
            code_lines.append(f"result = cv2.bitwise_and({color_source}, {color_source}, mask=mask)")
            param_info.append({"function": "cv2.bitwise_and", "params": {"color_step": color_step, "mask_step": mask_step}})
        
        # Add ROI apply code if enabled
        roi_apply = self._generate_roi_apply_code(params)
        code_lines.extend(roi_apply)
        
        return {
            "code_lines": code_lines,
            "param_info": param_info,
            "method": method_name
        }
