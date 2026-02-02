"""
Base Effect Module - Abstract base class for all effects

This module defines the interface that all effects must implement.
Following Interface Segregation Principle (ISP) from SOLID.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class EffectParam:
    """Represents a parameter for an effect"""
    name: str
    param_type: str  # "slider", "dropdown", "checkbox"
    label: str
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary format for OPENCV_METHODS config"""
        result = {
            "type": self.param_type,
            "label": self.label,
            "default": self.default
        }
        if self.param_type == "slider":
            result["min"] = self.min_value
            result["max"] = self.max_value
            result["step"] = self.step
        elif self.param_type == "dropdown":
            result["options"] = self.options or []
        return result


@dataclass
class EffectMethod:
    """Represents an effect method within a category"""
    name: str
    description: str
    function: str
    params: List[EffectParam] = field(default_factory=list)
    
    def get_params_dict(self) -> dict:
        """Get parameters as dictionary"""
        return {p.name: p.to_dict() for p in self.params}


class BaseEffect(ABC):
    """
    Abstract base class for all image effects.
    
    Each effect category should inherit from this class and implement
    the required methods. This follows the Dependency Inversion Principle.
    """
    
    @property
    @abstractmethod
    def category_name(self) -> str:
        """Return the category name for this effect group"""
        pass
    
    @property
    @abstractmethod
    def category_icon(self) -> str:
        """Return the icon for this category"""
        pass
    
    @property
    def requires_mask(self) -> bool:
        """
        Return True if this effect requires a drawable mask.
        Override in subclass if mask is needed (e.g., inpainting).
        """
        return False
    
    @property
    def requires_points(self) -> bool:
        """
        Return True if this effect requires point selection.
        Override in subclass if points are needed (e.g., affine, homography).
        """
        return False
    
    @property
    def num_points_required(self) -> int:
        """
        Return the number of points required for this effect.
        Override in subclass to specify (e.g., 6 for affine, 8 for homography).
        """
        return 0
    
    def get_point_labels(self, method_name: str) -> List[str]:
        """
        Return labels for each point to help users.
        Override in subclass to provide meaningful labels.
        """
        return [f"Point {i+1}" for i in range(self.num_points_required)]
    
    @abstractmethod
    def get_methods(self) -> Dict[str, EffectMethod]:
        """Return all methods in this category"""
        pass
    
    @abstractmethod
    def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
        """
        Apply the effect to an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            method_name: Name of the specific method to apply
            params: Dictionary of parameters
            
        Returns:
            Processed image as numpy array
        """
        pass
    
    @abstractmethod
    def generate_code(self, method_name: str, params: dict) -> dict:
        """
        Generate Python code for this effect.
        
        Args:
            method_name: Name of the specific method
            params: Dictionary of parameters
            
        Returns:
            Dictionary with 'code_lines', 'param_info', and 'method' keys
        """
        pass
    
    def get_config(self) -> dict:
        """Get configuration dictionary for backward compatibility with OPENCV_METHODS"""
        methods_dict = {}
        for name, method in self.get_methods().items():
            methods_dict[name] = {
                "description": method.description,
                "function": method.function,
                "params": method.get_params_dict()
            }
        
        return {
            "icon": self.category_icon,
            "methods": methods_dict
        }
    
    def _ensure_odd(self, value: int) -> int:
        """Ensure kernel size is odd"""
        return value + 1 if value % 2 == 0 else value
