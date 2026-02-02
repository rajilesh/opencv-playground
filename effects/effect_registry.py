"""
Effect Registry Module - Discovers and manages all effects

This module implements the Open/Closed Principle - open for extension
(new effects can be added), closed for modification (existing code
doesn't need to change).
"""

import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, Optional, List
from effects.base_effect import BaseEffect


class EffectRegistry:
    """
    Registry for discovering and managing all effects.
    
    This class automatically discovers all effect classes in the effects folder
    and registers them for use. New effects can be added simply by creating
    new Python files in the effects folder.
    """
    
    def __init__(self):
        self._effects: Dict[str, BaseEffect] = {}
        self._method_to_category: Dict[str, str] = {}
    
    def register(self, effect: BaseEffect) -> None:
        """Register an effect instance"""
        self._effects[effect.category_name] = effect
        # Map each method to its category for quick lookup
        for method_name in effect.get_methods().keys():
            self._method_to_category[method_name] = effect.category_name
    
    def discover_effects(self) -> None:
        """
        Discover and load all effect modules from the effects folder.
        
        This method scans the effects directory for Python files,
        imports them, and registers any classes that inherit from BaseEffect.
        """
        effects_dir = Path(__file__).parent
        
        # Get all Python files in the effects directory
        for file_path in effects_dir.glob("*.py"):
            filename = file_path.stem
            
            # Skip special files
            if filename.startswith("_") or filename in ["base_effect", "effect_registry"]:
                continue
            
            try:
                # Import the module
                module = importlib.import_module(f"effects.{filename}")
                
                # Find all classes that inherit from BaseEffect
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseEffect) and obj is not BaseEffect:
                        # Create an instance and register it
                        effect_instance = obj()
                        self.register(effect_instance)
                        
            except Exception as e:
                print(f"Warning: Failed to load effect module '{filename}': {e}")
    
    def get_effect(self, category: str, method_name: str) -> Optional[BaseEffect]:
        """Get an effect by category name"""
        return self._effects.get(category)
    
    def get_effect_by_method(self, method_name: str) -> Optional[BaseEffect]:
        """Get an effect by method name"""
        category = self._method_to_category.get(method_name)
        if category:
            return self._effects.get(category)
        return None
    
    def get_all_effects(self) -> Dict[str, BaseEffect]:
        """Get all registered effects"""
        return self._effects.copy()
    
    def get_categories(self) -> List[str]:
        """Get list of all category names"""
        return list(self._effects.keys())
    
    def get_opencv_methods_config(self) -> dict:
        """
        Get configuration in the format of the original OPENCV_METHODS.
        
        This provides backward compatibility with existing code.
        """
        config = {}
        for category_name, effect in self._effects.items():
            config[category_name] = effect.get_config()
        return config
    
    def apply_effect(self, image, category: str, method_name: str, params: dict):
        """Apply an effect to an image"""
        effect = self.get_effect(category, method_name)
        if effect:
            return effect.apply(image, method_name, params)
        return image
    
    def generate_code(self, category: str, method_name: str, params: dict) -> dict:
        """Generate code for an effect"""
        effect = self.get_effect(category, method_name)
        if effect:
            return effect.generate_code(method_name, params)
        return {
            "code_lines": ["result = img.copy()"],
            "param_info": [{"function": "copy", "params": {}}],
            "method": method_name
        }
    
    def effect_requires_mask(self, category: str) -> bool:
        """Check if an effect category requires a mask"""
        effect = self._effects.get(category)
        if effect:
            return getattr(effect, 'requires_mask', False)
        return False
    
    def effect_requires_points(self, category: str) -> bool:
        """Check if an effect category requires point selection"""
        effect = self._effects.get(category)
        if effect:
            return getattr(effect, 'requires_points', False)
        return False
    
    def get_num_points_required(self, category: str) -> int:
        """Get the number of points required for an effect"""
        effect = self._effects.get(category)
        if effect:
            return getattr(effect, 'num_points_required', 0)
        return 0
    
    def get_point_labels(self, category: str, method_name: str) -> list:
        """Get point labels for an effect"""
        effect = self._effects.get(category)
        if effect and hasattr(effect, 'get_point_labels'):
            return effect.get_point_labels(method_name)
        return []
