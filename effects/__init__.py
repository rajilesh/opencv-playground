"""
Effects Package - OpenCV Image Processing Effects

This package dynamically loads all effects from the effects folder.
To add a new effect:
1. Create a new Python file in the effects folder
2. Create a class that inherits from BaseEffect
3. The effect will be automatically discovered and loaded

SOLID Principles Applied:
- Single Responsibility: Each effect file handles one category
- Open/Closed: Add new effects without modifying existing code
- Liskov Substitution: All effects can be used interchangeably
- Interface Segregation: Effects implement only required methods
- Dependency Inversion: High-level modules depend on abstractions
"""

from effects.base_effect import BaseEffect, EffectParam
from effects.effect_registry import EffectRegistry

# Create global registry instance
registry = EffectRegistry()

# Load all effects from the effects folder
registry.discover_effects()

def get_all_effects():
    """Get all registered effects organized by category"""
    return registry.get_all_effects()

def get_effect(category: str, method_name: str):
    """Get a specific effect by category and method name"""
    return registry.get_effect(category, method_name)

def apply_effect(image, category: str, method_name: str, params: dict):
    """Apply an effect to an image"""
    effect = registry.get_effect(category, method_name)
    if effect:
        return effect.apply(image, params)
    return image

def get_opencv_methods_config():
    """Get the OPENCV_METHODS configuration for backward compatibility"""
    return registry.get_opencv_methods_config()

def generate_code(method_name: str, params: dict, category: str = None):
    """Generate OpenCV code for a method"""
    effect = registry.get_effect(category, method_name) if category else None
    if effect:
        return effect.generate_code(params)
    return {"code_lines": ["result = img.copy()"], "param_info": [{"function": "copy", "params": {}}], "method": method_name}
