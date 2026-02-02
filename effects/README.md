# Effects Package

This folder contains all the image processing effects for the OpenCV Playground application. Effects are dynamically loaded when the application starts.

## SOLID Principles Applied

This effect system follows SOLID design principles:

1. **Single Responsibility Principle (SRP)**: Each effect file handles one category of effects
2. **Open/Closed Principle (OCP)**: Add new effects without modifying existing code
3. **Liskov Substitution Principle (LSP)**: All effects can be used interchangeably
4. **Interface Segregation Principle (ISP)**: Effects implement only required methods
5. **Dependency Inversion Principle (DIP)**: High-level modules depend on abstractions (BaseEffect)

## Adding a New Effect

1. Create a new Python file in this folder (e.g., `my_effect.py`)
2. Import the base class and helpers:
   ```python
   from effects.base_effect import BaseEffect, EffectMethod, EffectParam
   import cv2
   import numpy as np
   ```
3. Create a class that inherits from `BaseEffect`:
   ```python
   class MyEffect(BaseEffect):
       @property
       def category_name(self) -> str:
           return "My Category"
       
       @property
       def category_icon(self) -> str:
           return "🌟"
       
       def get_methods(self):
           return {
               "My Method": EffectMethod(
                   name="My Method",
                   description="Description of the effect",
                   function="cv2.myFunction",
                   params=[
                       EffectParam("param1", "slider", "Parameter Label", 5, 1, 10, 1)
                   ]
               )
           }
       
       def apply(self, image: np.ndarray, method_name: str, params: dict) -> np.ndarray:
           if method_name == "My Method":
               # Apply your effect here
               result = image.copy()
               return result
           return image
       
       def generate_code(self, method_name: str, params: dict) -> dict:
           if method_name == "My Method":
               return {
                   "code_lines": ["result = img.copy()"],
                   "param_info": [{"function": "my_function", "params": {}}],
                   "method": method_name
               }
           return {"code_lines": [], "param_info": [], "method": method_name}
   ```

4. Restart the application - your effect will be automatically discovered!

## Removing an Effect

Simply delete the effect's Python file from this folder and restart the application.

## Effect Parameter Types

- **slider**: Numeric value with min, max, step
  ```python
  EffectParam("param_name", "slider", "Label", default, min, max, step)
  ```

- **dropdown**: Selection from a list of options
  ```python
  EffectParam("param_name", "dropdown", "Label", "default_option", options=["Option1", "Option2"])
  ```

- **checkbox**: Boolean value
  ```python
  EffectParam("param_name", "checkbox", "Label", False)
  ```

## File Structure

```
effects/
├── __init__.py              # Package initialization & discovery
├── base_effect.py           # Abstract base class
├── effect_registry.py       # Effect discovery & management
├── README.md               # This file
│
├── affine_transform.py     # Affine transformations (3-point, shear, etc.)
├── blurring.py             # Blur effects (Gaussian, Median, etc.)
├── color_transformations.py # Color space conversions
├── contour_detection.py    # Contour finding effects
├── edge_detection.py       # Edge detection (Canny, Sobel, etc.)
├── geometric_transformations.py # Resize, Rotate, Flip
├── helpers.py              # Helper effects (Normalize)
├── homography.py           # Perspective/Homography transforms
├── image_adjustments.py    # Brightness, Contrast, Gamma
├── image_features.py       # Feature detection (SIFT, ORB, Harris, etc.)
├── inpainting.py           # Image inpainting effects
├── morphological.py        # Morphological operations
├── noise.py                # Noise addition and removal
├── special_effects.py      # Artistic effects (Sketch, Cartoon, etc.)
└── thresholding.py         # Thresholding effects
```

## Interactive Effects

Some effects require user interaction (drawing masks or selecting points):

### Mask-Based Effects (Inpainting)
Effects that require a drawable mask should set:
```python
@property
def requires_mask(self) -> bool:
    return True
```
The mask will be passed to `apply()` via `params['_mask']`.

### Point-Based Effects (Affine, Homography)
Effects that require point selection should set:
```python
@property
def requires_points(self) -> bool:
    return True

@property
def num_points_required(self) -> int:
    return 6  # e.g., 3 source + 3 destination for affine

def get_point_labels(self, method_name: str) -> list:
    return ["Src P1", "Src P2", "Src P3", "Dst P1", "Dst P2", "Dst P3"]
```
Points will be passed to `apply()` via `params['_points']` as a list of [x, y] coordinates.
├── helpers.py              # Utility effects (Normalize)
├── image_adjustments.py    # Brightness, Contrast, CLAHE
├── morphological.py        # Erosion, Dilation, etc.
├── noise.py                # Noise addition/removal
├── special_effects.py      # Artistic effects (Sketch, Cartoon)
└── thresholding.py         # Binary, Adaptive thresholding
```

## Helper Methods in BaseEffect

- `_ensure_odd(value)`: Ensures kernel size is odd (required by many OpenCV functions)

## Example: Complete Effect File

See any existing effect file (e.g., `blurring.py`) for a complete example.
