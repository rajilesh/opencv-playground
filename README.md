# 🎨 OpenCV Playground

A beautiful and interactive Streamlit application to explore and apply OpenCV image processing methods in real-time.


[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)](https://opencv-playground.streamlit.app/)

## ✨ Features

- **50+ OpenCV Methods** - Comprehensive collection of image processing techniques
- **Real-time Preview** - See changes instantly as you adjust parameters
- **Chain Operations** - Use processed output as input for next operation
- **Processing History** - Track all your edits with thumbnails
- **Auto-save** - Automatically saves outputs when chaining operations
- **Beautiful UI** - Modern, clean interface with gradient styling

## 🛠️ Available Methods

### 🎨 Color Transformations
- Grayscale, HSV, LAB color conversions
- Color inversion

### 💫 Blurring & Smoothing
- Gaussian Blur, Median Blur
- Bilateral Filter, Box Filter

### 📐 Edge Detection
- Canny Edge Detection
- Sobel X/Y, Laplacian

### ⚫ Thresholding
- Binary, Adaptive (Mean/Gaussian)
- Otsu's Automatic Thresholding

### 🔲 Morphological Operations
- Erosion, Dilation
- Opening, Closing
- Gradient, Top Hat, Black Hat

### 🎚️ Image Adjustments
- Brightness & Contrast
- Histogram Equalization
- CLAHE, Gamma Correction

### 📐 Geometric Transformations
- Resize, Rotate
- Horizontal/Vertical Flip

### ✨ Special Effects
- Sharpen, Emboss
- Sketch Effect, Cartoon Effect
- Sepia, Vignette

### 🔍 Contour Detection
- Find and draw contours with customizable settings

### 📡 Noise
- Add Gaussian/Salt & Pepper noise
- Denoise with Non-local Means

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/rajilesh/opencv-playground.git
cd opencv-playground
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

## 📖 Usage

1. **Upload an Image** - Drag and drop or click to upload (supports JPG, PNG, BMP, TIFF, WebP)

2. **Select a Method** - Choose a category from the left sidebar, then select a specific method

3. **Adjust Parameters** - Use sliders, dropdowns, and checkboxes to fine-tune the effect

4. **Apply & Chain** - Click "Use as Input" to chain multiple operations

5. **Save & Export** - Save to history or download your processed images

## 🎯 Key Features

### Chain Mode
Click "Use as Input" to use the current output as the input for your next operation. This allows you to build complex processing pipelines.

### History Panel
The right sidebar shows thumbnails of all your processed images. You can:
- Load any previous result as the current input
- Download individual images
- Clear history when needed

### Auto-save
When you use "Use as Input", the current output is automatically saved to history.

## 📁 Project Structure

```
opencv-playground-claude/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── outputs/           # Saved processed images (auto-created)
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new OpenCV methods
- Improve the UI/UX
- Fix bugs
- Add video processing support

## 📄 License

MIT License - feel free to use this project for learning and building!

## 🙏 Acknowledgments

- [OpenCV](https://opencv.org/) - For the amazing computer vision library
- [Streamlit](https://streamlit.io/) - For the beautiful web framework
- Built with ❤️ for the image processing community


streamlit run app.py --server.port 8501
