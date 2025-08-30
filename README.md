# Computer Vision Repository

This repository contains a comprehensive collection of computer vision projects, examples, and implementations covering fundamental concepts to advanced techniques in computer vision and machine learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Exercises](#exercises)
- [Projects](#projects)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## 🔍 Overview

This repository serves as a learning resource and practical implementation guide for computer vision techniques. It includes:

- **Fundamental OpenCV operations** for image processing
- **Classical computer vision algorithms** including feature detection, edge detection, and template matching
- **Machine learning approaches** for image classification and clustering
- **Advanced deep learning projects** including style transfer and few-shot learning
- **Practical exercises** for hands-on learning

## 📁 Repository Structure

```
ComputerVision/
├── Examples/                 # Tutorial notebooks covering CV fundamentals
├── Exercices/               # Practice exercises and assignments
├── ProvaEsame/             # Exam materials and CAML implementation
├── StyleTransfer/          # Neural style transfer implementation
├── TIMO/                   # Text-Image Mutual Guidance Optimization
└── README.md               # This file
```

### Examples/
Contains Jupyter notebooks covering fundamental computer vision concepts:
- **Color Models**: OpenCV color space conversions and manipulations
- **Image Operations**: Drawing, cropping, resizing, and basic transformations
- **Point Operators**: Pixel-level operations and intensity transformations
- **Filtering**: Image smoothing, sharpening, and noise reduction
- **Thresholding**: Binary image creation and segmentation
- **Edge Detection**: Canny, Sobel, and other edge detection algorithms
- **Feature Detection**: Harris corner detector, SIFT, template matching
- **Hough Transform**: Line and shape detection
- **Face Detection**: Haar cascade-based face detection
- **Clustering**: K-means for color quantization and image categorization
- **Classification**: Basic image classification techniques

### Exercices/
Student exercises covering similar topics as Examples, providing hands-on practice with:
- Image processing fundamentals
- Feature detection and matching
- Object detection and recognition
- Advanced computer vision techniques

### ProvaEsame/
Contains exam materials including:
- **CAML**: Cross-modal Adversarial Meta-Learning implementation
- **Requirements**: Dependencies for advanced ML projects
- **Source code**: Complete implementation files

### StyleTransfer/
Neural style transfer implementation using PyTorch:
- **styleTransfer.py**: Complete style transfer pipeline
- **style.jpg**: Sample style image
- Deep learning-based artistic style transfer

### TIMO/
Text-Image Mutual Guidance Optimization for few-shot learning:
- Official PyTorch implementation of TIMO
- Training-free few-shot learning with CLIP
- Enhanced variant TIMO-S
- Complete documentation and examples

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Fonty02/ComputerVision.git
   cd ComputerVision
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv cv_env
   source cv_env/bin/activate  # On Windows: cv_env\Scripts\activate
   ```

3. **Install basic dependencies**:
   ```bash
   pip install opencv-python numpy matplotlib jupyter
   pip install scikit-learn pillow
   ```

4. **For deep learning projects**, install additional dependencies:
   ```bash
   pip install torch torchvision
   pip install transformers datasets
   ```

5. **For TIMO project**, follow specific installation instructions in `TIMO/README.md`

## 📚 Examples

### Basic Usage

Start with the fundamental examples:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and display an image
image = cv2.imread('path/to/image.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
```

### Running Jupyter Notebooks

```bash
jupyter notebook Examples/
```

Navigate to any `.ipynb` file to start learning!

### Key Examples to Try

1. **Start here**: `Examples/1_OpenCV_color_models.ipynb`
2. **Image operations**: `Examples/2_Drawing_cropping_resizing.ipynb`
3. **Feature detection**: `Examples/7a-Harris-corner-detector.ipynb`
4. **Object detection**: `Examples/7b-face-detector.ipynb`
5. **Machine learning**: `Examples/basic_classification.ipynb`

## 🎯 Exercises

The `Exercices/` directory contains practical assignments:
- Complete the provided notebooks
- Implement your own solutions
- Compare results with provided examples

## 🔬 Projects

### Style Transfer
```bash
cd StyleTransfer/
python styleTransfer.py
```

### TIMO (Few-Shot Learning)
```bash
cd TIMO/
# Follow installation instructions in TIMO/README.md
python main.py --config configs/[dataset_name].yaml --shot [shot_number]
```

## 📋 Requirements

### Core Dependencies
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Jupyter
- PIL/Pillow
- scikit-learn

### Deep Learning Projects
- PyTorch
- torchvision
- Transformers
- CLIP (for TIMO)

### Full Requirements
For complete dependency lists, see:
- `ProvaEsame/CAML/req.txt` for CAML project
- `TIMO/requirements.txt` for TIMO project

## 💻 Usage

### For Beginners
1. Start with `Examples/1_OpenCV_color_models.ipynb`
2. Work through examples sequentially
3. Practice with exercises in `Exercices/`

### For Advanced Users
- Jump to specific topics of interest
- Explore TIMO for state-of-the-art few-shot learning
- Implement your own variations of the provided algorithms

### For Researchers
- Use TIMO implementation for few-shot learning research
- Extend the style transfer implementation
- Build upon the CAML framework

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add appropriate documentation
5. Submit a pull request

### Guidelines
- Follow existing code style
- Add comments and documentation
- Include example usage
- Test your implementations

## 🙏 Acknowledgments

This repository builds upon the work of many researchers and developers:

- **OpenCV Community** for fundamental computer vision tools
- **PyTorch Team** for deep learning framework
- **TIMO Authors** for the Text-Image Mutual Guidance Optimization method
- **Style Transfer Research** community for neural artistic techniques

### Citations

If you use the TIMO implementation, please cite:
```bibtex
@article{Li_2024_Text,
  title={Text and Image Are Mutually Beneficial: Enhancing Training-Free Few-Shot Classification with CLIP},
  author={Li, Yayuan and Guo, Jintao and Qi, Lei and Li, Wenbin and Shi, Yinghuan},
  journal={arXiv preprint arXiv:2412.11375},
  year={2024}
}
```

## 📝 License

This repository is for educational and research purposes. Please respect the licenses of individual components and cite appropriately when using in your work.

---

**Happy Learning!** 🎓 Start exploring computer vision with hands-on examples and cutting-edge implementations.