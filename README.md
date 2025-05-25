# ComfyUI DeepStereo Nodes

A comprehensive set of ComfyUI nodes for generating AI-powered autostereograms (Magic Eye images) with procedural texture generation. This project brings the power of [DeepStereo](https://github.com/SanDiegoDude/DeepStereo/tree/main) into the ComfyUI node-based workflow environment.

## Overview

Transform any image into mesmerizing autostereograms using AI-generated depth maps and procedurally created textures. These nodes break down the complex stereogram generation process into modular, interconnected components that can be mixed and matched for endless creative possibilities.

## Features

### 🧠 AI Depth Estimation
- **MiDaS Integration**: Uses state-of-the-art MiDaS models for depth estimation
- **Multiple Model Support**: MiDaS_small, DPT_Large, and DPT_Hybrid
- **Smart Model Management**: Automatically saves models to ComfyUI's controlnet directory
- **Flexible Processing**: Configurable input resizing and depth map post-processing

### 🎨 Procedural Texture Generation
- **Method 1**: Content-driven color dots with multiple color modes
- **Method 2**: Density/size-driven elements based on image content
- **Method 3**: Voronoi/Worley noise patterns
- **Method 4**: Stylized glyph dithering
- **Advanced Blending**: Multiple blend modes for combining texture methods
- **Input Transformation**: Convert any image into colored/hazy textures

### ✨ Stereogram Generation
- **Multiple Algorithms**: Standard, improved, and layered generation methods
- **Random Dot Stereograms**: Traditional RDS without texture dependencies
- **Texture Preprocessing**: Automatic seamless tiling for better results
- **Flexible Parameters**: Adjustable separation distances for depth control

### 🔧 Utility Nodes
- **Image Resizing**: Megapixel-based and dimension-based resizing
- **Depth Map Processing**: Contrast, brightness, blur, and clipping controls
- **Texture Transformation**: Rotation, grid patterns, color inversion
- **Random Noise Generation**: Multiple noise types for texture creation
- **Batch Processing**: Apply operations to multiple images simultaneously

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/yourusername/ComfyUI-DeepStereo-Nodes.git
```

2. Install the required dependencies:
```bash
cd ComfyUI-DeepStereo-Nodes
pip install -r requirements.txt
```

3. Restart ComfyUI

**Note**: This extension requires PyTorch, which should already be installed with ComfyUI. If you encounter torch-related issues, ensure your ComfyUI installation includes proper CUDA support for GPU acceleration.

## Quick Start

### Basic Workflow
1. **Load Image** → **MiDaS Depth Estimator** → **Depth Map**
2. **Load Image** → **Procedural Texture Generator** → **Texture**
3. **Depth Map + Texture** → **Stereogram Generator** → **Magic Eye Image**

### Advanced Workflow
- Use **Texture Blender** to combine multiple procedural textures
- Apply **Texture Transformer** for rotation, grids, and effects
- Use **Depth Map Processor** to fine-tune depth perception
- Try **Random Dot Stereogram Generator** for classic RDS effects

## Node Categories

### DeepStereo/Depth
- **MiDaS Depth Estimator**: AI-powered depth map generation

### DeepStereo/Texture
- **Procedural Texture Generator**: Multi-method texture creation
- **Texture Transformer**: Apply effects and transformations
- **Input to Texture**: Convert images to colored textures

### DeepStereo/Generation
- **Stereogram Generator**: Create textured autostereograms
- **Random Dot Stereogram**: Generate classic RDS images

### DeepStereo/Utility
- **Image Resizer**: Flexible image resizing options
- **Depth Map Processor**: Fine-tune depth maps
- **Random Noise Generator**: Create noise textures
- **Batch Image Processor**: Process multiple images

### DeepStereo/Advanced Texture
- **Advanced Texture Method 2**: Density/size-driven patterns
- **Advanced Texture Method 3**: Voronoi/Worley noise
- **Texture Blender**: Combine textures with blend modes
- **Texture Tiler**: Create seamlessly tiling textures

## Requirements

- ComfyUI (with PyTorch)
- OpenCV (`opencv-python>=4.5.0`)
- NumPy (`numpy>=1.19.0`)
- tqdm (`tqdm>=4.50.0`)
- PIL/Pillow (usually included with ComfyUI)

## Credits and Inspiration

This ComfyUI node collection is based on my [DeepStereo](https://github.com/SanDiegoDude/DeepStereo/tree/main) standalone utility. DeepStereo is an integration of AI depth estimation with procedural texture generation for easy and fun autostereogram creation.

**Original DeepStereo Features Translated to Nodes:**
- Complete CLI functionality broken into modular nodes
- All texture generation methods (M1-M4) available
- Multiple stereogram algorithms implemented
- Advanced texture transformation and blending capabilities

If you prefer command-line usage or need batch processing capabilities, check out the original [DeepStereo repository](https://github.com/SanDiegoDude/DeepStereo/tree/main).

## License

This project is open source. See LICENSE file for details.

## Contributing

Issues and pull requests welcome! This project aims to make autostereogram generation accessible to everyone through ComfyUI's intuitive interface.

---

*Create mesmerizing Magic Eye images with the power of AI and procedural generation!*
