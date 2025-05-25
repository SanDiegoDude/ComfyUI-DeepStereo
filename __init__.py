from .nodes.depth_nodes import MiDaSDepthEstimator
from .nodes.texture_nodes import ProceduralTextureGenerator, TextureTransformer, InputToTextureTransformer
from .nodes.stereogram_nodes import StereogramGenerator, RandomDotStereogramGenerator
from .nodes.utility_nodes import ImageResizer, DepthMapProcessor, RandomNoiseGenerator, BatchImageProcessor
from .nodes.advanced_texture_nodes import AdvancedTextureMethod2, AdvancedTextureMethod3, TextureBlender, TextureTiler

NODE_CLASS_MAPPINGS = {
    "MiDaSDepthEstimator": MiDaSDepthEstimator,
    "ProceduralTextureGenerator": ProceduralTextureGenerator,
    "TextureTransformer": TextureTransformer,
    "InputToTextureTransformer": InputToTextureTransformer,
    "StereogramGenerator": StereogramGenerator,
    "RandomDotStereogramGenerator": RandomDotStereogramGenerator,
    "ImageResizer": ImageResizer,
    "DepthMapProcessor": DepthMapProcessor,
    "RandomNoiseGenerator": RandomNoiseGenerator,
    "BatchImageProcessor": BatchImageProcessor,
    "AdvancedTextureMethod2": AdvancedTextureMethod2,
    "AdvancedTextureMethod3": AdvancedTextureMethod3,
    "TextureBlender": TextureBlender,
    "TextureTiler": TextureTiler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiDaSDepthEstimator": "MiDaS Depth Estimator",
    "ProceduralTextureGenerator": "Procedural Texture Generator",
    "TextureTransformer": "Texture Transformer",
    "InputToTextureTransformer": "Input to Texture",
    "StereogramGenerator": "Stereogram Generator",
    "RandomDotStereogramGenerator": "Random Dot Stereogram",
    "ImageResizer": "Image Resizer",
    "DepthMapProcessor": "Depth Map Processor",
    "RandomNoiseGenerator": "Random Noise Generator",
    "BatchImageProcessor": "Batch Image Processor",
    "AdvancedTextureMethod2": "Advanced Texture Method 2",
    "AdvancedTextureMethod3": "Advanced Texture Method 3", 
    "TextureBlender": "Texture Blender",
    "TextureTiler": "Texture Tiler",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
