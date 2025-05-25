from .nodes.depth_nodes import MiDaSDepthEstimator
from .nodes.texture_nodes import ProceduralTextureGenerator, TextureTransformer, InputToTextureTransformer
from .nodes.stereogram_nodes import StereogramGenerator
from .nodes.utility_nodes import ImageResizer, DepthMapProcessor

NODE_CLASS_MAPPINGS = {
    "MiDaSDepthEstimator": MiDaSDepthEstimator,
    "ProceduralTextureGenerator": ProceduralTextureGenerator,
    "TextureTransformer": TextureTransformer,
    "InputToTextureTransformer": InputToTextureTransformer,
    "StereogramGenerator": StereogramGenerator,
    "ImageResizer": ImageResizer,
    "DepthMapProcessor": DepthMapProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiDaSDepthEstimator": "MiDaS Depth Estimator",
    "ProceduralTextureGenerator": "Procedural Texture Generator",
    "TextureTransformer": "Texture Transformer",
    "InputToTextureTransformer": "Input to Texture",
    "StereogramGenerator": "Stereogram Generator",
    "ImageResizer": "Image Resizer",
    "DepthMapProcessor": "Depth Map Processor",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
