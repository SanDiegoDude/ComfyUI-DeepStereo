# Start with just basic nodes to identify the issue
try:
    from .nodes.depth_nodes import MiDaSDepthEstimator
    print("✓ Depth nodes loaded")
except Exception as e:
    print(f"✗ Depth nodes failed: {e}")

try:
    from .nodes.texture_nodes import ProceduralTextureGenerator, TextureTransformer, InputToTextureTransformer
    print("✓ Texture nodes loaded")
except Exception as e:
    print(f"✗ Texture nodes failed: {e}")

try:
    from .nodes.stereogram_nodes import StereogramGenerator, RandomDotStereogramGenerator
    print("✓ Stereogram nodes loaded")
except Exception as e:
    print(f"✗ Stereogram nodes failed: {e}")

try:
    from .nodes.utility_nodes import ImageResizer, DepthMapProcessor, RandomNoiseGenerator, BatchImageProcessor
    print("✓ Utility nodes loaded")
except Exception as e:
    print(f"✗ Utility nodes failed: {e}")

# Comment out advanced nodes for now
# try:
#     from .nodes.advanced_texture_nodes import AdvancedTextureMethod2, AdvancedTextureMethod3, TextureBlender, TextureTiler
#     print("✓ Advanced texture nodes loaded")
# except Exception as e:
#     print(f"✗ Advanced texture nodes failed: {e}")

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
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
