import torch
import numpy as np
from PIL import Image, ImageDraw
import random
import math
import sys
import os

# Add the parent directory to sys.path to import deeptexture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deeptexture

class AdvancedTextureMethod2:
    """Density/Size driven texture generation (Method 2)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["density", "size"], {"default": "density"}),
                "element_color": ("STRING", {"default": "white"}),
                "bg_color": ("STRING", {"default": "black"}),
                "base_size": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
            },
            "optional": {
                "max_size": ("INT", {"default": 12, "min": 2, "max": 100, "step": 1}),
                "invert_influence": ("BOOLEAN", {"default": False}),
                "density_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("texture",)
    FUNCTION = "generate_method2_texture"
    CATEGORY = "DeepStereo/Advanced Texture"

    def generate_method2_texture(self, image, mode, element_color, bg_color, base_size, 
                                max_size=12, invert_influence=False, density_factor=1.0):
        
        # Convert ComfyUI tensor to PIL Image
        if len(image.shape) == 4:
            image = image[0]
        
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        # Apply Method 2
        result_pil = deeptexture.apply_method2_density_size(
            img_pil, mode, deeptexture.parse_color(element_color), 
            deeptexture.parse_color(bg_color), base_size, max_size, 
            invert_influence, density_factor
        )
        
        # Convert back to ComfyUI tensor
        result_np = np.array(result_pil)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)


class AdvancedTextureMethod3:
    """Voronoi/Worley noise texture generation (Method 3)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_points": ("INT", {"default": 200, "min": 10, "max": 2000, "step": 10}),
                "metric": (["F1", "F2", "F2-F1"], {"default": "F1"}),
                "color_source": (["distance", "content_point_color", "voronoi_cell_content_color"], 
                               {"default": "distance"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("texture",)
    FUNCTION = "generate_method3_texture"
    CATEGORY = "DeepStereo/Advanced Texture"

    def generate_method3_texture(self, image, num_points, metric, color_source):
        
        # Convert ComfyUI tensor to PIL Image
        if len(image.shape) == 4:
            image = image[0]
        
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        # Apply Method 3
        result_pil = deeptexture.apply_method3_voronoi(
            img_pil, num_points, metric, color_source
        )
        
        # Convert back to ComfyUI tensor
        result_np = np.array(result_pil)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)


class TextureBlender:
    """Blend multiple textures using various blend modes"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texture1": ("IMAGE",),
                "texture2": ("IMAGE",),
                "blend_mode": (["average", "lighten", "darken", "multiply", "screen", "add", "difference", "overlay"], 
                             {"default": "overlay"}),
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_texture",)
    FUNCTION = "blend_textures"
    CATEGORY = "DeepStereo/Advanced Texture"

    def blend_textures(self, texture1, texture2, blend_mode, opacity):
        # Convert ComfyUI tensors to PIL Images
        if len(texture1.shape) == 4:
            texture1 = texture1[0]
        if len(texture2.shape) == 4:
            texture2 = texture2[0]
        
        img1_np = (texture1.cpu().numpy() * 255).astype(np.uint8)
        img2_np = (texture2.cpu().numpy() * 255).astype(np.uint8)
        
        img1_pil = Image.fromarray(img1_np, 'RGB')
        img2_pil = Image.fromarray(img2_np, 'RGB')
        
        # Resize texture2 to match texture1 if needed
        if img1_pil.size != img2_pil.size:
            img2_pil = img2_pil.resize(img1_pil.size, Image.Resampling.LANCZOS)
        
        # Blend using deeptexture function
        result_pil = deeptexture.blend_images(img1_pil, img2_pil, blend_mode, opacity)
        
        # Convert back to ComfyUI tensor
        result_np = np.array(result_pil)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)


class TextureTiler:
    """Create seamlessly tiling textures"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 32}),
                "blend_edges": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "blend_width": ("INT", {"default": 20, "min": 5, "max": 100, "step": 1}),
                "output_tiles_x": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "output_tiles_y": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("tiled_texture", "seamless_tile")
    FUNCTION = "create_tiling_texture"
    CATEGORY = "DeepStereo/Advanced Texture"

    def create_tiling_texture(self, image, tile_size, blend_edges, blend_width=20, 
                             output_tiles_x=2, output_tiles_y=2):
        
        # Convert ComfyUI tensor to PIL Image
        if len(image.shape) == 4:
            image = image[0]
        
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        # Create a seamless tile
        if img_pil.size[0] < tile_size or img_pil.size[1] < tile_size:
            # Scale up if too small
            scale = max(tile_size / img_pil.size[0], tile_size / img_pil.size[1])
            new_size = (int(img_pil.size[0] * scale), int(img_pil.size[1] * scale))
            img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
        
        # Extract a square tile from center
        center_x, center_y = img_pil.size[0] // 2, img_pil.size[1] // 2
        left = center_x - tile_size // 2
        top = center_y - tile_size // 2
        tile_pil = img_pil.crop((left, top, left + tile_size, top + tile_size))
        
        # Make edges
