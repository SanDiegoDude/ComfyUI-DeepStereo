import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops, ImageColor, ImageEnhance
import random
import math
import os

# Import deeptexture functions from our local module
from .deeptexture_functions import (
    parse_color, 
    get_pixel_value_safe, 
    resize_to_megapixels,
    generate_texture_from_config
)

class ProceduralTextureGenerator:
    """Node for generating procedural textures using the deeptexture methods"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "combination_mode": (["sequential", "blend"], {"default": "blend"}),
            },
            "optional": {
                "blend_type": (["average", "lighten", "darken", "multiply", "screen", "add", "difference", "overlay"], {"default": "average"}),
                "blend_opacity": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # Method 1: Color Dots
                "enable_color_dots": ("BOOLEAN", {"default": True}),
                "dot_density": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "dot_size": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "dot_bg_color": ("STRING", {"default": "black"}),
                "dot_color_mode": (["content_pixel", "random_rgb", "random_from_palette", "transformed_hue", "transformed_invert"], {"default": "transformed_hue"}),
                "hue_shift_degrees": ("FLOAT", {"default": 90.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                
                # Method 4: Glyph Dither
                "enable_glyph_dither": ("BOOLEAN", {"default": True}),
                "glyph_num_colors": ("INT", {"default": 8, "min": 2, "max": 32, "step": 1}),
                "glyph_size": ("INT", {"default": 10, "min": 2, "max": 50, "step": 1}),
                "glyph_style": (["random_dots", "lines", "circles", "solid"], {"default": "random_dots"}),
                "use_quantized_color": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("texture",)
    FUNCTION = "generate_texture"
    CATEGORY = "DeepStereo/Texture"

    def generate_texture(self, image, max_megapixels, combination_mode, blend_type="average", blend_opacity=0.75,
                        enable_color_dots=True, dot_density=0.7, dot_size=50, dot_bg_color="black", 
                        dot_color_mode="transformed_hue", hue_shift_degrees=90.0,
                        enable_glyph_dither=True, glyph_num_colors=8, glyph_size=10, 
                        glyph_style="random_dots", use_quantized_color=True):
        
        # Convert ComfyUI tensor to PIL Image
        if len(image.shape) == 4:
            image = image[0]  # Take first image if batch
        
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        # Create args object for deeptexture
        class TextureArgs:
            def __init__(self):
                self.tex_max_megapixels = max_megapixels
                self.tex_combination_mode = combination_mode
                self.tex_blend_type = blend_type
                self.tex_blend_opacity = blend_opacity
                
                # Method 1 args
                self.tex_method1_color_dots = enable_color_dots
                self.tex_m1_density = dot_density
                self.tex_m1_dot_size = dot_size
                self.tex_m1_bg_color = dot_bg_color
                self.tex_m1_color_mode = dot_color_mode
                self.tex_m1_hue_shift_degrees = hue_shift_degrees
                
                # Method 2 (disabled by default for this simplified node)
                self.tex_method2_density_size = False
                
                # Method 3 (disabled by default for this simplified node)
                self.tex_method3_voronoi = False
                
                # Method 4 args
                self.tex_method4_glyph_dither = enable_glyph_dither
                self.tex_m4_num_colors = glyph_num_colors
                self.tex_m4_glyph_size = glyph_size
                self.tex_m4_glyph_style = glyph_style
                self.tex_m4_use_quantized_color_for_glyph_element = use_quantized_color
        
        args = TextureArgs()
        
        # Generate texture using deeptexture
        result_pil = deeptexture.generate_texture_from_config(img_pil, args, verbose=False)
        
        # Convert back to ComfyUI tensor
        result_np = np.array(result_pil)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)  # Add batch dimension
        
        return (result_tensor,)


class TextureTransformer:
    """Node for applying final transforms to textures (rotate, grid, invert, stretch)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "rotate_degrees": ("INT", {"default": 0, "min": 0, "max": 359, "step": 1}),
                "grid_rows": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "grid_cols": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "invert_colors": ("BOOLEAN", {"default": False}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, 
                                       "tooltip": "Stretch to target width (0 = no stretch)"}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, 
                                        "tooltip": "Stretch to target height (0 = no stretch)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("transformed_texture",)
    FUNCTION = "transform_texture"
    CATEGORY = "DeepStereo/Texture"

    def transform_texture(self, image, rotate_degrees=0, grid_rows=0, grid_cols=0, 
                         invert_colors=False, target_width=0, target_height=0):
        
        # Convert ComfyUI tensor to PIL Image
        if len(image.shape) == 4:
            image = image[0]  # Take first image if batch
        
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        transformed_image = img_pil.copy()
        
        # Apply rotation
        if rotate_degrees != 0:
            fillcolor = (0, 0, 0)
            if transformed_image.mode == 'RGBA':
                fillcolor = (0, 0, 0, 0)
            transformed_image = transformed_image.rotate(
                rotate_degrees, 
                resample=Image.Resampling.BICUBIC, 
                expand=True, 
                fillcolor=fillcolor
            )
        
        # Apply grid transformation
        if grid_rows > 0 and grid_cols > 0:
            original_width, original_height = transformed_image.size
            cell_width = original_width // grid_cols
            cell_height = original_height // grid_rows
            
            if cell_width > 0 and cell_height > 0:
                cell_texture = transformed_image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
                grid_image = Image.new(transformed_image.mode, (original_width, original_height))
                
                for r in range(grid_rows):
                    for c in range(grid_cols):
                        grid_image.paste(cell_texture, (c * cell_width, r * cell_height))
                
                transformed_image = grid_image
        
        # Apply color inversion
        if invert_colors:
            if transformed_image.mode == 'L':
                transformed_image = ImageOps.invert(transformed_image)
            elif transformed_image.mode == 'RGB':
                transformed_image = ImageChops.invert(transformed_image)
            elif transformed_image.mode == 'RGBA':
                r, g, b, a = transformed_image.split()
                r_inv, g_inv, b_inv = ImageChops.invert(r), ImageChops.invert(g), ImageChops.invert(b)
                transformed_image = Image.merge('RGBA', (r_inv, g_inv, b_inv, a))
        
        # Apply stretching if target dimensions are specified
        if target_width > 0 or target_height > 0:
            current_width, current_height = transformed_image.size
            new_width = target_width if target_width > 0 else current_width
            new_height = target_height if target_height > 0 else current_height
            transformed_image = transformed_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert back to ComfyUI tensor
        result_np = np.array(transformed_image)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)  # Add batch dimension
        
        return (result_tensor,)


class InputToTextureTransformer:
    """Transform input image to colored/hazy texture effect"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hex_color": ("STRING", {"default": "#0000FF", "tooltip": "Hex color (e.g., #FF0000 or FF0000)"}),
            },
            "optional": {
                "darken_factor": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, 
                                          "tooltip": "0.0 = black, 1.0 = no change"}),
                "blur_radius": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
                "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, 
                                           "tooltip": "Color overlay strength"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("colored_texture",)
    FUNCTION = "transform_to_texture"
    CATEGORY = "DeepStereo/Texture"

    def transform_to_texture(self, image, hex_color, darken_factor=0.3, blur_radius=5, blend_strength=0.5):
        # Convert ComfyUI tensor to PIL Image
        if len(image.shape) == 4:
            image = image[0]  # Take first image if batch
        
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        # Parse hex color
        try:
            if not hex_color.startswith('#'):
                hex_color = '#' + hex_color
            target_color = ImageColor.getrgb(hex_color)
        except ValueError:
            print(f"Warning: Invalid hex color '{hex_color}'. Using dark blue (#000080).")
            target_color = (0, 0, 128)
        
        # Convert to RGB if needed
        working_img = img_pil.convert('RGB')
        
        # Create a solid color overlay
        color_overlay = Image.new('RGB', working_img.size, target_color)
        
        # Blend using overlay mode
        result_img = Image.blend(working_img, color_overlay, blend_strength)
        
        # Apply subtle darkening
        if darken_factor < 1.0:
            result_img = ImageEnhance.Brightness(result_img).enhance(darken_factor)
        
        # Apply blur
        if blur_radius > 0:
            result_img = result_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Convert back to ComfyUI tensor
        result_np = np.array(result_img)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)  # Add batch dimension
        
        return (result_tensor,)
