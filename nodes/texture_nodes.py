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
                "dot_bg_color": ("STRING", {"default": "000000", "tooltip": "Background color as hex (without #)"}),
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
                        enable_color_dots=True, dot_density=0.7, dot_size=50, dot_bg_color="000000", 
                        dot_color_mode="transformed_hue", hue_shift_degrees=90.0,
                        enable_glyph_dither=True, glyph_num_colors=8, glyph_size=10, 
                        glyph_style="random_dots", use_quantized_color=True):
        
        # Convert ComfyUI tensor to PIL Image
        if len(image.shape) == 4:
            image = image[0]  # Take first image if batch
        
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        # Convert hex color to RGB tuple for deeptexture
        def hex_to_rgb(hex_color):
            try:
                hex_clean = hex_color.replace("#", "")
                if len(hex_clean) == 3:
                    hex_clean = ''.join([c*2 for c in hex_clean])
                elif len(hex_clean) != 6:
                    hex_clean = "000000"  # Default black
                
                return tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
            except ValueError:
                return (0, 0, 0)  # Default black
        
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
                self.tex_m1_bg_color = hex_to_rgb(dot_bg_color)  # Convert hex to RGB tuple
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
        
        # Generate texture using deeptexture functions
        result_pil = generate_texture_from_config(img_pil, args, verbose=False)
        
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
                "rotation_fill_mode": (["black_fill", "tile", "crop_to_fit"], {"default": "tile", 
                                      "tooltip": "How to handle rotation: black_fill=fill with color, tile=tile rotated image, crop_to_fit=crop to original size"}),
                "rotation_bg_color": ("STRING", {"default": "000000", "tooltip": "Background hex color for black_fill mode (without #)"}),
                "grid_rows": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "grid_cols": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "invert_colors": ("BOOLEAN", {"default": False}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, 
                                       "tooltip": "Stretch to target width (0 = no stretch)"}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, 
                                        "tooltip": "Stretch to target height (0 = no stretch)"}),
                "match_input_dimensions": ("BOOLEAN", {"default": False, 
                                                     "tooltip": "Resize output to match input dimensions (overrides target width/height)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("transformed_texture",)
    FUNCTION = "transform_texture"
    CATEGORY = "DeepStereo/Texture"

    def transform_texture(self, image, rotate_degrees=0, rotation_fill_mode="tile", rotation_bg_color="000000",
                         grid_rows=0, grid_cols=0, invert_colors=False, target_width=0, target_height=0, 
                         match_input_dimensions=False):
        
        # Convert ComfyUI tensor to PIL Image
        if len(image.shape) == 4:
            image = image[0]  # Take first image if batch
        
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        # Store original dimensions
        original_width, original_height = img_pil.size
        
        # Determine target dimensions early for rotation
        if match_input_dimensions:
            final_target_width, final_target_height = original_width, original_height
        elif target_width > 0 or target_height > 0:
            final_target_width = target_width if target_width > 0 else original_width
            final_target_height = target_height if target_height > 0 else original_height
        else:
            final_target_width, final_target_height = original_width, original_height
        
        transformed_image = img_pil.copy()
        
        # Apply rotation with smart filling
        if rotate_degrees != 0:
            transformed_image = self._apply_smart_rotation(
                transformed_image, rotate_degrees, rotation_fill_mode, rotation_bg_color,
                final_target_width, final_target_height
            )
        
        # Apply grid transformation
        if grid_rows > 0 and grid_cols > 0:
            current_width, current_height = transformed_image.size
            cell_width = current_width // grid_cols
            cell_height = current_height // grid_rows
            
            if cell_width > 0 and cell_height > 0:
                cell_texture = transformed_image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
                grid_image = Image.new(transformed_image.mode, (current_width, current_height))
                
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
        
        # Final resize only if rotation didn't already handle it and we need different dimensions
        current_size = transformed_image.size
        if current_size != (final_target_width, final_target_height) and rotate_degrees == 0:
            transformed_image = transformed_image.resize((final_target_width, final_target_height), Image.Resampling.LANCZOS)
        
        # Convert back to ComfyUI tensor
        result_np = np.array(transformed_image)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)  # Add batch dimension
        
        return (result_tensor,)

    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        try:
            hex_clean = hex_color.replace("#", "")
            if len(hex_clean) == 3:
                hex_clean = ''.join([c*2 for c in hex_clean])
            elif len(hex_clean) != 6:
                hex_clean = "000000"  # Default black
            
            return tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            return (0, 0, 0)  # Default black

    def _apply_smart_rotation(self, image, degrees, fill_mode, bg_color_hex, target_width, target_height):
        """Apply rotation with various fill strategies"""
        
        # Convert hex to RGB for fill color
        fill_color_rgb = self._hex_to_rgb(bg_color_hex)
        
        if fill_mode == "tile":
            # Pre-tile before rotation to avoid any background fill
            import math
            
            diagonal_factor = math.sqrt(2)
            safe_size = max(target_width, target_height)
            pre_rotation_size = int(safe_size * diagonal_factor * 1.5)
            
            # Create a larger canvas and tile the original image on it
            large_canvas = Image.new(image.mode, (pre_rotation_size, pre_rotation_size))
            
            # Calculate how many tiles we need
            orig_width, orig_height = image.size
            tiles_x = (pre_rotation_size // orig_width) + 2
            tiles_y = (pre_rotation_size // orig_height) + 2
            
            # Tile the original image across the large canvas
            for tile_y in range(tiles_y):
                for tile_x in range(tiles_x):
                    paste_x = tile_x * orig_width
                    paste_y = tile_y * orig_height
                    if paste_x < pre_rotation_size and paste_y < pre_rotation_size:
                        large_canvas.paste(image, (paste_x, paste_y))
            
            # Rotate the large tiled canvas
            rotated_large = large_canvas.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=False)
            
            # Crop from center to get target dimensions
            crop_x = (pre_rotation_size - target_width) // 2
            crop_y = (pre_rotation_size - target_height) // 2
            
            result = rotated_large.crop((
                crop_x,
                crop_y, 
                crop_x + target_width,
                crop_y + target_height
            ))
            
            return result
            
        elif fill_mode == "crop_to_fit":
            import math
            
            orig_width, orig_height = image.size
            
            # Rotate with expansion and background fill
            rotated_with_bg = image.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill_color_rgb)
            rotated_width, rotated_height = rotated_with_bg.size
            
            # Calculate inscribed rectangle using simple geometry
            angle_rad = math.radians(abs(degrees % 360))
            cos_a = abs(math.cos(angle_rad))
            sin_a = abs(math.sin(angle_rad))
            
            # Conservative calculation for inscribed rectangle
            if cos_a > sin_a:
                inscribed_width = int(orig_width * cos_a)
                inscribed_height = int(orig_height * cos_a)
            else:
                inscribed_width = int(orig_width * sin_a)
                inscribed_height = int(orig_height * sin_a)
            
            # Ensure minimum size and don't exceed rotated dimensions
            inscribed_width = max(50, min(inscribed_width, rotated_width - 20))
            inscribed_height = max(50, min(inscribed_height, rotated_height - 20))
            
            # Crop from center
            center_x = rotated_width // 2
            center_y = rotated_height // 2
            
            crop_left = center_x - inscribed_width // 2
            crop_top = center_y - inscribed_height // 2
            crop_right = crop_left + inscribed_width
            crop_bottom = crop_top + inscribed_height
            
            # Ensure crop coordinates are within bounds
            crop_left = max(0, crop_left)
            crop_top = max(0, crop_top)
            crop_right = min(rotated_width, crop_right)
            crop_bottom = min(rotated_height, crop_bottom)
            
            cropped_content = rotated_with_bg.crop((crop_left, crop_top, crop_right, crop_bottom))
            
            # Scale to target dimensions
            if cropped_content.size != (target_width, target_height):
                final_result = cropped_content.resize((target_width, target_height), Image.Resampling.LANCZOS)
            else:
                final_result = cropped_content
            
            return final_result
            
        else:  # black_fill
            # Fill with specified color and resize to target
            fillcolor = fill_color_rgb
            if image.mode == 'RGBA':
                fillcolor = fill_color_rgb + (255,)  # Add alpha
            
            # Rotate with expansion
            rotated = image.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fillcolor)
            
            # Resize to target dimensions
            if rotated.size != (target_width, target_height):
                rotated = rotated.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            return rotated
            
class InputToTextureTransformer:
    """Transform input image to colored/hazy texture effect"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hex_color": ("STRING", {"default": "0000FF", "tooltip": "Hex color (6 digits, # optional)"}),
            },
            "optional": {
                "darken_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, 
                                          "tooltip": "0.0 = no darkening, 1.0 = maximum darkening"}),
                "blur_radius": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
                "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, 
                                           "tooltip": "Color overlay strength"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("colored_texture",)
    FUNCTION = "transform_to_texture"
    CATEGORY = "DeepStereo/Texture"

    def transform_to_texture(self, image, hex_color, darken_amount=0.0, blur_radius=5, blend_strength=0.5):
        # Convert ComfyUI tensor to PIL Image
        if len(image.shape) == 4:
            image = image[0]
        
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        # Parse hex color (accept with or without #)
        try:
            hex_clean = hex_color.replace("#", "")
            if len(hex_clean) == 3:
                hex_clean = ''.join([c*2 for c in hex_clean])
            elif len(hex_clean) != 6:
                hex_clean = "000080"  # Default dark blue
            
            target_color = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            print(f"Warning: Invalid hex color '{hex_color}'. Using dark blue.")
            target_color = (0, 0, 128)
        
        # Convert to RGB if needed
        working_img = img_pil.convert('RGB')
        
        # Create a solid color overlay
        color_overlay = Image.new('RGB', working_img.size, target_color)
        
        # Blend using overlay mode
        result_img = Image.blend(working_img, color_overlay, blend_strength)
        
        # Apply darkening if requested (0.0 = no change, 1.0 = maximum darkening)
        if darken_amount > 0.0:
            # Convert darken_amount to brightness factor (1.0 = no change, 0.0 = black)
            brightness_factor = 1.0 - darken_amount
            result_img = ImageEnhance.Brightness(result_img).enhance(brightness_factor)
        
        # Apply blur
        if blur_radius > 0:
            result_img = result_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Convert back to ComfyUI tensor
        result_np = np.array(result_img)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)
