import torch
import numpy as np
from PIL import Image
import random
from tqdm import tqdm

class StereogramGenerator:
    """Generate autostereograms using various algorithms"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("IMAGE",),
                "texture": ("IMAGE",),
                "min_separation": ("INT", {"default": 40, "min": 10, "max": 200, "step": 1}),
                "max_separation": ("INT", {"default": 100, "min": 20, "max": 300, "step": 1}),
                "algorithm": (["standard", "improved", "layered", "central"], {"default": "layered"}),
                "reference_type": (["left_to_right", "right_to_left", "center_out", "vertical"], {"default": "left_to_right"}),
            },
            "optional": {
                "stretch_texture": ("BOOLEAN", {"default": False, "tooltip": "Stretch texture to match depth map size"}),
                "center_offset": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1, "tooltip": "Offset from center for reference point"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stereogram",)
    FUNCTION = "generate_stereogram"
    CATEGORY = "DeepStereo/Generation"

    def generate_stereogram(self, depth_map, texture, min_separation, max_separation, algorithm, reference_type="left_to_right", stretch_texture=False, center_offset=0):
        if min_separation >= max_separation:
            raise ValueError("min_separation must be less than max_separation")
        
        # Convert ComfyUI tensors to PIL Images
        if len(depth_map.shape) == 4:
            depth_map = depth_map[0]
        if len(texture.shape) == 4:
            texture = texture[0]
        
        depth_np = (depth_map.cpu().numpy() * 255).astype(np.uint8)
        texture_np = (texture.cpu().numpy() * 255).astype(np.uint8)
        
        depth_pil = Image.fromarray(depth_np).convert('L')
        texture_pil = Image.fromarray(texture_np).convert('RGB')
        
        # Stretch texture if requested
        if stretch_texture and texture_pil.size != depth_pil.size:
            texture_pil = texture_pil.resize(depth_pil.size, Image.Resampling.LANCZOS)
        
        # Generate stereogram based on algorithm and reference type
        if reference_type == "center_out":
            result_pil = self._generate_from_center(depth_pil, texture_pil, min_separation, max_separation, center_offset)
        elif reference_type == "right_to_left":
            # Flip the image, generate, then flip back
            depth_pil = depth_pil.transpose(Image.FLIP_LEFT_RIGHT)
            result_pil = self._generate_standard(depth_pil, texture_pil, min_separation, max_separation)
            result_pil = result_pil.transpose(Image.FLIP_LEFT_RIGHT)
        elif reference_type == "vertical":
            result_pil = self._generate_vertical(depth_pil, texture_pil, min_separation, max_separation)
        else:  # left_to_right
            if algorithm == "standard":
                result_pil = self._generate_standard(depth_pil, texture_pil, min_separation, max_separation)
            elif algorithm == "improved":
                result_pil = self._generate_improved(depth_pil, texture_pil, min_separation, max_separation)
            elif algorithm == "layered":
                result_pil = self._generate_layered(depth_pil, texture_pil, min_separation, max_separation)
            elif algorithm == "central":
                result_pil = self._generate_central_pattern(depth_pil, texture_pil, min_separation, max_separation)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Convert back to ComfyUI tensor
        result_np = np.array(result_pil)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)

    def _generate_standard(self, depth_map_pil, texture_pil, min_sep, max_sep):
        """Standard stereogram algorithm"""
        width, height = depth_map_pil.size
        texture_width, texture_height = texture_pil.size
        stereogram_img = Image.new('RGB', (width, height))
        
        depth_pixels = depth_map_pil.load()
        texture_pixels = texture_pil.load()
        output_pixels = stereogram_img.load()

        for y in range(height):
            for x in range(width):
                depth_value_normalized = depth_pixels[x, y] / 255.0
                current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)
                current_separation = max(1, current_separation)
                
                if x < current_separation:
                    tx = x % texture_width
                    ty = y % texture_height
                    output_pixels[x, y] = texture_pixels[tx, ty]
                else:
                    ref_x = x - current_separation
                    output_pixels[x, y] = output_pixels[ref_x, y]
        
        return stereogram_img

    def _generate_from_center(self, depth_map_pil, texture_pil, min_sep, max_sep, center_offset=0):
        """Generate stereogram from center outwards"""
        width, height = depth_map_pil.size
        texture_width, texture_height = texture_pil.size
        stereogram_img = Image.new('RGB', (width, height))
        
        depth_pixels = depth_map_pil.load()
        texture_pixels = texture_pil.load()
        output_pixels = stereogram_img.load()

        # Calculate center point
        center_x = width // 2 + center_offset
        
        # Initialize center strip with texture
        for y in range(height):
            for x in range(max(0, center_x - max_sep), min(width, center_x + max_sep)):
                tx = x % texture_width
                ty = y % texture_height
                output_pixels[x, y] = texture_pixels[tx, ty]
        
        # Generate left side
        for y in range(height):
            for x in range(center_x - max_sep - 1, -1, -1):
                depth_value_normalized = depth_pixels[x, y] / 255.0
                current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)
                ref_x = x + current_separation
                if ref_x < width:
                    output_pixels[x, y] = output_pixels[ref_x, y]
                else:
                    tx = x % texture_width
                    ty = y % texture_height
                    output_pixels[x, y] = texture_pixels[tx, ty]
        
        # Generate right side
        for y in range(height):
            for x in range(center_x + max_sep, width):
                depth_value_normalized = depth_pixels[x, y] / 255.0
                current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)
                ref_x = x - current_separation
                if ref_x >= 0:
                    output_pixels[x, y] = output_pixels[ref_x, y]
                else:
                    tx = x % texture_width
                    ty = y % texture_height
                    output_pixels[x, y] = texture_pixels[tx, ty]
        
        return stereogram_img

    def _generate_vertical(self, depth_map_pil, texture_pil, min_sep, max_sep):
        """Generate stereogram using vertical parallax"""
        width, height = depth_map_pil.size
        texture_width, texture_height = texture_pil.size
        stereogram_img = Image.new('RGB', (width, height))
        
        depth_pixels = depth_map_pil.load()
        texture_pixels = texture_pil.load()
        output_pixels = stereogram_img.load()

        # Initialize top strip
        for x in range(width):
            for y in range(max_sep):
                tx = x % texture_width
                ty = y % texture_height
                output_pixels[x, y] = texture_pixels[tx, ty]
        
        # Generate rest of image
        for x in range(width):
            for y in range(max_sep, height):
                depth_value_normalized = depth_pixels[x, y] / 255.0
                current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)
                ref_y = y - current_separation
                if ref_y >= 0:
                    output_pixels[x, y] = output_pixels[x, ref_y]
                else:
                    tx = x % texture_width
                    ty = y % texture_height
                    output_pixels[x, y] = texture_pixels[tx, ty]
        
        return stereogram_img

    def _generate_central_pattern(self, depth_map_pil, texture_pil, min_sep, max_sep):
        """Generate stereogram with a central pattern focus"""
        width, height = depth_map_pil.size
        texture_width, texture_height = texture_pil.size
        stereogram_img = Image.new('RGB', (width, height))
        
        depth_pixels = depth_map_pil.load()
        texture_pixels = texture_pil.load()
        output_pixels = stereogram_img.load()

        # Create a central pattern region
        center_x = width // 2
        pattern_width = max_sep * 2
        
        # Initialize with unique patterns in the center
        for y in range(height):
            for x in range(max(0, center_x - pattern_width), min(width, center_x + pattern_width)):
                # Create more varied pattern in center
                tx = ((x * 17 + y * 13) % texture_width + x // 4) % texture_width
                ty = ((y * 19 + x * 11) % texture_height + y // 4) % texture_height
                output_pixels[x, y] = texture_pixels[tx, ty]
        
        # Generate outwards from center
        for y in range(height):
            # Left side
            for x in range(center_x - pattern_width - 1, -1, -1):
                depth_value_normalized = depth_pixels[x, y] / 255.0
                current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)
                ref_x = x + current_separation
                if ref_x < width:
                    output_pixels[x, y] = output_pixels[ref_x, y]
                
            # Right side
            for x in range(center_x + pattern_width, width):
                depth_value_normalized = depth_pixels[x, y] / 255.0
                current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)
                ref_x = x - current_separation
                if ref_x >= 0:
                    output_pixels[x, y] = output_pixels[ref_x, y]
        
        return stereogram_img

    def _generate_improved(self, depth_map_pil, texture_pil, min_sep, max_sep):
        """Improved algorithm with better texture distribution"""
        width, height = depth_map_pil.size
        texture_width, texture_height = texture_pil.size
        stereogram_img = Image.new('RGB', (width, height))
        
        depth_pixels = depth_map_pil.load()
        texture_pixels = texture_pil.load()
        output_pixels = stereogram_img.load()

        avg_sep = (min_sep + max_sep) // 2
        effective_texture_width = min(texture_width, avg_sep * 2)

        for y in range(height):
            # Shift texture start position per row
            row_offset = (y * 17) % effective_texture_width
            
            for x in range(width):
                depth_value_normalized = depth_pixels[x, y] / 255.0
                current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)
                current_separation = max(1, current_separation)
                
                constraint_x = x - current_separation
                
                if constraint_x >= 0:
                    output_pixels[x, y] = output_pixels[constraint_x, y]
                else:
                    texture_x = (x + row_offset) % effective_texture_width
                    wp_x = texture_x % texture_width
                    wp_y = y % texture_height
                    output_pixels[x, y] = texture_pixels[wp_x, wp_y]
        
        return stereogram_img

    def _prepare_texture_for_stereogram(self, texture_pil, min_sep, max_sep):
        """Pre-process texture to tile seamlessly"""
        avg_sep = (min_sep + max_sep) // 2
        texture_width, texture_height = texture_pil.size
        
        if texture_width <= avg_sep:
            return texture_pil
        
        new_width = avg_sep
        new_texture = Image.new('RGB', (new_width, texture_height))
        blend_width = min(20, avg_sep // 4)
        
        for y in range(texture_height):
            for x in range(new_width):
                if x < new_width - blend_width:
                    new_texture.putpixel((x, y), texture_pil.getpixel((x, y)))
                else:
                    blend_factor = (x - (new_width - blend_width)) / blend_width
                    color1 = texture_pil.getpixel((x, y))
                    color2 = texture_pil.getpixel((x - new_width, y))
                    blended = tuple(int(c1 * (1 - blend_factor) + c2 * blend_factor) 
                                  for c1, c2 in zip(color1, color2))
                    new_texture.putpixel((x, y), blended)
        
        return new_texture

    def _generate_layered(self, depth_map_pil, texture_pil, min_sep, max_sep):
        """Layered algorithm with texture preprocessing"""
        # Pre-process texture
        preprocessed_texture = self._prepare_texture_for_stereogram(texture_pil, min_sep, max_sep)
        
        width, height = depth_map_pil.size
        texture_width, texture_height = preprocessed_texture.size
        stereogram_img = Image.new('RGB', (width, height))
        
        depth_pixels = depth_map_pil.load()
        texture_pixels = preprocessed_texture.load()
        output_pixels = stereogram_img.load()

        for y in range(height):
            for x in range(width):
                depth_value_normalized = depth_pixels[x, y] / 255.0
                current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)
                current_separation = max(1, current_separation)
                
                constraint_x = x - current_separation
                
                if constraint_x >= 0:
                    output_pixels[x, y] = output_pixels[constraint_x, y]
                else:
                    wp_x = x % texture_width
                    wp_y = y % texture_height
                    output_pixels[x, y] = texture_pixels[wp_x, wp_y]
        
        return stereogram_img


class RandomDotStereogramGenerator:
    """Generate random dot stereograms (RDS) without texture"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("IMAGE",),
                "min_separation": ("INT", {"default": 40, "min": 10, "max": 200, "step": 1}),
                "max_separation": ("INT", {"default": 100, "min": 20, "max": 300, "step": 1}),
                "dot_density": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "background_color": ("STRING", {"default": "#808080", "tooltip": "Hex color for background"}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "tooltip": "0 for random seed"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rds_stereogram",)
    FUNCTION = "generate_rds"
    CATEGORY = "DeepStereo/Generation"

    def generate_rds(self, depth_map, min_separation, max_separation, dot_density, 
                    background_color="#808080", random_seed=0):
        if min_separation >= max_separation:
            raise ValueError("min_separation must be less than max_separation")
        
        # Set random seed if specified
        if random_seed > 0:
            random.seed(random_seed)
        
        # Convert ComfyUI tensor to PIL Image
        if len(depth_map.shape) == 4:
            depth_map = depth_map[0]
        
        depth_np = (depth_map.cpu().numpy() * 255).astype(np.uint8)
        depth_pil = Image.fromarray(depth_np).convert('L')
        
        # Parse background color
        try:
            if not background_color.startswith('#'):
                background_color = '#' + background_color
            from PIL import ImageColor
            bg_rgb = ImageColor.getrgb(background_color)
        except:
            bg_rgb = (128, 128, 128)  # Default gray
        
        width, height = depth_pil.size
        stereogram_img = Image.new('RGB', (width, height), bg_rgb)
        
        depth_pixels = depth_pil.load()
        output_pixels = stereogram_img.load()

        for y in range(height):
            for x in range(width):
                depth_value_normalized = depth_pixels[x, y] / 255.0
                current_separation = int(min_separation + (max_separation - min_separation) * depth_value_normalized)
                current_separation = max(1, current_separation)
                
                constraint_x = x - current_separation
                
                if constraint_x >= 0:
                    output_pixels[x, y] = output_pixels[constraint_x, y]
                else:
                    if random.random() < dot_density:
                        # Random colored dot
                        output_pixels[x, y] = (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255)
                        )
                    else:
                        # Background color
                        output_pixels[x, y] = bg_rgb
        
        # Convert back to ComfyUI tensor
        result_np = np.array(stereogram_img)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)
