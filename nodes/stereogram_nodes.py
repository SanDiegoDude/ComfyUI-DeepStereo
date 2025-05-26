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
                "min_separation": ("INT", {"default": 30, "min": 10, "max": 200, "step": 1}),
                "max_separation": ("INT", {"default": 100, "min": 20, "max": 300, "step": 1}),
                "algorithm": (["standard", "improved", "layered", "central"], {"default": "central"}),
                "reference_type": (["left_to_right", "right_to_left", "center_out"], {"default": "center_out"}),
                "depth_layers": ("INT", {"default": 50, "min": 0, "max": 50, "step": 1, 
                    "tooltip": "Number of distinct depth layers (0 for continuous)"}),
                "layer_smoothing": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Amount of smoothing between layers"}),
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

    def _process_depth_value(self, depth_value, depth_layers, layer_smoothing):
        """Process depth value with optional layering and smoothing"""
        if depth_layers <= 1:
            return depth_value
        
        # Calculate the size of each layer
        layer_size = 1.0 / (depth_layers - 1)
        
        # Find the nearest layers
        base_layer = int(depth_value * (depth_layers - 1))
        base_value = base_layer * layer_size
        
        if base_layer >= depth_layers - 1:
            return 1.0
        
        # Calculate distance to next layer
        next_value = (base_layer + 1) * layer_size
        layer_position = (depth_value - base_value) / layer_size
        
        # Apply smoothing
        if layer_smoothing <= 0:
            return base_value
        elif layer_smoothing >= 1:
            return depth_value
        
        # Smooth transition between layers using cosine interpolation
        t = layer_position
        if layer_smoothing < 1:
            # Adjust the transition width based on smoothing
            threshold = layer_smoothing / 2
            if t < threshold:
                t = 0
            elif t > (1 - threshold):
                t = 1
            else:
                # Normalize t to 0-1 range within the smoothing window
                t = (t - threshold) / (1 - 2 * threshold)
                # Apply cosine interpolation
                t = (1 - np.cos(t * np.pi)) / 2
        
        return base_value + (next_value - base_value) * t

    def generate_stereogram(self, depth_map, texture, min_separation, max_separation, algorithm, 
                          reference_type="left_to_right", depth_layers=0, layer_smoothing=0.5,
                          stretch_texture=False, center_offset=0):
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
            result_pil = self._generate_from_center(depth_pil, texture_pil, min_separation, max_separation, center_offset, depth_layers, layer_smoothing)
        elif reference_type == "right_to_left":
            # Flip the image, generate, then flip back
            depth_pil = depth_pil.transpose(Image.FLIP_LEFT_RIGHT)
            result_pil = self._generate_standard(depth_pil, texture_pil, min_separation, max_separation, depth_layers, layer_smoothing)
            result_pil = result_pil.transpose(Image.FLIP_LEFT_RIGHT)
        else:  # left_to_right or any other type
            if algorithm == "standard":
                result_pil = self._generate_standard(depth_pil, texture_pil, min_separation, max_separation, depth_layers, layer_smoothing)
            elif algorithm == "improved":
                result_pil = self._generate_improved(depth_pil, texture_pil, min_separation, max_separation, depth_layers, layer_smoothing)
            elif algorithm == "layered":
                result_pil = self._generate_layered(depth_pil, texture_pil, min_separation, max_separation, depth_layers, layer_smoothing)
            elif algorithm == "central":
                result_pil = self._generate_central_pattern(depth_pil, texture_pil, min_separation, max_separation, depth_layers, layer_smoothing)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Convert back to ComfyUI tensor
        result_np = np.array(result_pil)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)

    def _generate_standard(self, depth_map_pil, texture_pil, min_sep, max_sep, depth_layers=0, layer_smoothing=0.5):
        """Standard stereogram algorithm"""
        width, height = depth_map_pil.size
        texture_width, texture_height = texture_pil.size
        stereogram_img = Image.new('RGB', (width, height))
        
        depth_pixels = depth_map_pil.load()
        texture_pixels = texture_pil.load()
        output_pixels = stereogram_img.load()

        for y in range(height):
            for x in range(width):
                raw_depth = depth_pixels[x, y] / 255.0
                depth_value_normalized = self._process_depth_value(raw_depth, depth_layers, layer_smoothing)
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

    def _generate_from_center(self, depth_map_pil, texture_pil, min_sep, max_sep, center_offset=0, depth_layers=0, layer_smoothing=0.5):
        """Generate stereogram from center outwards with continuous depth effect"""
        width, height = depth_map_pil.size
        texture_width, texture_height = texture_pil.size
        stereogram_img = Image.new('RGB', (width, height))
        
        depth_pixels = depth_map_pil.load()
        texture_pixels = texture_pil.load()
        output_pixels = stereogram_img.load()

        # Calculate center point
        center_x = width // 2 + center_offset

        # Pre-initialize with smoothly varying texture
        # Use different prime numbers for x and y to create non-repeating patterns
        for y in range(height):
            row_seed = (y * 31) % 257  # Large prime for row variation
            for x in range(width):
                # Create a smooth pattern that varies both horizontally and vertically
                tx = ((x * 17 + row_seed) % texture_width + (x * y) // (width + height)) % texture_width
                ty = ((y * 19 + x * 13) % texture_height + (x * y) // (width + height)) % texture_height
                output_pixels[x, y] = texture_pixels[tx, ty]

        # Process each row independently
        for y in range(height):
            # Create temporary buffer for this row
            row_buffer = [(0,0,0)] * width
            for x in range(width):
                row_buffer[x] = output_pixels[x, y]

            # Process both sides simultaneously to maintain pattern consistency
            left_x = center_x - 1
            right_x = center_x + 1

            while left_x >= 0 or right_x < width:
                # Process left side
                if left_x >= 0:
                    raw_depth = depth_pixels[left_x, y] / 255.0
                    depth_value = self._process_depth_value(raw_depth, depth_layers, layer_smoothing)
                    separation = int(min_sep + (max_sep - min_sep) * depth_value)
                    ref_x = left_x + separation
                    
                    if ref_x < width:
                        row_buffer[left_x] = row_buffer[ref_x]
                    # else keep pre-initialized texture
                    left_x -= 1

                # Process right side
                if right_x < width:
                    raw_depth = depth_pixels[right_x, y] / 255.0
                    depth_value = self._process_depth_value(raw_depth, depth_layers, layer_smoothing)
                    separation = int(min_sep + (max_sep - min_sep) * depth_value)
                    ref_x = right_x - separation
                    
                    if ref_x >= 0:
                        row_buffer[right_x] = row_buffer[ref_x]
                    # else keep pre-initialized texture
                    right_x += 1

            # Copy buffer back to image
            for x in range(width):
                output_pixels[x, y] = row_buffer[x]
        
        return stereogram_img

    def _generate_vertical(self, depth_map_pil, texture_pil, min_sep, max_sep, depth_layers=0, layer_smoothing=0.5):
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
                raw_depth = depth_pixels[x, y] / 255.0
                depth_value_normalized = self._process_depth_value(raw_depth, depth_layers, layer_smoothing)
                current_separation = int(min_sep + (max_sep - min_sep) * depth_value_normalized)
                ref_y = y - current_separation
                if ref_y >= 0:
                    output_pixels[x, y] = output_pixels[x, ref_y]
                else:
                    tx = x % texture_width
                    ty = y % texture_height
                    output_pixels[x, y] = texture_pixels[tx, ty]
        
        return stereogram_img

    def _generate_central_pattern(self, depth_map_pil, texture_pil, min_sep, max_sep, depth_layers=0, layer_smoothing=0.5):
        """Generate stereogram with enhanced pattern variation"""
        width, height = depth_map_pil.size
        texture_width, texture_height = texture_pil.size
        stereogram_img = Image.new('RGB', (width, height))
        
        depth_pixels = depth_map_pil.load()
        texture_pixels = texture_pil.load()
        output_pixels = stereogram_img.load()

        # Pre-initialize with varying texture pattern
        for y in range(height):
            row_seed = (y * 31) % 257  # Large prime for row variation
            for x in range(width):
                # Create a complex pattern that varies both horizontally and vertically
                pattern_x = ((x * 17 + row_seed) % texture_width + (x * y) // (width + height)) % texture_width
                pattern_y = ((y * 19 + x * 13) % texture_height + (x * y) // (width + height)) % texture_height
                output_pixels[x, y] = texture_pixels[pattern_x, pattern_y]

        # Process each row with enhanced pattern consistency
        for y in range(height):
            # Create row buffer for consistent pattern propagation
            row_buffer = [(0,0,0)] * width
            for x in range(width):
                row_buffer[x] = output_pixels[x, y]

            # Process from left to right with pattern awareness
            for x in range(width):
                raw_depth = depth_pixels[x, y] / 255.0
                depth_value = self._process_depth_value(raw_depth, depth_layers, layer_smoothing)
                current_separation = int(min_sep + (max_sep - min_sep) * depth_value)
                
                if x < current_separation:
                    # Keep the pre-initialized pattern
                    continue
                else:
                    ref_x = x - current_separation
                    row_buffer[x] = row_buffer[ref_x]

            # Copy buffer back to image
            for x in range(width):
                output_pixels[x, y] = row_buffer[x]
        
        return stereogram_img

    def _generate_improved(self, depth_map_pil, texture_pil, min_sep, max_sep, depth_layers=0, layer_smoothing=0.5):
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

    def _generate_layered(self, depth_map_pil, texture_pil, min_sep, max_sep, depth_layers=0, layer_smoothing=0.5):
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
                "min_separation": ("INT", {"default": 30, "min": 10, "max": 200, "step": 1}),
                "max_separation": ("INT", {"default": 100, "min": 20, "max": 300, "step": 1}),
                "dot_density": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "depth_layers": ("INT", {"default": 50, "min": 0, "max": 50, "step": 1, 
                    "tooltip": "Number of distinct depth layers (0 for continuous)"}),
                "layer_smoothing": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Amount of smoothing between layers"}),
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
                    depth_layers=0, layer_smoothing=0.5, background_color="#808080", random_seed=0):
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
