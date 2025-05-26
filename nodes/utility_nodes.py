import torch
import numpy as np
from PIL import Image, ImageFilter
import random

class ImageResizeAndTransform:
    """Resize and transform images with optional upscaler model support"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "operation": (["resize_only", "resize_with_upscaler", "rotate", "flip_horizontal", "flip_vertical"], {"default": "resize_only"}),
            },
            "optional": {
                # Resize options
                "target_megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1, 
                                              "tooltip": "Target MP (0 = use width/height instead)"}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1, 
                                       "tooltip": "Target width (0 = auto from height or MP)"}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1, 
                                        "tooltip": "Target height (0 = auto from width or MP)"}),
                "maintain_aspect": ("BOOLEAN", {"default": True}),
                "resample_method": (["LANCZOS", "BICUBIC", "BILINEAR", "NEAREST"], {"default": "LANCZOS"}),
                
                # Transform options
                "rotation_degrees": ("INT", {"default": 90, "min": 0, "max": 359, "step": 1}),
                
                # Upscaler options
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Connect from Load Upscale Model node"}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("processed_image", "width", "height")
    FUNCTION = "process_image"
    CATEGORY = "DeepStereo/Utility"

    def process_image(self, image, operation, target_megapixels=0.0, target_width=0, target_height=0, 
                     maintain_aspect=True, resample_method="LANCZOS", rotation_degrees=90,
                     upscale_model=None, upscale_factor=2.0):
        
        # Handle batch input
        if len(image.shape) == 4 and image.shape[0] > 1:
            # Process batch
            processed_images = []
            for i in range(image.shape[0]):
                single_img = image[i:i+1]  # Keep batch dimension
                result, w, h = self._process_single_image(
                    single_img, operation, target_megapixels, target_width, target_height,
                    maintain_aspect, resample_method, rotation_degrees, upscale_model, upscale_factor
                )
                processed_images.append(result[0])  # Remove batch dimension for stacking
            
            batch_result = torch.stack(processed_images, dim=0)
            return (batch_result, w, h)
        else:
            return self._process_single_image(
                image, operation, target_megapixels, target_width, target_height,
                maintain_aspect, resample_method, rotation_degrees, upscale_model, upscale_factor
            )

    def _process_single_image(self, image, operation, target_megapixels, target_width, target_height,
                             maintain_aspect, resample_method, rotation_degrees, upscale_model, upscale_factor):
        
        # Convert ComfyUI tensor to PIL Image
        if len(image.shape) == 4:
            image = image[0]
        
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        if operation == "resize_with_upscaler" and upscale_model is not None:
            # Use upscaler model
            # Convert back to tensor for upscaler
            img_tensor = torch.from_numpy(img_np).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC -> BCHW
            
            # Apply upscaler
            try:
                from comfy.model_management import get_torch_device
                device = get_torch_device()
                img_tensor = img_tensor.to(device)
                upscale_model.to(device)
                
                with torch.no_grad():
                    upscaled = upscale_model(img_tensor)
                
                # Convert back to PIL
                upscaled = upscaled.squeeze(0).permute(1, 2, 0).cpu().numpy()
                upscaled = (upscaled * 255).astype(np.uint8)
                img_pil = Image.fromarray(upscaled, 'RGB')
                
            except Exception as e:
                print(f"Upscaler failed: {e}, falling back to PIL resize")
                new_size = (int(img_pil.width * upscale_factor), int(img_pil.height * upscale_factor))
                img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
        
        elif operation == "resize_only":
            # Standard resize logic
            original_width, original_height = img_pil.size
            
            if target_megapixels > 0:
                current_mp = (original_width * original_height) / 1_000_000.0
                if current_mp > target_megapixels:
                    import math
                    scale_factor = math.sqrt(target_megapixels / current_mp)
                    new_width = int(original_width * scale_factor)
                    new_height = int(original_height * scale_factor)
                else:
                    new_width, new_height = original_width, original_height
            elif target_width > 0 or target_height > 0:
                if maintain_aspect:
                    if target_width > 0 and target_height > 0:
                        scale_w = target_width / original_width
                        scale_h = target_height / original_height
                        scale = min(scale_w, scale_h)
                        new_width = int(original_width * scale)
                        new_height = int(original_height * scale)
                    elif target_width > 0:
                        scale = target_width / original_width
                        new_width = target_width
                        new_height = int(original_height * scale)
                    else:
                        scale = target_height / original_height
                        new_width = int(original_width * scale)
                        new_height = target_height
                else:
                    new_width = target_width if target_width > 0 else original_width
                    new_height = target_height if target_height > 0 else original_height
            else:
                new_width, new_height = original_width, original_height
            
            if (new_width, new_height) != (original_width, original_height):
                resample_map = {
                    "LANCZOS": Image.Resampling.LANCZOS,
                    "BICUBIC": Image.Resampling.BICUBIC,
                    "BILINEAR": Image.Resampling.BILINEAR,
                    "NEAREST": Image.Resampling.NEAREST,
                }
                resample = resample_map.get(resample_method, Image.Resampling.LANCZOS)
                img_pil = img_pil.resize((new_width, new_height), resample)
        
        elif operation == "rotate":
            img_pil = img_pil.rotate(rotation_degrees, expand=True)
        elif operation == "flip_horizontal":
            img_pil = img_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif operation == "flip_vertical":
            img_pil = img_pil.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        # Convert back to ComfyUI tensor
        result_np = np.array(img_pil)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor, img_pil.width, img_pil.height)


class ColorPickerNode:
    """Generate hex color codes from RGB sliders or presets"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color_mode": (["rgb_sliders", "preset_colors", "hex_input"], {"default": "rgb_sliders"}),
            },
            "optional": {
                # RGB sliders
                "red": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                
                # Preset colors
                "preset_color": (["red", "green", "blue", "cyan", "magenta", "yellow", "orange", "purple", 
                                "dark_red", "dark_green", "dark_blue", "light_gray", "dark_gray", "black", "white"], 
                               {"default": "blue"}),
                
                # Direct hex input (without #)
                "hex_input": ("STRING", {"default": "0000FF", "tooltip": "6-digit hex code without #"}),
                
                # Output format
                "include_hash": ("BOOLEAN", {"default": True, "tooltip": "Include # in hex output"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("hex_color", "red", "green", "blue")
    FUNCTION = "generate_color"
    CATEGORY = "DeepStereo/Utility"

    def generate_color(self, color_mode, red=0, green=0, blue=255, preset_color="blue", 
                      hex_input="0000FF", include_hash=True):
        
        if color_mode == "rgb_sliders":
            r, g, b = red, green, blue
        elif color_mode == "preset_colors":
            preset_map = {
                "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
                "cyan": (0, 255, 255), "magenta": (255, 0, 255), "yellow": (255, 255, 0),
                "orange": (255, 165, 0), "purple": (128, 0, 128),
                "dark_red": (128, 0, 0), "dark_green": (0, 128, 0), "dark_blue": (0, 0, 128),
                "light_gray": (192, 192, 192), "dark_gray": (64, 64, 64),
                "black": (0, 0, 0), "white": (255, 255, 255)
            }
            r, g, b = preset_map.get(preset_color, (0, 0, 255))
        else:  # hex_input
            try:
                # Remove # if present
                hex_clean = hex_input.replace("#", "")
                # Ensure 6 digits
                if len(hex_clean) == 3:
                    hex_clean = ''.join([c*2 for c in hex_clean])
                elif len(hex_clean) != 6:
                    hex_clean = "0000FF"  # Default blue
                
                r = int(hex_clean[0:2], 16)
                g = int(hex_clean[2:4], 16)
                b = int(hex_clean[4:6], 16)
            except ValueError:
                r, g, b = 0, 0, 255  # Default blue
        
        # Generate hex output
        hex_color = f"{r:02X}{g:02X}{b:02X}"
        if include_hash:
            hex_color = "#" + hex_color
        
        return (hex_color, r, g, b)


class DepthMapProcessor:
    """Process and modify depth maps"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("IMAGE",),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "clip_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "clip_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_depth_map",)
    FUNCTION = "process_depth_map"
    CATEGORY = "DeepStereo/Utility"

    def process_depth_map(self, depth_map, invert=False, contrast=1.0, brightness=0.0, 
                         blur_radius=0, clip_min=0.0, clip_max=1.0):
        
        # Convert ComfyUI tensor to PIL Image
        if len(depth_map.shape) == 4:
            depth_map = depth_map[0]
        
        img_np = depth_map.cpu().numpy()
        
        # Apply brightness adjustment
        if brightness != 0.0:
            img_np = np.clip(img_np + brightness, 0.0, 1.0)
        
        # Apply contrast adjustment
        if contrast != 1.0:
            img_np = np.clip((img_np - 0.5) * contrast + 0.5, 0.0, 1.0)
        
        # Apply clipping
        if clip_min > 0.0 or clip_max < 1.0:
            img_np = np.clip(img_np, clip_min, clip_max)
            # Renormalize to 0-1 range
            if clip_max > clip_min:
                img_np = (img_np - clip_min) / (clip_max - clip_min)
        
        # Apply inversion
        if invert:
            img_np = 1.0 - img_np
        
        # Convert to PIL for blur
        img_pil_np = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_pil_np)
        
        # Apply blur
        if blur_radius > 0:
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Convert back to ComfyUI tensor
        result_np = np.array(img_pil).astype(np.float32) / 255.0
        
        # Ensure we maintain the original shape (might be grayscale)
        if len(result_np.shape) == 2:
            # Convert grayscale to RGB
            result_np = np.stack([result_np, result_np, result_np], axis=-1)
        
        result_tensor = torch.from_numpy(result_np)
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)


class RandomNoiseGenerator:
    """Generate random noise textures with seamless tiling for stereograms"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1,
                                     "tooltip": "Width of final output image"}),
                "output_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1,
                                      "tooltip": "Height of final output image"}),
                "pattern_width": ("INT", {"default": 100, "min": 30, "max": 200, "step": 1,
                                      "tooltip": "Width of the actual pattern (should match max_separation)"}),
                "noise_type": ([
                    "rgb", "grayscale", "colored_dots", "perlin",
                    "voronoi", "kaleidoscope", "waves", "cellular",
                    "fractal", "spiral", "interference", "crystalline"
                ], {"default": "rgb"}),
            },
            "optional": {
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, 
                                   "tooltip": "0 for random seed"}),
                "dot_density": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05, 
                                     "tooltip": "For colored_dots type"}),
                "base_color": ("STRING", {"default": "#808080", 
                                     "tooltip": "Background color for colored_dots"}),
                "noise_scale": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01, 
                                     "tooltip": "Scale of the noise pattern"}),
                "frequency": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 50.0, "step": 0.1,
                                   "tooltip": "Frequency of the pattern"}),
                "octaves": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1,
                                "tooltip": "Number of noise layers (for fractal patterns)"}),
                "color_variation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                         "tooltip": "Amount of color variation"}),
                "symmetry": ("INT", {"default": 6, "min": 2, "max": 16, "step": 1,
                                "tooltip": "Symmetry points for kaleidoscope"}),
                "seamless_tiling": ("BOOLEAN", {"default": True,
                                           "tooltip": "Create seamlessly tiling pattern"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("noise_texture",)
    FUNCTION = "generate_noise"
    CATEGORY = "DeepStereo/Utility"

    def generate_noise(self, output_width, output_height, pattern_width, noise_type, 
                      random_seed=0, dot_density=0.5, base_color="#808080", 
                      noise_scale=0.1, frequency=5.0, octaves=4,
                      color_variation=1.0, symmetry=6, seamless_tiling=True):
        
        def apply_color_variation(pattern, c, phase_shift=0):
            """Apply color variation to a pattern"""
            if len(pattern.shape) == 2:
                shifted = np.roll(pattern, int(phase_shift * pattern_width/8), axis=1)
                return shifted
            return pattern[:,:,c]

        def create_seamless_tile(pattern, blend_width=None):
            """Create a seamless tileable version of the pattern"""
            if not blend_width:
                blend_width = pattern_width // 4
            
            h, w = pattern.shape[:2]
            result = pattern.copy()
            
            # Horizontal blending
            for x in range(blend_width):
                alpha = x / blend_width
                result[:, x] = pattern[:, x] * alpha + pattern[:, -blend_width + x] * (1 - alpha)
                result[:, -blend_width + x] = pattern[:, -blend_width + x] * (1 - alpha) + pattern[:, x] * alpha
            
            # Vertical blending
            for y in range(blend_width):
                alpha = y / blend_width
                result[y, :] = pattern[y, :] * alpha + pattern[-blend_width + y, :] * (1 - alpha)
                result[-blend_width + y, :] = pattern[-blend_width + y, :] * (1 - alpha) + pattern[y, :] * alpha
            
            return result

        def tile_pattern(pattern):
            """Tile pattern across the output dimensions"""
            h, w = pattern.shape[:2]
            tiles_y = (output_height + h - 1) // h
            tiles_x = (output_width + w - 1) // w
            
            # Create tiled image
            tiled = np.tile(pattern, (tiles_y, tiles_x, 1) if len(pattern.shape) == 3 else (tiles_y, tiles_x))
            
            # Crop to output size
            return tiled[:output_height, :output_width]

        # Set random seed if specified
        if random_seed > 0:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Initialize coordinates for pattern generation
        y_coords, x_coords = np.mgrid[0:pattern_width, 0:pattern_width]
        
        # Generate base pattern
        if noise_type == "rgb":
            pattern = np.random.randint(0, 256, (pattern_width, pattern_width, 3), dtype=np.uint8)
            
        elif noise_type == "grayscale":
            pattern = np.random.randint(0, 256, (pattern_width, pattern_width), dtype=np.uint8)
            pattern = np.stack([pattern] * 3, axis=-1)
            
        elif noise_type == "colored_dots":
            try:
                if not base_color.startswith('#'):
                    base_color = '#' + base_color
                from PIL import ImageColor
                bg_rgb = ImageColor.getrgb(base_color)
            except:
                bg_rgb = (128, 128, 128)
            
            pattern = np.full((pattern_width, pattern_width, 3), bg_rgb, dtype=np.uint8)
            num_dots = int(pattern_width * pattern_width * dot_density)
            
            for _ in range(num_dots):
                x = random.randint(0, pattern_width - 1)
                y = random.randint(0, pattern_width - 1)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                pattern[y, x] = color
                
        elif noise_type in ["perlin", "fractal"]:
            pattern = np.zeros((pattern_width, pattern_width, 3), dtype=np.uint8)
            
            for octave in range(octaves):
                freq = (2 ** octave) * noise_scale * frequency
                amplitude = 1.0 / (2 ** octave)
                
                if noise_type == "fractal":
                    # More complex fractal pattern
                    angle = np.arctan2(y_coords - pattern_width/2, x_coords - pattern_width/2)
                    dist = np.sqrt((x_coords - pattern_width/2)**2 + (y_coords - pattern_width/2)**2)
                    noise_octave = np.sin(dist * freq * 0.1 + angle * octaves)
                    noise_octave *= np.cos(x_coords * freq * 0.05) * np.sin(y_coords * freq * 0.05)
                else:
                    # Enhanced Perlin-like noise
                    noise_octave = np.sin(x_coords * freq * 0.1) * np.cos(y_coords * freq * 0.1)
                    noise_octave += np.sin((x_coords + y_coords) * freq * 0.07)
                
                noise_octave = (noise_octave * amplitude * 255).astype(np.uint8)
                
                for c in range(3):
                    phase = c * 2 * np.pi / 3 * color_variation
                    colored_noise = apply_color_variation(noise_octave, c, phase)
                    pattern[:, :, c] = np.clip(pattern[:, :, c] + colored_noise, 0, 255)
                    
        elif noise_type == "kaleidoscope":
            center_y, center_x = pattern_width // 2, pattern_width // 2
            r = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            theta = np.arctan2(y_coords - center_y, x_coords - center_x)
            
            angle_segment = 2 * np.pi / symmetry
            theta_mod = (theta % angle_segment) / angle_segment * 2 * np.pi
            
            base_pattern = np.sin(r * noise_scale + theta_mod * frequency)
            base_pattern = (base_pattern * 128 + 128).astype(np.uint8)
            
            pattern = np.zeros((pattern_width, pattern_width, 3), dtype=np.uint8)
            for c in range(3):
                pattern[:, :, c] = apply_color_variation(base_pattern, c, c * color_variation)
                
        elif noise_type == "waves":
            pattern = np.zeros((pattern_width, pattern_width, 3), dtype=np.uint8)
            for wave in range(octaves):
                angle = wave * np.pi / octaves
                wave_x = np.cos(angle) * x_coords + np.sin(angle) * y_coords
                wave_pattern = np.sin(wave_x * frequency * noise_scale)
                wave_pattern = (wave_pattern * 128 + 128).astype(np.uint8)
                
                for c in range(3):
                    phase = (c * 2 * np.pi / 3 + wave) * color_variation
                    pattern[:, :, c] = np.clip(
                        pattern[:, :, c] + apply_color_variation(wave_pattern, c, phase),
                        0, 255
                    )
                    
        elif noise_type in ["voronoi", "cellular"]:
            # Generate random points for both Voronoi and cellular patterns
            num_points = int(20 * noise_scale)
            points = np.random.rand(num_points, 2)
            point_colors = np.random.randint(0, 256, (num_points, 3))
            x_norm, y_norm = x_coords / pattern_width, y_coords / pattern_width
            
            # Calculate distances to all points
            distances = np.zeros((len(points), pattern_width, pattern_width))
            for i, (px, py) in enumerate(points):
                distances[i] = np.sqrt((x_norm - px)**2 + (y_norm - py)**2)
            
            if noise_type == "voronoi":
                # For Voronoi, use nearest point's color
                nearest_point = distances.argmin(axis=0)
                pattern = np.zeros((pattern_width, pattern_width, 3), dtype=np.uint8)
                for c in range(3):
                    pattern[:, :, c] = point_colors[nearest_point, c]
            else:  # cellular
                # For cellular, use difference between nearest and second nearest
                distances.sort(axis=0)
                base_pattern = (distances[1] - distances[0]) * 255
                
                pattern = np.zeros((pattern_width, pattern_width, 3), dtype=np.uint8)
                for c in range(3):
                    pattern[:, :, c] = apply_color_variation(base_pattern, c, c * color_variation)
                
        elif noise_type == "spiral":
            center_y, center_x = pattern_width // 2, pattern_width // 2
            r = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            theta = np.arctan2(y_coords - center_y, x_coords - center_x)
            
            base_pattern = np.sin(r * noise_scale + theta * frequency)
            base_pattern = (base_pattern * 128 + 128).astype(np.uint8)
            
            pattern = np.zeros((pattern_width, pattern_width, 3), dtype=np.uint8)
            for c in range(3):
                pattern[:, :, c] = apply_color_variation(base_pattern, c, c * color_variation)
                
        elif noise_type == "interference":
            center1_x, center1_y = pattern_width * 0.3, pattern_width * 0.3
            center2_x, center2_y = pattern_width * 0.7, pattern_width * 0.7
            
            r1 = np.sqrt((x_coords - center1_x)**2 + (y_coords - center1_y)**2)
            r2 = np.sqrt((x_coords - center2_x)**2 + (y_coords - center2_y)**2)
            
            base_pattern = np.sin(r1 * frequency * noise_scale) * np.sin(r2 * frequency * noise_scale)
            base_pattern = (base_pattern * 128 + 128).astype(np.uint8)
            
            pattern = np.zeros((pattern_width, pattern_width, 3), dtype=np.uint8)
            for c in range(3):
                pattern[:, :, c] = apply_color_variation(base_pattern, c, c * color_variation)
                
        elif noise_type == "crystalline":
            pattern = np.zeros((pattern_width, pattern_width, 3), dtype=np.uint8)
            
            for crystal in range(octaves):
                angle = crystal * np.pi / octaves
                transformed_x = np.cos(angle) * x_coords + np.sin(angle) * y_coords
                transformed_y = -np.sin(angle) * x_coords + np.cos(angle) * y_coords
                
                base_pattern = np.abs(np.sin(transformed_x * frequency * noise_scale) * 
                                    np.cos(transformed_y * frequency * noise_scale))
                base_pattern = (base_pattern * 255).astype(np.uint8)
                
                for c in range(3):
                    phase = (c * 2 * np.pi / 3 + crystal) * color_variation
                    pattern[:, :, c] = np.clip(
                        pattern[:, :, c] + apply_color_variation(base_pattern, c, phase),
                        0, 255
                    )
        
        # Apply seamless tiling if requested
        if seamless_tiling:
            pattern = create_seamless_tile(pattern)
        
        # Tile the pattern to final size
        noise_data = tile_pattern(pattern)
        
        # Convert to ComfyUI tensor
        result_tensor = torch.from_numpy(noise_data).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)
        
        if noise_type == "rgb":
            # Generate base pattern
            pattern = np.random.randint(0, 256, (current_height, current_width, 3), dtype=np.uint8)
            
            if seamless_tiling:
                pattern = create_seamless_tile(pattern)
            
            # Tile the pattern
            noise_data = tile_pattern(pattern)
            
        elif noise_type == "grayscale":
            # Random grayscale noise
            gray_noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            noise_data = np.stack([gray_noise, gray_noise, gray_noise], axis=-1)
            
        elif noise_type == "colored_dots":
            # Colored dots on background
            try:
                if not base_color.startswith('#'):
                    base_color = '#' + base_color
                from PIL import ImageColor
                bg_rgb = ImageColor.getrgb(base_color)
            except:
                bg_rgb = (128, 128, 128)
            
            noise_data = np.full((height, width, 3), bg_rgb, dtype=np.uint8)
            
            num_dots = int(width * height * dot_density)
            for _ in range(num_dots):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                noise_data[y, x] = color
                
        elif noise_type in ["perlin", "fractal"]:
            noise_data = np.zeros((height, width, 3), dtype=np.uint8)
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            for octave in range(octaves):
                freq = (2 ** octave) * noise_scale * frequency
                amplitude = 1.0 / (2 ** octave)
                
                if noise_type == "fractal":
                    # More complex fractal pattern
                    angle = np.arctan2(y_coords - height/2, x_coords - width/2)
                    dist = np.sqrt((x_coords - width/2)**2 + (y_coords - height/2)**2)
                    noise_octave = np.sin(dist * freq * 0.1 + angle * octaves)
                    noise_octave *= np.cos(x_coords * freq * 0.05) * np.sin(y_coords * freq * 0.05)
                else:
                    # Enhanced Perlin-like noise
                    noise_octave = np.sin(x_coords * freq * 0.1) * np.cos(y_coords * freq * 0.1)
                    noise_octave += np.sin((x_coords + y_coords) * freq * 0.07)
                
                noise_octave = (noise_octave * amplitude * 255).astype(np.uint8)
                
                for c in range(3):
                    phase = c * 2 * np.pi / 3 * color_variation
                    colored_noise = np.roll(noise_octave, int(phase * width/8), axis=1)
                    noise_data[:, :, c] = np.clip(noise_data[:, :, c] + colored_noise, 0, 255)
        
        elif noise_type == "voronoi":
            # Generate Voronoi-like pattern
            points = np.random.rand(int(20 * noise_scale), 2)
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            y_norm, x_norm = y_coords / height, x_coords / width
            
            noise_data = np.zeros((height, width, 3), dtype=np.uint8)
            for i, (px, py) in enumerate(points):
                dist = np.sqrt((x_norm - px)**2 + (y_norm - py)**2)
                color = np.random.randint(0, 255, 3)
                for c in range(3):
                    mask = dist == dist.min(axis=None)
                    noise_data[:, :, c] = np.where(mask, color[c], noise_data[:, :, c])
        
        elif noise_type == "kaleidoscope":
            # Generate kaleidoscope pattern
            center_y, center_x = height // 2, width // 2
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            # Convert to polar coordinates
            r = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            theta = np.arctan2(y_coords - center_y, x_coords - center_x)
            
            # Create kaleidoscope effect
            angle_segment = 2 * np.pi / symmetry
            theta_mod = (theta % angle_segment) / angle_segment * 2 * np.pi
            
            # Generate base pattern
            pattern = np.sin(r * noise_scale + theta_mod * frequency)
            pattern = (pattern * 128 + 128).astype(np.uint8)
            
            noise_data = np.zeros((height, width, 3), dtype=np.uint8)
            for c in range(3):
                phase = c * 2 * np.pi / 3 * color_variation
                noise_data[:, :, c] = np.roll(pattern, int(phase * width/8), axis=1)
        
        elif noise_type == "waves":
            # Generate wave interference pattern
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            noise_data = np.zeros((height, width, 3), dtype=np.uint8)
            
            for wave in range(octaves):
                angle = wave * np.pi / octaves
                wave_x = np.cos(angle) * x_coords + np.sin(angle) * y_coords
                wave_pattern = np.sin(wave_x * frequency * noise_scale)
                wave_pattern = (wave_pattern * 128 + 128).astype(np.uint8)
                
                for c in range(3):
                    phase = c * 2 * np.pi / 3 * color_variation
                    noise_data[:, :, c] = np.clip(
                        noise_data[:, :, c] + np.roll(wave_pattern, int(phase * width/8), axis=1),
                        0, 255
                    )
        
        elif noise_type == "cellular":
            # Generate cellular/organic pattern
            noise_data = np.zeros((height, width, 3), dtype=np.uint8)
            points = np.random.rand(int(30 * noise_scale), 2)
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            y_norm, x_norm = y_coords / height, x_coords / width
            
            distances = np.zeros((len(points), height, width))
            for i, (px, py) in enumerate(points):
                distances[i] = np.sqrt((x_norm - px)**2 + (y_norm - py)**2)
            
            # Sort distances for each pixel
            distances.sort(axis=0)
            pattern = (distances[1] - distances[0]) * 255  # Difference between closest and second closest
            
            for c in range(3):
                phase = c * 2 * np.pi / 3 * color_variation
                noise_data[:, :, c] = np.roll(pattern, int(phase * width/8), axis=1).astype(np.uint8)
        
        elif noise_type == "spiral":
            # Generate spiral pattern
            center_y, center_x = height // 2, width // 2
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            r = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            theta = np.arctan2(y_coords - center_y, x_coords - center_x)
            
            spiral = np.sin(r * noise_scale + theta * frequency)
            spiral = (spiral * 128 + 128).astype(np.uint8)
            
            noise_data = np.zeros((height, width, 3), dtype=np.uint8)
            for c in range(3):
                phase = c * 2 * np.pi / 3 * color_variation
                noise_data[:, :, c] = np.roll(spiral, int(phase * width/8), axis=1)
        
        elif noise_type == "interference":
            # Generate interference pattern
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            center1_x, center1_y = width * 0.3, height * 0.3
            center2_x, center2_y = width * 0.7, height * 0.7
            
            r1 = np.sqrt((x_coords - center1_x)**2 + (y_coords - center1_y)**2)
            r2 = np.sqrt((x_coords - center2_x)**2 + (y_coords - center2_y)**2)
            
            pattern = np.sin(r1 * frequency * noise_scale) * np.sin(r2 * frequency * noise_scale)
            pattern = (pattern * 128 + 128).astype(np.uint8)
            
            noise_data = np.zeros((height, width, 3), dtype=np.uint8)
            for c in range(3):
                phase = c * 2 * np.pi / 3 * color_variation
                noise_data[:, :, c] = np.roll(pattern, int(phase * width/8), axis=1)
        
        elif noise_type == "crystalline":
            # Generate crystalline pattern
            noise_data = np.zeros((height, width, 3), dtype=np.uint8)
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            for crystal in range(octaves):
                angle = crystal * np.pi / octaves
                transformed_x = np.cos(angle) * x_coords + np.sin(angle) * y_coords
                transformed_y = -np.sin(angle) * x_coords + np.cos(angle) * y_coords
                
                pattern = np.abs(np.sin(transformed_x * frequency * noise_scale) * 
                               np.cos(transformed_y * frequency * noise_scale))
                pattern = (pattern * 255).astype(np.uint8)
                
                for c in range(3):
                    phase = (c * 2 * np.pi / 3 + crystal) * color_variation
                    noise_data[:, :, c] = np.clip(
                        noise_data[:, :, c] + np.roll(pattern, int(phase * width/8), axis=1),
                        0, 255
                    )
        
        else:
            # Fallback to RGB noise
            noise_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Convert to ComfyUI tensor
        result_tensor = torch.from_numpy(noise_data).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)
