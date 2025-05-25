import torch
import numpy as np
from PIL import Image, ImageFilter
import random

class ImageResizer:
    """Resize images with various options"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "target_megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1, 
                                              "tooltip": "Target MP (0 = use width/height instead)"}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1, 
                                       "tooltip": "Target width (0 = auto from height or MP)"}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1, 
                                        "tooltip": "Target height (0 = auto from width or MP)"}),
                "maintain_aspect": ("BOOLEAN", {"default": True}),
                "resample_method": (["LANCZOS", "BICUBIC", "BILINEAR", "NEAREST"], {"default": "LANCZOS"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("resized_image", "width", "height")
    FUNCTION = "resize_image"
    CATEGORY = "DeepStereo/Utility"

    def resize_image(self, image, target_megapixels=0.0, target_width=0, target_height=0, 
                    maintain_aspect=True, resample_method="LANCZOS"):
        
        # Convert ComfyUI tensor to PIL Image
        if len(image.shape) == 4:
            image = image[0]
        
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        original_width, original_height = img_pil.size
        
        # Determine target dimensions
        if target_megapixels > 0:
            # Calculate size based on megapixels
            current_mp = (original_width * original_height) / 1_000_000.0
            if current_mp <= target_megapixels:
                new_width, new_height = original_width, original_height
            else:
                import math
                scale_factor = math.sqrt(target_megapixels / current_mp)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
        elif target_width > 0 or target_height > 0:
            # Calculate size based on width/height
            if maintain_aspect:
                if target_width > 0 and target_height > 0:
                    # Both specified, choose smaller scale to maintain aspect
                    scale_w = target_width / original_width
                    scale_h = target_height / original_height
                    scale = min(scale_w, scale_h)
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)
                elif target_width > 0:
                    # Only width specified
                    scale = target_width / original_width
                    new_width = target_width
                    new_height = int(original_height * scale)
                else:
                    # Only height specified
                    scale = target_height / original_height
                    new_width = int(original_width * scale)
                    new_height = target_height
            else:
                new_width = target_width if target_width > 0 else original_width
                new_height = target_height if target_height > 0 else original_height
        else:
            # No resize needed
            new_width, new_height = original_width, original_height
        
        # Apply resize if dimensions changed
        if (new_width, new_height) != (original_width, original_height):
            resample_map = {
                "LANCZOS": Image.Resampling.LANCZOS,
                "BICUBIC": Image.Resampling.BICUBIC,
                "BILINEAR": Image.Resampling.BILINEAR,
                "NEAREST": Image.Resampling.NEAREST,
            }
            resample = resample_map.get(resample_method, Image.Resampling.LANCZOS)
            img_pil = img_pil.resize((new_width, new_height), resample)
        
        # Convert back to ComfyUI tensor
        result_np = np.array(img_pil)
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor, new_width, new_height)


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
    """Generate random noise textures"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "noise_type": (["rgb", "grayscale", "colored_dots", "perlin"], {"default": "rgb"}),
            },
            "optional": {
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "tooltip": "0 for random seed"}),
                "dot_density": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05, 
                                        "tooltip": "For colored_dots type"}),
                "base_color": ("STRING", {"default": "#808080", "tooltip": "Background color for colored_dots"}),
                "noise_scale": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01, 
                                        "tooltip": "For perlin noise"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("noise_texture",)
    FUNCTION = "generate_noise"
    CATEGORY = "DeepStereo/Utility"

    def generate_noise(self, width, height, noise_type, random_seed=0, dot_density=0.5, 
                      base_color="#808080", noise_scale=0.1):
        
        # Set random seed if specified
        if random_seed > 0:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        if noise_type == "rgb":
            # Random RGB noise
            noise_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
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
                
        elif noise_type == "perlin":
            # Simple pseudo-perlin noise
            noise_data = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Generate multiple octaves of noise
            for octave in range(4):
                freq = (2 ** octave) * noise_scale
                amplitude = 1.0 / (2 ** octave)
                
                # Simple noise generation (not true Perlin, but effective)
                y_coords, x_coords = np.mgrid[0:height, 0:width]
                noise_octave = np.sin(x_coords * freq * 0.1) * np.cos(y_coords * freq * 0.1)
                noise_octave += np.random.random((height, width)) * 0.5
                noise_octave = (noise_octave * amplitude * 255).astype(np.uint8)
                
                for c in range(3):
                    noise_data[:, :, c] = np.clip(noise_data[:, :, c] + noise_octave, 0, 255)
        
        else:
            # Fallback to RGB noise
            noise_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        # Convert to ComfyUI tensor
        result_tensor = torch.from_numpy(noise_data).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)
        
        return (result_tensor,)


class BatchImageProcessor:
    """Process multiple images in a batch with the same settings"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "operation": (["resize", "rotate", "flip_horizontal", "flip_vertical"], {"default": "resize"}),
            },
            "optional": {
                "target_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "rotation_degrees": ("INT", {"default": 90, "min": 0, "max": 359, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_images",)
    FUNCTION = "process_batch"
    CATEGORY = "DeepStereo/Utility"

    def process_batch(self, images, operation, target_width=512, target_height=512, rotation_degrees=90):
        processed_images = []
        
        # Process each image in the batch
        for i in range(images.shape[0]):
            img_tensor = images[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, 'RGB')
            
            if operation == "resize":
                img_pil = img_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
            elif operation == "rotate":
                img_pil = img_pil.rotate(rotation_degrees, expand=True)
            elif operation == "flip_horizontal":
                img_pil = img_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif operation == "flip_vertical":
                img_pil = img_pil.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            
            # Convert back to tensor
            result_np = np.array(img_pil)
            result_tensor = torch.from_numpy(result_np).float() / 255.0
            processed_images.append(result_tensor)
        
        # Stack back into batch format
        batch_tensor = torch.stack(processed_images, dim=0)
        
        return (batch_tensor,)
