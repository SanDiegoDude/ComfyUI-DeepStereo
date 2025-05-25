import torch
import numpy as np
from PIL import Image
import cv2
import os
import folder_paths
import logging

class MiDaSDepthEstimator:
    def __init__(self):
        self.model = None
        self.transform = None
        self.device = None
        self.current_model_type = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": (["MiDaS_small", "DPT_Large", "DPT_Hybrid"], {"default": "MiDaS_small"}),
                "invert_depth": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "target_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32, 
                                       "tooltip": "Target width for processing (0 = use original size, rounded to 32px)"}),
                "save_depth_map": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("depth_map", "original_image")
    FUNCTION = "estimate_depth"
    CATEGORY = "DeepStereo/Depth"

    def load_model(self, model_type):
        """Load MiDaS model, saving to ComfyUI's controlnet directory"""
        if self.current_model_type == model_type and self.model is not None:
            return  # Model already loaded
        
        # Define where to save/load models in ComfyUI structure
        controlnet_path = folder_paths.get_folder_paths("controlnet")[0]
        midas_cache_dir = os.path.join(controlnet_path, "midas_models")
        os.makedirs(midas_cache_dir, exist_ok=True)
        
        # Set torch hub cache to our directory
        torch.hub.set_dir(midas_cache_dir)
        
        print(f"Loading MiDaS model ({model_type}) to {midas_cache_dir}...")
        
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            elif model_type == "MiDaS_small":
                self.transform = midas_transforms.small_transform
            else:
                print(f"Warning: Unknown MiDaS model type '{model_type}'. Using small_transform.")
                self.transform = midas_transforms.small_transform
                
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model.to(self.device)
            self.model.eval()
            self.current_model_type = model_type
            
            print(f"MiDaS model loaded successfully on device: {self.device}")
            
        except Exception as e:
            print(f"Error loading MiDaS model: {e}")
            raise

    def estimate_depth(self, image, model_type, invert_depth, target_width=0, save_depth_map=False):
        # Load model if needed
        self.load_model(model_type)
        
        # Convert ComfyUI tensor to PIL Image
        # ComfyUI images are typically in format [batch, height, width, channels] with values 0-1
        if len(image.shape) == 4:
            image = image[0]  # Take first image if batch
        
        # Convert from tensor to numpy array with proper scaling
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, 'RGB')
        
        original_size = img_pil.size
        processing_image = img_pil.copy()
        
        # Handle target width resizing
        if target_width > 0:
            aspect_ratio = img_pil.height / img_pil.width
            target_width_rounded = (target_width // 32) * 32
            target_height_rounded = (int(target_width_rounded * aspect_ratio) // 32) * 32
            
            if target_width_rounded > 0 and target_height_rounded > 0:
                processing_image = img_pil.resize((target_width_rounded, target_height_rounded), Image.Resampling.LANCZOS)
                print(f"Resized for MiDaS processing: {original_size} -> {processing_image.size}")
        
        # Convert to OpenCV format for MiDaS
        img_cv = np.array(processing_image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Apply MiDaS transform and run inference
        input_batch = self.transform(img_cv).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), 
                size=processing_image.size[::-1],  # height, width
                mode="bicubic", 
                align_corners=False
            ).squeeze()
        
        # Process depth output
        depth_output = prediction.cpu().numpy()
        depth_min, depth_max = np.min(depth_output), np.max(depth_output)
        
        if depth_max > depth_min:
            depth_normalized = (depth_output - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_output)
        
        # MiDaS outputs inverse depth, so we flip it to get normal depth
        processed_depth_normalized = 1.0 - depth_normalized
        
        if invert_depth:
            processed_depth_normalized = 1.0 - processed_depth_normalized
        
        # Convert to 8-bit grayscale
        depth_map_visual = (processed_depth_normalized * 255).astype(np.uint8)
        depth_map_pil = Image.fromarray(depth_map_visual, 'L')
        
        # Resize back to original size if we resized for processing
        if depth_map_pil.size != original_size:
            depth_map_pil = depth_map_pil.resize(original_size, Image.Resampling.LANCZOS)
        
        # Convert depth map to RGB for ComfyUI compatibility
        depth_map_rgb = depth_map_pil.convert('RGB')
        
        # Save depth map if requested
        if save_depth_map:
            output_dir = folder_paths.get_output_directory()
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            depth_filename = f"depth_map_{model_type}_{timestamp}.png"
            depth_path = os.path.join(output_dir, depth_filename)
            depth_map_pil.save(depth_path)
            print(f"Depth map saved to: {depth_path}")
        
        # Convert back to ComfyUI tensor format
        depth_tensor = torch.from_numpy(np.array(depth_map_rgb)).float() / 255.0
        depth_tensor = depth_tensor.unsqueeze(0)  # Add batch dimension
        
        original_tensor = torch.from_numpy(np.array(img_pil)).float() / 255.0
        original_tensor = original_tensor.unsqueeze(0)  # Add batch dimension
        
        return (depth_tensor, original_tensor)
