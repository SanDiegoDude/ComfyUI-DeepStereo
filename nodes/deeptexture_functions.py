import argparse
import os
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops, ImageColor
import numpy as np
import random
import math
from tqdm import tqdm
from datetime import datetime

# Copy all the functions from your original deeptexture.py here
# Remove the main() function and argparse stuff, keep just the core functions

# --- Helper Functions ---
def parse_color(color_str, default_color=(0, 0, 0)):
    try:
        if ',' in color_str: return tuple(map(int, color_str.split(',')))
    except ValueError: pass
    try: return ImageColor.getrgb(color_str)
    except ValueError: return default_color

def get_pixel_value_safe(pil_image, x, y):
    x = max(0, min(x, pil_image.width - 1)); y = max(0, min(y, pil_image.height - 1))
    return pil_image.getpixel((x, y))

def resize_to_megapixels(image_pil, target_mp, verbose=True):
    if target_mp <= 0: return image_pil
    w, h = image_pil.size; current_mp = (w * h) / 1_000_000.0
    if current_mp <= target_mp:
        if verbose: print(f"Image is already within target megapixels ({current_mp:.2f}MP <= {target_mp:.2f}MP). No resize for texture base.")
        return image_pil
    scale_factor = math.sqrt(target_mp / current_mp)
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    if new_w == 0 or new_h == 0:
        if verbose: print(f"Warning: Calculated new dimensions for texture base are too small ({new_w}x{new_h}). Keeping original size."); return image_pil
    if verbose: print(f"Resizing image for texture base from {w}x{h} ({current_mp:.2f}MP) to {new_w}x{new_h} (~{target_mp:.2f}MP)...")
    return image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

def rgb_to_hsv(r, g, b):
    r,g,b=r/255.0,g/255.0,b/255.0; mx=max(r,g,b); mn=min(r,g,b); df=mx-mn
    if mx==mn: h=0
    elif mx==r: h=(60*((g-b)/df)+360)%360
    elif mx==g: h=(60*((b-r)/df)+120)%360
    elif mx==b: h=(60*((r-g)/df)+240)%360
    if mx==0: s=0
    else: s=df/mx
    v=mx; return h,s,v

def hsv_to_rgb(h, s, v):
    i=math.floor(h/60)%6; f=(h/60)-math.floor(h/60); p=v*(1-s); q=v*(1-f*s); t=v*(1-(1-f)*s)
    if i==0: r,g,b=v,t,p
    elif i==1: r,g,b=q,v,p
    elif i==2: r,g,b=p,v,t
    elif i==3: r,g,b=p,q,v
    elif i==4: r,g,b=t,p,v
    elif i==5: r,g,b=v,p,q
    return int(r*255),int(g*255),int(b*255)

VIBRANT_PALETTE = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,128,0),(128,0,255)]

# --- Texture Generation Methods ---
def apply_method1_color_dots(content_image_pil, density=0.7, dot_size=1, bg_color=(0,0,0), color_mode="content_pixel", hue_shift_degrees=60):
    width,height=content_image_pil.size; output_image=Image.new("RGB",(width,height),bg_color); draw=ImageDraw.Draw(output_image)
    num_dots=int(width*height*density); content_rgb=content_image_pil.convert("RGB")
    for _ in range(num_dots):  # Remove tqdm for ComfyUI
        x,y=random.randint(0,width-1),random.randint(0,height-1); dot_color=(0,0,0)
        r_orig,g_orig,b_orig=get_pixel_value_safe(content_rgb,x,y)
        if color_mode=="content_pixel": dot_color=(r_orig,g_orig,b_orig)
        elif color_mode=="random_rgb": dot_color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
        elif color_mode=="random_from_palette": dot_color=random.choice(VIBRANT_PALETTE)
        elif color_mode=="transformed_hue":
            h,s,v=rgb_to_hsv(r_orig,g_orig,b_orig); h=(h+random.uniform(-hue_shift_degrees,hue_shift_degrees))%360
            dot_color=hsv_to_rgb(h,s,v)
        elif color_mode=="transformed_invert": dot_color=(255-r_orig,255-g_orig,255-b_orig)
        if dot_size==1: draw.point((x,y),fill=dot_color)
        else: x0,y0=x-dot_size//2,y-dot_size//2; x1,y1=x0+dot_size-1,y0+dot_size-1; draw.rectangle([x0,y0,x1,y1],fill=dot_color)
    return output_image

def apply_method2_density_size(content_image_pil, mode="density", element_color=(0,0,0), bg_color=(255,255,255), base_size=2, max_size=10, invert_influence=False, density_factor=1.0):
    width,height=content_image_pil.size; output_image=Image.new("RGB",(width,height),bg_color); draw=ImageDraw.Draw(output_image)
    content_gray=content_image_pil.convert("L")
    if mode=="density":
        step=max(1,base_size)
        for r in range(0,height,step):  # Remove tqdm for ComfyUI
            for c in range(0,width,step):
                brightness=get_pixel_value_safe(content_gray,c,r)/255.0; influence=brightness if invert_influence else 1.0-brightness
                if random.random()<(influence*density_factor): draw.rectangle([c,r,c+base_size-1,r+base_size-1],fill=element_color)
    elif mode=="size":
        num_elements=int((width*height*0.1)*density_factor)
        for _ in range(num_elements):  # Remove tqdm for ComfyUI
            x,y=random.randint(0,width-1),random.randint(0,height-1); brightness=get_pixel_value_safe(content_gray,x,y)/255.0
            influence=brightness if invert_influence else 1.0-brightness
            current_size=max(1,int(base_size+(max_size-base_size)*influence))
            ex1,ey1=x-current_size//2,y-current_size//2; ex2,ey2=ex1+current_size-1,ey1+current_size-1
            draw.rectangle([ex1,ey1,ex2,ey2],fill=element_color)
    return output_image

def apply_method3_voronoi(content_image_pil, num_points=100, metric="F1", color_source="distance", point_color=(255,0,0)):
    width,height=content_image_pil.size; output_image=Image.new("RGB",(width,height)); output_pixels=output_image.load()
    points=[]; content_rgb=content_image_pil.convert("RGB")
    for _ in range(num_points):
        px,py=random.randint(0,width-1),random.randint(0,height-1)
        p_content_color=get_pixel_value_safe(content_rgb,px,py); points.append((px,py,p_content_color))
    if not points: return content_image_pil.copy()
    max_dist_val=math.sqrt(width**2+height**2)/3
    for r in range(height):  # Remove tqdm for ComfyUI
        for c in range(width):
            distances_sq_data=[((c-px)**2+(r-py)**2,pcc) for px,py,pcc in points]; distances_sq_data.sort(key=lambda item:item[0])
            final_color=(0,0,0)
            if color_source=="distance":
                dist_val=0
                if metric=="F1" and len(distances_sq_data)>0: dist_val=math.sqrt(distances_sq_data[0][0])
                elif metric=="F2" and len(distances_sq_data)>1: dist_val=math.sqrt(distances_sq_data[1][0])
                elif metric=="F2-F1" and len(distances_sq_data)>1: dist_val=abs(math.sqrt(distances_sq_data[1][0])-math.sqrt(distances_sq_data[0][0]))
                norm_dist=min(dist_val/max_dist_val,1.0) if max_dist_val>0 else 0; gray_val=int(norm_dist*255)
                final_color=(gray_val,gray_val,gray_val)
            elif color_source=="content_point_color":
                idx=0;
                if metric=="F2" and len(distances_sq_data)>1: idx=1
                final_color=distances_sq_data[idx][1] if len(distances_sq_data)>idx else (0,0,0)
            elif color_source=="voronoi_cell_content_color":
                final_color=distances_sq_data[0][1] if distances_sq_data else (0,0,0)
            output_pixels[c,r]=final_color
    return output_image

def generate_glyph(glyph_style, size, glyph_element_color=(0,0,0), glyph_bg_color=(255,255,255)):
    glyph_img=Image.new("RGB",(size,size),glyph_bg_color); draw=ImageDraw.Draw(glyph_img)
    dot_density_factor=0.4
    if glyph_style=="random_dots":
        for _ in range(int(size*size*dot_density_factor)): draw.point((random.randint(0,size-1),random.randint(0,size-1)),fill=glyph_element_color)
    elif glyph_style=="lines":
        for i in range(0,size,max(1,size//3)): draw.line([(i,0),(i,size-1)],fill=glyph_element_color,width=max(1,size//8))
    elif glyph_style=="circles":
        padding=max(1,size//8); fill_color=glyph_bg_color if random.random()>0.6 else glyph_element_color
        draw.ellipse([(padding,padding,size-1-padding,size-1-padding)],outline=glyph_element_color,fill=fill_color,width=max(1,size//8))
    elif glyph_style=="solid": draw.rectangle([(0,0),(size-1,size-1)],fill=glyph_element_color)
    return glyph_img

def apply_method4_glyph_dither(content_image_pil, num_colors=8, glyph_size=8, glyph_style="random_dots", use_quantized_color_for_glyph_element=True):
    width,height=content_image_pil.size; output_image=Image.new("RGB",(width,height),(128,128,128))
    try:
        quantized_content=content_image_pil.convert("RGB").quantize(colors=num_colors,method=Image.Quantize.MAXCOVERAGE)
        quantized_content_rgb=quantized_content.convert("RGB")
    except Exception as e:
        print(f"Warning: Quantization failed. Using original. Error: {e}"); quantized_content_rgb=content_image_pil.convert("RGB")
    for r_block in range(0,height,glyph_size):  # Remove tqdm for ComfyUI
        for c_block in range(0,width,glyph_size):
            block_center_x,block_center_y=min(c_block+glyph_size//2,width-1),min(r_block+glyph_size//2,height-1)
            glyph_main_color=get_pixel_value_safe(quantized_content_rgb,block_center_x,block_center_y)
            if not use_quantized_color_for_glyph_element: glyph_main_color=(0,0,0) if sum(glyph_main_color)>384 else (255,255,255)
            glyph_internal_bg_color=(0,0,0) if sum(glyph_main_color)>384 else (255,255,255)
            if glyph_style=="solid": glyph_internal_bg_color=glyph_main_color
            glyph=generate_glyph(glyph_style,glyph_size,glyph_element_color=glyph_main_color,glyph_bg_color=glyph_internal_bg_color)
            output_image.paste(glyph,(c_block,r_block))
    return output_image

# --- Image Blending Functions ---
def blend_images(base_image, blend_image, mode="average", opacity=1.0):
    if base_image.size!=blend_image.size or base_image.mode!=blend_image.mode:
        blend_image=blend_image.convert(base_image.mode)
        if base_image.size!=blend_image.size: raise ValueError("Images must be the same size to blend.")
    base_rgb=base_image.convert("RGB"); blend_rgb=blend_image.convert("RGB"); result_rgb=None
    if mode=="average": result_rgb=Image.blend(base_rgb,blend_rgb,0.5)
    elif mode=="lighten": result_rgb=ImageChops.lighter(base_rgb,blend_rgb)
    elif mode=="darken": result_rgb=ImageChops.darker(base_rgb,blend_rgb)
    elif mode=="multiply": result_rgb=ImageChops.multiply(base_rgb,blend_rgb)
    elif mode=="screen": result_rgb=ImageChops.screen(base_rgb,blend_rgb)
    elif mode=="add": result_rgb=ImageChops.add(base_rgb,blend_rgb)
    elif mode=="difference": result_rgb=ImageChops.difference(base_rgb,blend_rgb)
    elif mode=="overlay":
        base_arr=np.array(base_rgb,dtype=float)/255.0; blend_arr=np.array(blend_rgb,dtype=float)/255.0
        overlay_arr=np.zeros_like(base_arr); low_mask=base_arr<=0.5; high_mask=~low_mask
        overlay_arr[low_mask]=2*base_arr[low_mask]*blend_arr[low_mask]
        overlay_arr[high_mask]=1-2*(1-base_arr[high_mask])*(1-blend_arr[high_mask])
        result_rgb=Image.fromarray((np.clip(overlay_arr,0,1)*255).astype(np.uint8),"RGB")
    else: result_rgb=base_rgb
    if opacity<1.0 and result_rgb!=base_rgb:
        final_blend=Image.blend(base_image.convert("RGB"),result_rgb,opacity)
        return final_blend
    return result_rgb

def generate_texture_from_config(base_image_pil, texture_generation_args, verbose=True):
    """
    Core texture generation logic, callable for import.
    Takes a PIL image and an args-like object containing texture parameters.
    Returns a processed PIL image.
    """
    if verbose: print("--- Starting On-the-fly Texture Generation ---")

    image_for_texture_gen = base_image_pil.copy()
    if hasattr(texture_generation_args, 'tex_max_megapixels') and texture_generation_args.tex_max_megapixels > 0:
        image_for_texture_gen = resize_to_megapixels(image_for_texture_gen, texture_generation_args.tex_max_megapixels, verbose=verbose)
        if verbose: print(f"Base image for texture resized to: {image_for_texture_gen.size}.")
    
    methods_to_apply_info = []
    if getattr(texture_generation_args, 'tex_method1_color_dots', False):
        methods_to_apply_info.append({
            'name_suffix': "_m1", 'description': "Texture Method 1: Color Dots",
            # Fixed: Don't call parse_color on tex_m1_bg_color since it's already an RGB tuple
            'func': lambda img_in: apply_method1_color_dots(
                img_in, 
                texture_generation_args.tex_m1_density, 
                texture_generation_args.tex_m1_dot_size, 
                texture_generation_args.tex_m1_bg_color,  # Already an RGB tuple, no parse_color needed
                texture_generation_args.tex_m1_color_mode, 
                texture_generation_args.tex_m1_hue_shift_degrees
            )
        })
    if getattr(texture_generation_args, 'tex_method2_density_size', False):
        methods_to_apply_info.append({
            'name_suffix': "_m2", 'description': f"Texture Method 2: {texture_generation_args.tex_m2_mode.capitalize()}",
            'func': lambda img_in: apply_method2_density_size(
                img_in, 
                texture_generation_args.tex_m2_mode, 
                parse_color(texture_generation_args.tex_m2_element_color), 
                parse_color(texture_generation_args.tex_m2_bg_color), 
                texture_generation_args.tex_m2_base_size, 
                texture_generation_args.tex_m2_max_size, 
                texture_generation_args.tex_m2_invert_influence, 
                texture_generation_args.tex_m2_density_factor
            )
        })
    if getattr(texture_generation_args, 'tex_method3_voronoi', False):
        methods_to_apply_info.append({
            'name_suffix': "_m3", 'description': "Texture Method 3: Voronoi",
            'func': lambda img_in: apply_method3_voronoi(
                img_in, 
                texture_generation_args.tex_m3_num_points, 
                texture_generation_args.tex_m3_metric, 
                texture_generation_args.tex_m3_color_source
            )
        })
    if getattr(texture_generation_args, 'tex_method4_glyph_dither', False):
        methods_to_apply_info.append({
            'name_suffix': "_m4", 'description': "Texture Method 4: Glyph Dither",
            'func': lambda img_in: apply_method4_glyph_dither(
                img_in, 
                texture_generation_args.tex_m4_num_colors, 
                texture_generation_args.tex_m4_glyph_size, 
                texture_generation_args.tex_m4_glyph_style, 
                texture_generation_args.tex_m4_use_quantized_color_for_glyph_element
            )
        })

    if not methods_to_apply_info:
        if verbose: print("No texture processing methods specified for on-the-fly generation. Using base image as texture.")
        return image_for_texture_gen

    final_texture_image = None
    combination_mode = getattr(texture_generation_args, 'tex_combination_mode', 'sequential')
    blend_type = getattr(texture_generation_args, 'tex_blend_type', 'overlay')
    blend_opacity = getattr(texture_generation_args, 'tex_blend_opacity', 1.0)

    if combination_mode == "sequential":
        if verbose: print("Texture Gen: Applying methods sequentially...")
        current_image_for_seq = image_for_texture_gen.copy()
        for method_info in methods_to_apply_info:
            if verbose: print(f"Texture Gen: Applying {method_info.get('description', method_info.get('name_suffix'))} sequentially...")
            current_image_for_seq = method_info['func'](current_image_for_seq)
        final_texture_image = current_image_for_seq

    elif combination_mode == "blend":
        if verbose: print(f"Texture Gen: Applying methods individually for blending (Mode: {blend_type}, Opacity: {blend_opacity})...")
        individual_method_outputs = []
        for method_info in methods_to_apply_info:
            if verbose: print(f"Texture Gen: Running {method_info.get('description', method_info.get('name_suffix'))} (for blending)...")
            method_output = method_info['func'](image_for_texture_gen.copy())
            individual_method_outputs.append(method_output)

        if not individual_method_outputs: 
            final_texture_image = image_for_texture_gen.copy()
        else:
            final_texture_image = individual_method_outputs[0]
            if len(individual_method_outputs) > 1:
                for i in range(1, len(individual_method_outputs)):
                    if verbose: print(f"Texture Gen: Blending current result with output of next method using '{blend_type}'...")
                    final_texture_image = blend_images(final_texture_image, individual_method_outputs[i], mode=blend_type, opacity=blend_opacity)
    
    if final_texture_image is None: 
        final_texture_image = image_for_texture_gen.copy()
    if verbose: print("--- On-the-fly Texture Generation Complete ---")
    return final_texture_image
