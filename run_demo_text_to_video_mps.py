
import os
import argparse
import datetime
import PIL.Image
import numpy as np
import torch
import gc
import regex as re
import html
import ftfy

from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel

# Helper functions copied from pipeline to allow standalone encoding
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

def get_t5_prompt_embeds(tokenizer, text_encoder, prompt, max_sequence_length=512, device=None, dtype=None):
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    mask = mask.to(device=device)

    # duplicate text embeddings for each generation per prompt
    # Since num_videos_per_prompt is usually 1 here:
    num_videos_per_prompt = 1
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, 1, seq_len, -1)

    return prompt_embeds, mask

def encode_prompt_standalone(tokenizer, text_encoder, prompt, negative_prompt=None, device=None, dtype=None):
    prompt_embeds, prompt_attention_mask = get_t5_prompt_embeds(
        tokenizer, text_encoder, prompt, max_sequence_length=512, device=device, dtype=dtype
    )

    if negative_prompt is not None:
        negative_prompt_embeds, negative_prompt_attention_mask = get_t5_prompt_embeds(
            tokenizer, text_encoder, negative_prompt, max_sequence_length=512, device=device, dtype=dtype
        )
    else:
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        
    return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask


def generate(args):
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # case setup
    prompt = "In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    checkpoint_dir = args.checkpoint_dir
    # Use float32 for stability on MPS (avoid datatype mismatch), relying on offloading to prevent OOM
    dtype = torch.float32

    print("--- Stage 0: Text Encoding ---")
    print("Loading Tokenizer and Text Encoder...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=dtype)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=dtype)
    
    print(f"Moving Text Encoder to {device}...")
    try:
        text_encoder.to(device)
    except Exception as e:
        print(f"Failed to move Text Encoder to {device}: {e}")
        # Could fallback to CPU

    print("Encoding prompt...")
    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
            encode_prompt_standalone(tokenizer, text_encoder, prompt, negative_prompt, device=device, dtype=dtype)
    
    print("Unloading Text Encoder...")
    del tokenizer
    del text_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.mps, 'empty_cache'): # Not really available but standardizing
        pass

    # Ensure embeddings are on device and correct type
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    prompt_attention_mask = prompt_attention_mask.to(device=device)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(device=device)


    print("--- Stage 1: Generation ---")
    print("Loading DiT, VAE, Scheduler...")
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=dtype)
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir, 
        subfolder="dit", 
        cp_split_hw=[1, 1], 
        torch_dtype=dtype,
        enable_flashattn2=False,
        enable_flashattn3=False,
        enable_xformers=False,
    )

    print(f"Moving DiT and VAE to {device}...")
    try:
        dit.to(device)
        vae.to(device)
    except Exception as e:
         print(f"Warning: Failed to move models to {device}. Error: {e}")

    # Initialize pipeline with None for text_encoder/tokenizer to save memory
    pipe = LongCatVideoPipeline(
        tokenizer = None,
        text_encoder = None,
        vae = vae,
        scheduler = scheduler,
        dit = dit,
    )
    pipe.device = device
    
    # Generate
    height = args.height
    width = args.width
    num_frames = args.num_frames
    num_inference_steps = args.num_inference_steps
    
    print(f"Generating T2V (Size: {width}x{height}, Frames: {num_frames}, Steps: {num_inference_steps})")

    output = pipe.generate_t2v(
        prompt="", # Dummy prompt to satisfy check_inputs
        negative_prompt=None,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=4.0,
        generator=torch.Generator(device=device).manual_seed(42),
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )[0]

    # Post-process
    output_tensor = torch.from_numpy(np.array(output))
    output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
    write_video("output_t2v_mps.mp4", output_tensor, fps=15, video_codec="libx264", options={"crf": f"{18}"})
    print("Saved output_t2v_mps.mp4")
    
    print("Done!")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="weights/LongCat-Video")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse_args()
    generate(args)
