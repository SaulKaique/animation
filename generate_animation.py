%%writefile generate_animation.py
import os
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from PIL import Image
import argparse
import imageio
import numpy as np

def load_image(image_path):
    return Image.open(image_path).convert("RGB").resize((512, 512))

def main(prompt, duration=10, image_path=None, output_path="video.mp4", fps=8):
    base_model = "runwayml/stable-diffusion-v1-5"
    motion_module = "./models/Motion_Module/mm_sd_v14.safetensors"

    # Carregar pipeline AnimateDiff
    adapter = MotionAdapter.from_pretrained(motion_module, subfolder="", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        base_model,
        motion_adapter=adapter,
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    init_image = load_image(image_path) if image_path and os.path.exists(image_path) else None

    num_frames = duration * fps

    result = pipe(
        prompt=prompt,
        num_frames=num_frames,
        guidance_scale=7.5,
        height=512,
        width=512,
        num_inference_steps=25,
        init_image=init_image
    )

    video_frames = [frame for frame in result.frames]
    video_frames_np = [np.array(f) for f in video_frames]
    imageio.mimsave(output_path, video_frames_np, fps=fps)
    print(f"✅ Vídeo salvo em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Descrição da animação (PT ou EN)")
    parser.add_argument("--duration", type=int, default=10, help="Duração em segundos (máx 60s)")
    parser.add_argument("--image", type=str, default=None, help="Caminho da imagem base (opcional)")
    parser.add_argument("--output", type=str, default="video.mp4", help="Arquivo de saída")
    parser.add_argument("--fps", type=int, default=8, help="Frames por segundo")
    args = parser.parse_args()

    main(args.prompt, args.duration, args.image, args.output, args.fps)
