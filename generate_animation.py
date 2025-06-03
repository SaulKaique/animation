import os
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from PIL import Image
import argparse
import imageio
import numpy as np
from huggingface_hub import login

def load_image(image_path):
    return Image.open(image_path).convert("RGB").resize((512, 512))

def main(prompt, duration=10, image_path=None, output_path="video/edit.mp4", fps=8, hf_token=None):
    if duration > 60:
        print("⚠️ Duração maior que 60s não é suportada, ajustando para 60s")
        duration = 60

    base_model = "runwayml/stable-diffusion-v1-5"

    # Coloque aqui o modelo correto e disponível do motion adapter
    motion_module = "guoyww/animatediff-motion-adapter-v1-4"

    # Se tiver token, passa para from_pretrained para autenticar
    token_args = {"use_auth_token": hf_token} if hf_token else {}

    print("🔄 Carregando o modelo e o pipeline...")

    adapter = MotionAdapter.from_pretrained(motion_module, **token_args)
    pipe = AnimateDiffPipeline.from_pretrained(
        base_model,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
        **token_args
    ).to("cuda")

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    init_image = load_image(image_path) if image_path and os.path.exists(image_path) else None

    num_frames = duration * fps

    print("🎬 Gerando frames da animação... Isso pode levar alguns minutos.")

    result = pipe(
        prompt=prompt,
        num_frames=num_frames,
        guidance_scale=7.5,
        height=512,
        width=512,
        num_inference_steps=25,
        init_image=init_image
    )

    video_frames_np = [np.array(f) for f in result.frames]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, video_frames_np, fps=fps)

    print(f"✅ Vídeo salvo em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Descrição em português ou inglês")
    parser.add_argument("--duration", type=int, default=10, help="Duração em segundos (máx 60s)")
    parser.add_argument("--image", type=str, default=None, help="Caminho da imagem base (opcional)")
    parser.add_argument("--output", type=str, default="video/edit.mp4", help="Nome do arquivo de saída")
    parser.add_argument("--fps", type=int, default=8, help="Frames por segundo")
    parser.add_argument("--token", type=str, default=None, help="Token Hugging Face para modelos privados")

    args = parser.parse_args()

    # Faz login se token for passado (opcional)
    if args.token:
        login(token=args.token)

    main(args.prompt, args.duration, args.image, args.output, args.fps, args.token)
