import sys
import os

CYCLE_GAN_PATH = os.path.abspath(os.path.join(os.getcwd(), "pytorch-CycleGAN-and-pix2pix"))

if CYCLE_GAN_PATH not in sys.path:
    sys.path.append(CYCLE_GAN_PATH)

from models.networks import define_G
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

# Caminho do checkpoint treinado
CHECKPOINT_PATH = r"D:\IIA\Projeto 2\pytorch-CycleGAN-and-pix2pix\checkpoints\pix2pix_folhas\100_net_G.pth"

# ===== Carregar modelo Pix2Pix =====
def load_generator():
    # Importa o gerador do repositório do CycleGAN
    from models.networks import define_G

    netG = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="unet_256",
        norm="batch",
        use_dropout=False,
        init_type="normal",
        init_gain=0.02,
    )

    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    netG.load_state_dict(state_dict)
    netG.eval()

    return netG


# ===== Transformações =====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# ===== Função de inferência =====
def run_pix2pix(netG, image_path: str, output_path: str):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)  # adiciona batch dimension

    with torch.no_grad():
        fake = netG(img_t)[0]

    # Dessormalizar e converter para imagem
    fake = (fake * 0.5 + 0.5).clamp(0, 1)
    fake_img = transforms.ToPILImage()(fake.cpu())

    fake_img.save(output_path)
    return output_path
