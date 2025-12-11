import sys
import os
import torch
from PIL import Image
from torchvision import transforms

# 1. SETUP PATHS DYNAMICALLY
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
CYCLE_GAN_DIR = os.path.join(PROJECT_ROOT, "pytorch-CycleGAN-and-pix2pix")
CHECKPOINT_PATH = os.path.join(CYCLE_GAN_DIR, "checkpoints", "pix2pix_folhas", "100_net_G.pth")

# 2. MANAGE IMPORTS
if CYCLE_GAN_DIR not in sys.path:
    sys.path.append(CYCLE_GAN_DIR)

try:
    from models.networks import define_G
except ImportError as e:
    raise ImportError(f"Could not import 'models.networks'. Checked path: {CYCLE_GAN_DIR}. Error: {e}")

# ===== Carregar modelo Pix2Pix =====
def load_generator():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint file not found!\n"
            f"Expected location: {CHECKPOINT_PATH}\n"
            f"Please ensure the 'checkpoints' folder is inside 'pytorch-CycleGAN-and-pix2pix'."
        )

    # REMOVED 'gpu_ids=[]' to fix the TypeError
    netG = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="unet_256",
        norm="batch",
        use_dropout=False,
        init_type="normal",
        init_gain=0.02
        # gpu_ids=[]  <-- Deleted this line
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

    # Desnormalizar e converter para imagem
    fake = (fake * 0.5 + 0.5).clamp(0, 1)
    fake_img = transforms.ToPILImage()(fake.cpu())

    fake_img.save(output_path)
    return output_path
