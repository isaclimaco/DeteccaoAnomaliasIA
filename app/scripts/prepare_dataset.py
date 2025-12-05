# Uso: python prepare_dataset.py --src_root "D:/IIA/Projeto 2/Projeto_Ramularia" --out_root "./datasets/folhas" --size 256
import os
import shutil
from PIL import Image
import argparse
from tqdm import tqdm

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def process_folder(src_folder, dest_A, dest_B, size=256):
    """
    src_folder: pasta com .jpg originais (RGB)
    dest_A: destino para A (RGB)
    dest_B: destino para B (grayscale)
    """
    ensure_dir(dest_A)
    ensure_dir(dest_B)

    files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()
    for f in tqdm(files, desc=os.path.basename(src_folder)):
        src_path = os.path.join(src_folder, f)
        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize((size, size), Image.LANCZOS)
            # save A (RGB)
            base_name = os.path.splitext(f)[0] + ".jpg"
            img.save(os.path.join(dest_A, base_name), quality=95)
            # save B (grayscale)
            gray = img.convert("L")  # single channel
            gray = gray.convert("RGB")  # convert back to 3ch so pix2pix aligned pairs are both 3ch
            gray.save(os.path.join(dest_B, base_name), quality=95)
        except Exception as e:
            print(f"Erro em {src_path}: {e}")

def main(args):
    src_root = os.path.abspath(args.src_root)
    out_root = os.path.abspath(args.out_root)
    size = args.size

    # Pastas originais (conforme seu layout)
    train_src = os.path.join(src_root, "Healthy_Train50")
    test_healthy = os.path.join(src_root, "Healthy_Test50")
    test_disease = os.path.join(src_root, "Disease_Test100")

    # destinos
    train_A = os.path.join(out_root, "train", "A")
    train_B = os.path.join(out_root, "train", "B")
    test_A = os.path.join(out_root, "test", "A")
    test_B = os.path.join(out_root, "test", "B")

    # limpa out_root se --clean
    if args.clean and os.path.exists(out_root):
        shutil.rmtree(out_root)

    # Preparar treino (apenas Healthy_Train50)
    process_folder(train_src, train_A, train_B, size=size)

    # Preparar teste -> unir Healthy_Test50 e Disease_Test100
    # colocamos todos em test/A e test/B (mantendo nomes únicos)
    process_folder(test_healthy, test_A, test_B, size=size)
    process_folder(test_disease, test_A, test_B, size=size)

    print("Dataset preparado em:", out_root)
    print("Estrutura esperada:")
    print(out_root + "/train/A  ->", os.listdir(train_A)[:3])
    print(out_root + "/train/B  ->", os.listdir(train_B)[:3])
    print(out_root + "/test/A   ->", len(os.listdir(test_A)), "arquivos")
    print(out_root + "/test/B   ->", len(os.listdir(test_B)), "arquivos")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, default=".", help="Pasta raiz com Healthy_Train50, Healthy_Test50, Disease_Test100")
    parser.add_argument("--out_root", type=str, default="./datasets/folhas", help="Destino para dataset pix2pix")
    parser.add_argument("--size", type=int, default=256, help="Tamanho (px) - imagens serão redimensionadas para size x size")
    parser.add_argument("--clean", action="store_true", help="Remover out_root antes de gerar")
    args = parser.parse_args()
    main(args)
