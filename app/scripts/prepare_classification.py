import os
import shutil
from pathlib import Path

SRC = Path("Projeto_Ramularia")
OUT = Path("datasets/classification")

for folder in ["train/healthy", "train/diseased", "val/healthy", "val/diseased"]:
    (OUT / folder).mkdir(parents=True, exist_ok=True)

print("📁 Estrutura criada!")

def copy_all(src_dir, dst_dir):
    src = Path(src_dir)
    dst = Path(dst_dir)
    for img in src.iterdir():
        if img.is_file():
            shutil.copy(img, dst)

copy_all(SRC / "Healthy_Train50", OUT / "train/healthy")
copy_all(SRC / "Healthy_Test50", OUT / "val/healthy")
copy_all(SRC / "Disease_Test100", OUT / "val/diseased")

print("✅ Imagens classificadas copiadas com sucesso!")
print("⚠️ Não há imagens doentes para treino! Duplicando algumas de val/diseased para treino/diseased...")

diseased_val = list((OUT / "val/diseased").glob("*"))

for i, img in enumerate(diseased_val[:50]):  
    shutil.copy(img, OUT / "train/diseased")

print("🎉 Dataset preparado em datasets/classification/")
