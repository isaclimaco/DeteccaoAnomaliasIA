# Projeto – Detecção de Anomalias em Folhas (IA – UnB)

Backend desenvolvido em FastAPI contendo:

- Pipeline Pix2Pix
- Pipeline ΔE2000 (CIEDE2000)
- Pipeline Grad-CAM
- Endpoints para upload de imagem e retorno de resultados

## Como rodar

Instalar dependências:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fastapi
pip install uvicorn
pip install python-multipart
pip install pillow

Detecção de Anomalias com IA
Instruções para configurar e rodar o projeto (Backend e Frontend).
1. Backend
Instalação
 * Clone o repositório e navegue até a pasta do modelo:
<!-- end list -->
git clone https://github.com/isaclimaco/DeteccaoAnomaliasIA.git
cd DeteccaoAnomaliasIA/pytorch-CycleGAN-and-pix2pix

 * Passo Manual: Cole o conteúdo da pasta do Google Drive especificada dentro da pasta pytorch-CycleGAN-and-pix2pix: (https://drive.google.com/file/d/1eoV-3iX_H6gM36lPuqrdy0lun3zQEhXr/view?usp=sharing)
 * 
 * Instale as dependências:
<!-- end list -->
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn python-multipart pillow

Execução
Inicie o servidor backend:
python -m uvicorn app.main:app --reload

2. Frontend
Pré-requisitos
 * Node.js (ou via repositório da sua distro Linux).
Instalação
Em um novo terminal, clone e configure a interface:
git clone https://github.com/CaioCord987/Interface-Projeto-2-IIA.git
cd Interface-Projeto-2-IIA
npm install

Execução
Rode a aplicação:
npm run dev

Acesse a interface em: http://localhost:3000


