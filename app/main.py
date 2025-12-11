from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
from app.services.pix2pix_inference import load_generator, run_pix2pix
from fastapi import File, UploadFile
from app.services.simple_classifier import classify_image


app = FastAPI()

# Carrega o modelo uma vez só
netG = load_generator()

OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {"status": "backend online"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Salvar imagem temporária
    input_path = os.path.join(OUTPUT_DIR, "input_temp.jpg")
    output_path = os.path.join(OUTPUT_DIR, "output_pix2pix.jpg")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Executar inferência
    result_path = run_pix2pix(netG, input_path, output_path)

    return {
        "message": "Inferência concluída",
        "output_image_path": result_path
    }

@app.post("/classify")
async def classify(file: UploadFile = File(...)):

    image_bytes = await file.read()
    label, conf = classify_image(image_bytes)

    return {
        "classe": label,
        "probabilidade": conf
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
