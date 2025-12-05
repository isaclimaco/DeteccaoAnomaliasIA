from fastapi import FastAPI, UploadFile, File
from app.pipelines.pix2pix import run_pix2pix
from app.pipelines.anomaly import calculate_anomaly
from app.pipelines.gradcam import run_gradcam

app = FastAPI(title="Detecção de Anomalias em Folhas")

@app.get("/")
def root():
    return {"message": "API funcionando! Envie imagens para análise."}

@app.post("/reconstruir")
async def reconstruir_folha(file: UploadFile = File(...)):
    imagem_bytes = await file.read()
    reconstruida = run_pix2pix(imagem_bytes)
    return {"status": "ok", "mensagem": "Reconstrução feita", "resultado": reconstruida}

@app.post("/anomalia")
async def detectar_anomalia(file: UploadFile = File(...)):
    imagem_bytes = await file.read()
    resultado = calculate_anomaly(imagem_bytes)
    return {"status": "ok", "anomalia": resultado}

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    imagem_bytes = await file.read()
    heatmap = run_gradcam(imagem_bytes)
    return {"status": "ok", "heatmap": heatmap}
