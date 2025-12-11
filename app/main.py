from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Import your services
from app.services.pix2pix_inference import load_generator, run_pix2pix
from app.services.simple_classifier import classify_image

app = FastAPI()

# 1. CONFIGURE CORS
# This allows your Next.js frontend (localhost:3000) to talk to this backend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. SETUP MODEL & DIRECTORIES
netG = load_generator()

OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "API Pix2Pix funcionando!"}

@app.get("/health")
def health():
    return {"status": "backend online"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Define paths
    input_path = os.path.join(OUTPUT_DIR, "input_temp.jpg")
    output_path = os.path.join(OUTPUT_DIR, "output_pix2pix.jpg")

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Run Inference
    run_pix2pix(netG, input_path, output_path)

    # Return the actual image file to the frontend
    return FileResponse(output_path, media_type="image/jpeg")

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
