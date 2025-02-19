from fastapi import APIRouter, File, UploadFile, HTTPException
import ffmpeg
import io
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, Dinov2WithRegistersForImageClassification
from models.video_model import VideoModel

def get_router(image_processor, model):
    router = APIRouter()

    def extract_middle_frame(video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count == 0:
            raise ValueError("Não foi possível extrair frames do vídeo.")

        middle_frame_index = frame_count // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Não foi possível ler o frame do meio.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return Image.fromarray(frame)

    @router.post("/upload/")
    async def upload_video(file: UploadFile = File(...)):
        MAX_SIZE_MB = 500  # 500MB
        MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

        if file.size and file.size > MAX_SIZE_BYTES:
            raise HTTPException(status_code=400, detail="O arquivo é muito grande! Máximo permitido: 500MB.")

        content_type = file.content_type

        if not content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="O arquivo deve ser um vídeo!")

        file_content = await file.read()
        video_path = f"/tmp/{file.filename}"

        with open(video_path, "wb") as f:
            f.write(file_content)

        try:
            probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='stream=duration')
            duration = float(probe['streams'][0]['duration'])
        except ffmpeg.Error as e:
            raise HTTPException(status_code=400, detail="Não foi possível processar o vídeo com FFmpeg.")

        try:
            middle_frame = extract_middle_frame(video_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao extrair o frame do meio: {str(e)}")

        inputs = image_processor(middle_frame, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = logits.argmax().item()
        label = model.config.id2label[predicted_label]

        video_data = VideoModel(
            filename=file.filename,
            content_type=content_type,
            size=file.size,
            duration=duration
        )

        return {
            "message": "Upload realizado com sucesso!",
            "video_info": video_data,
            "classification": label
        }

    return router
