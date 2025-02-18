# !pip install nest_asyncio
import nest_asyncio
nest_asyncio.apply()

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.llms import LlamaCpp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for crime clips
CRIME_CLIPS_DIR = "crime_clips"
os.makedirs(CRIME_CLIPS_DIR, exist_ok=True)

# Set device (CPU/GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    torch_dtype=torch.float16
).to(device)

# Load Llama Model for generating reports
model_filename = "mistral-7b-instruct.Q4_K_M.gguf"
llm = LlamaCpp(
    model_path=model_filename,
    n_gpu_layers=40,
    n_batch=512,
    n_threads=8,
    n_ctx=1024,
    max_tokens=300,
    temperature=0.3,
    verbose=False
)

# Suspicious keywords for crime detection
SUSPICIOUS_KEYWORDS = ["weapon", "fighting", "theft", "violence", "crime", "assault"]


def process_video(video_path, frame_skip=5, batch_size=4):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    descriptions = []
    crime_clips = []
    frames_batch = []
    timestamps = []
    frame_count = 0

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), disable=False) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                timestamp = frame_count / fps
                frames_batch.append(cv2.resize(frame, (640, 480)))
                timestamps.append(timestamp)

                if len(frames_batch) == batch_size:
                    pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_batch]
                    inputs = processor(images=pil_images, return_tensors="pt").to(device, torch.float16)

                    with torch.no_grad():
                        outputs = model.generate(**inputs)

                    batch_descriptions = processor.batch_decode(outputs, skip_special_tokens=True)
                    descriptions.extend(batch_descriptions)
                    frames_batch = []

            frame_count += 1
            pbar.update(1)

    cap.release()

    # Identify crime timestamps
    crime_timestamps = [timestamps[i] for i, desc in enumerate(descriptions) if any(w in desc.lower() for w in SUSPICIOUS_KEYWORDS)]

    # Extract crime clips
    for idx, start_time in enumerate(crime_timestamps):
        end_time = start_time + 5  # Extract 5-second clip
        clip_path = os.path.join(CRIME_CLIPS_DIR, f"crime_clip_{idx}.mp4")
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=clip_path)
        crime_clips.append(f"http://127.0.0.1:8000/clips/{os.path.basename(clip_path)}")

    return descriptions, crime_clips


def generate_report(descriptions):
    detected = [desc for desc in descriptions if any(word in desc.lower() for word in SUSPICIOUS_KEYWORDS)]

    if not detected:
        return "✅ No Suspicious Activity Detected."

    summary = " ".join(detected[:20])  # Limit to 20 frames
    prompt = f"""Analyze these security camera observations:
    {summary}

    Generate a concise police report with:
    1. Incident Summary
    2. Suspect Details
    3. Key Timeline
    4. Recommended Actions
    Keep it within 800 tokens.
    """

    return llm(prompt)


@app.get("/")
def home():
    return {"message": "🚔 Crime Detection API is Running!"}


@app.post("/process")
async def process(file: UploadFile = File(...)):
    file_path = os.path.join("uploaded_video.mp4")

    # Save uploaded video
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    descriptions, crime_clips = process_video(file_path)
    report = generate_report(descriptions)

    return JSONResponse(content={"report": report, "clips": crime_clips})


@app.get("/clips/{filename}")
def get_clip(filename: str):
    clip_path = os.path.join(CRIME_CLIPS_DIR, filename)
    if os.path.exists(clip_path):
        return FileResponse(clip_path, media_type="video/mp4")
    return JSONResponse(content={"error": "File not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)