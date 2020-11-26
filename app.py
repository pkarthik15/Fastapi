from typing import Optional, List
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
import io
import sys
import cv2
import numpy as np
import PIL
from PIL import Image
import easyocr


app = FastAPI()

reader = easyocr.Reader(['en', 'hi'])


class OCRResponse(BaseModel):
    filename: str
    id_type: str
    version: str
    ocr: List[str] = []


@app.post("/predict", response_model=OCRResponse)
async def predict(file: UploadFile = File(...), id_type: str = Form(...), version:str = Form(...)):
    # Ensure that this is an image
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        # Read image contents
        contents = await file.read() 
        pil_image = Image.open(io.BytesIO(contents))   
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  
        result = reader.readtext(open_cv_image, detail=0)                   
        return OCRResponse(
            filename = file.filename,
            id_type = id_type,
            version = version,
            ocr = result
        )
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))