from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps
import base64
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import io
import re

app = FastAPI()

load_dotenv()

def get_openai_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set on the server.",
        )

    return AsyncOpenAI(api_key=api_key)

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8000
MAX_OCR_DIMENSION = 1600
OCR_JPEG_QUALITY = 70
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
HEX_COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production e frontend domain diben
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "AI Image Backend Running"}

@app.post("/magic-eraser")
async def magic_eraser(
    image: UploadFile = File(...),
    mask: UploadFile = File(...)
):
    input_image_bytes = await read_valid_image(image)
    mask_bytes = await read_valid_image(mask)

    from simple_lama_inpainting import SimpleLama

    img = Image.open(io.BytesIO(input_image_bytes)).convert("RGB")
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")

    # Important: The mask might have grey pixels from scaling/brushing.
    # Convert mask strictly to binary (0 and 255) as LAMA expects a hard mask
    mask_img = mask_img.point(lambda p: 255 if p > 10 else 0)

    if img.size != mask_img.size:
        mask_img = mask_img.resize(img.size, Image.Resampling.NEAREST)

    orig_w, orig_h = img.size
    max_dim = max(orig_w, orig_h)
    
    # Let's increase limit to 2048 to retain high detail
    if max_dim > 2048:
        scale_factor = 2048.0 / max_dim
        new_w = max(1, int(orig_w * scale_factor))
        new_h = max(1, int(orig_h * scale_factor))
        
        # Round up to nearest multiple of 8, LAMA often expects dimensions divisible by 8 or 16
        new_w = new_w - (new_w % 8)
        new_h = new_h - (new_h % 8)

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        mask_img = mask_img.resize((new_w, new_h), Image.Resampling.NEAREST)

    # Ensure mask has the exact same size as the image before entering LAMA
    if mask_img.size != img.size:
        mask_img = mask_img.resize(img.size, Image.Resampling.NEAREST)

    try:
        from simple_lama_inpainting import SimpleLama

        simple_lama = SimpleLama()
        
        # Dilate mask slightly to cover object edges better 
        import numpy as np
        import cv2
        mask_np = np.array(mask_img)
        kernel = np.ones((5, 5), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=2)
        mask_img = Image.fromarray(mask_np)

        result = simple_lama(img, mask_img)
        
        if max_dim > 2048:
            result = result.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        output = io.BytesIO()
        result.save(output, format="JPEG", quality=95)
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=erased.jpg"},
        )
    except Exception as e:
        print(f"LAMA Image Edit failed: {e}")
        raise HTTPException(status_code=500, detail=f"LAMA processing failed: {str(e)}")

async def read_valid_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG, and WebP images are supported.",
        )

    input_bytes = await file.read()

    if len(input_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Image must be smaller than 10MB.",
        )

    return input_bytes


def remove_image_background(input_bytes: bytes) -> bytes:
    from rembg import remove

    return remove(input_bytes)


def open_image(input_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(input_bytes))
        return ImageOps.exif_transpose(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read this image.") from exc


def image_to_jpeg_stream(image: Image.Image, quality: int = 95) -> io.BytesIO:
    if image.mode in {"RGBA", "LA"}:
        alpha = image.getchannel("A")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image.convert("RGBA"), mask=alpha)
        final = background
    else:
        final = image.convert("RGB")

    output = io.BytesIO()
    final.save(output, format="JPEG", quality=quality, optimize=True)
    output.seek(0)
    return output


def optimize_image_for_ocr(input_bytes: bytes) -> tuple[bytes, str]:
    image = open_image(input_bytes)
    max_dim = max(image.width, image.height)

    if max_dim > MAX_OCR_DIMENSION:
        scale = MAX_OCR_DIMENSION / max_dim
        new_size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    if image.mode != "RGB":
        image = image.convert("RGB")

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=OCR_JPEG_QUALITY, optimize=True)
    output.seek(0)
    return output.read(), "image/jpeg"


@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    input_bytes = await read_valid_image(file)
    output_bytes = remove_image_background(input_bytes)

    return StreamingResponse(
        io.BytesIO(output_bytes),
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=background-removed.png"},
    )


@app.post("/apply-color-background")
async def apply_color_background(
    file: UploadFile = File(...),
    color: str = Form("#ffffff"),
    format: str = Form("png"),
):
    if not HEX_COLOR_PATTERN.match(color):
        raise HTTPException(
            status_code=400,
            detail="Color must be a valid hex color like #ffffff.",
        )

    input_bytes = await read_valid_image(file)
    subject_bytes = remove_image_background(input_bytes)
    subject = Image.open(io.BytesIO(subject_bytes)).convert("RGBA")

    bg_color = color.lstrip("#")
    r, g, b = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))

    background = Image.new("RGBA", subject.size, (r, g, b, 255))
    final = Image.alpha_composite(background, subject)

    output = io.BytesIO()

    if format == "jpg" or format == "jpeg":
        final.convert("RGB").save(output, format="JPEG", quality=95)
        media_type = "image/jpeg"
    elif format == "webp":
        final.save(output, format="WEBP", quality=95)
        media_type = "image/webp"
    else:
        final.save(output, format="PNG")
        media_type = "image/png"

    output.seek(0)

    extension = "jpg" if format in {"jpg", "jpeg"} else format

    return StreamingResponse(
        output,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename=background.{extension}"},
    )


@app.post("/resize-image")
async def resize_image(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
):
    if width < 1 or height < 1:
        raise HTTPException(status_code=400, detail="Width and height must be positive.")

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Width and height must be {MAX_IMAGE_DIMENSION}px or smaller.",
        )

    input_bytes = await read_valid_image(file)
    image = open_image(input_bytes)
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    output = image_to_jpeg_stream(resized, quality=95)

    return StreamingResponse(
        output,
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename=resized-{width}x{height}.jpg"},
    )


@app.post("/compress-image")
async def compress_image(
    file: UploadFile = File(...),
    target_kb: int = Form(...) 
):
    if target_kb < 1:
        raise HTTPException(status_code=400, detail="Target size must be at least 1KB.")

    if target_kb > MAX_FILE_SIZE // 1024:
        raise HTTPException(status_code=400, detail="Target size must be 10MB or smaller.")

    input_bytes = await read_valid_image(file)
    target_bytes = target_kb * 1024

    if len(input_bytes) <= target_bytes:
        extension = "jpg"
        media_type = file.content_type or "image/jpeg"

        if media_type == "image/png":
            extension = "png"
        elif media_type == "image/webp":
            extension = "webp"

        return StreamingResponse(
            io.BytesIO(input_bytes),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=under-{target_kb}kb.{extension}"},
        )

    img = open_image(input_bytes).convert("RGB")
    quality = 95
    
    output = io.BytesIO()
    
    while True:
        output.seek(0)
        output.truncate(0)
        
        img.save(output, format="JPEG", quality=quality, optimize=True)
        size = output.tell()
        
        if size <= target_bytes or quality <= 10:
            break
            
        quality -= 5

    output.seek(0)
    return StreamingResponse(
        output,
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename=compressed-{target_kb}kb.jpg"},
    )


@app.post("/extract-text")
async def extract_text_with_openai(file: UploadFile = File(...)):
    input_bytes = await read_valid_image(file)
    optimized_bytes, mime_type = optimize_image_for_ocr(input_bytes)

    base64_image = base64.b64encode(optimized_bytes).decode("utf-8")
    client = get_openai_client()
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all the text from this image. Return ONLY the extracted text. Maintain the original formatting, paragraphs, and language perfectly. If there is no text, reply with 'NO_TEXT_FOUND'.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                            },
                        },
                    ],
                },
            ],
            max_tokens=1500,
        )

        extracted_text = response.choices[0].message.content.strip()

        if extracted_text == "NO_TEXT_FOUND":
            return {"success": False, "text": ""}

        return {"success": True, "text": extracted_text}

    except Exception as e:
        return {"success": False, "error": str(e)}