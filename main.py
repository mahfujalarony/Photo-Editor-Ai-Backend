from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import re

app = FastAPI()

MAX_FILE_SIZE = 10 * 1024 * 1024
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
