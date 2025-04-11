# Import necessary libraries and modules
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import base64
from fastapi.staticfiles import StaticFiles
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
from music21 import converter, meter
import re

# Initialize FastAPI app and templates
app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load the Stable Diffusion image generation model
model_path = 'D:/innopolis/courses/3_2/GAI/project/FastAPILocal/app/static/final_model'
pipe = StableDiffusionPipeline.from_pretrained(model_path, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe.to(device)

# Load the T5-based music generation model
music_model = T5ForConditionalGeneration.from_pretrained(
    "D:/innopolis/courses/3_2/GAI/project/FastAPILocal/app/static/t5_abc_model"
).to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Standardize ABC notation format
def standardize_abc(abc: str) -> str:
    abc = abc.strip()
    if not abc.startswith("X:"):
        abc = "X:1\nT:Generated\n" + abc
    if "M:" not in abc:
        abc = "M:4/4\n" + abc
    if "K:" not in abc:
        abc = "K:C\n" + abc
    if "Z:" not in abc:
        abc += "\nZ:1"
    return abc

# Heuristic check for spam ABC music
def is_spam(abc: str) -> bool:
    tokens = abc.split()
    return tokens.count("f2") > 15 or len(set(tokens)) < 6

# Remove unnecessary trailing tokens and ensure clean output
def clean_abc_output(abc: str) -> str:
    abc = re.split(r"Z:", abc)[0]
    if "K:" in abc:
        abc = "K:" + abc.split("K:")[1]
    return abc.strip()

# Generate ABC music string using T5 model with retries
def generate_abc(prompt: str, max_new_tokens=400, retries=3) -> str:
    for _ in range(retries):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = music_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.7,
            no_repeat_ngram_size=6,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        abc_cleaned = clean_abc_output(decoded)
        if not is_spam(abc_cleaned):
            return abc_cleaned
    return abc_cleaned

# Save ABC notation as a MIDI file using music21
def save_abc_to_midi(abc_string: str, filename="generated.mid", folder="app/static") -> str | None:
    abc = re.sub(r"(?m)^[XTMKLZ]:.*$", "", abc_string).strip()
    abc_full = f"X:1\nT:Generated\nM:4/4\nK:C\n{abc}\nZ:1"
    try:
        score = converter.parse(abc_full, format='abc')
        # Ensure time signature is correct
        for ts in score.recurse().getElementsByClass(meter.TimeSignature):
            ts.activeSite.replace(ts, meter.TimeSignature("4/4"))
        midi_path = f"{folder}/{filename}"
        score.write("midi", fp=midi_path)
        return midi_path
    except Exception as e:
        print("❌ Error generating MIDI:", e)
        return None

# Define Pydantic model for text input
class TextRequest(BaseModel):
    text: str

# Route: Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Route: Survey page
@app.get("/survey", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("survey.html", {"request": request})

# Route: Image generation page
@app.get("/generate-page", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route: Music generation page
@app.get("/generate-music-page", response_class=HTMLResponse)
async def generate_music_page(request: Request):
    return templates.TemplateResponse("music_generate.html", {"request": request})

# API Endpoint: Generate image from prompt using Stable Diffusion
@app.post("/generate")
async def generate_image(request: Request, text_request: TextRequest):
    text = text_request.text
    with torch.no_grad():
        image = pipe(text).images[0]

    # Convert image to base64-encoded string
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    image_url = f"data:image/png;base64,{base64.b64encode(img_byte_arr.getvalue()).decode()}"

    return JSONResponse(content={"image_url": image_url})

# API Endpoint: Generate ABC music and convert to MIDI
@app.post("/generate-music")
async def generate_music(request: Request, text_request: TextRequest):
    prompt_text = text_request.text
    abc_string = generate_abc(prompt_text)

    if abc_string:
        filename = "generated_music.mid"
        midi_path = save_abc_to_midi(abc_string, filename=filename)

        if midi_path:
            return JSONResponse(content={"midi_url": f"/static/{filename}"})
        else:
            return JSONResponse(content={"error": "Error generating MIDI file."}, status_code=500)
    else:
        return JSONResponse(content={"error": "Error generating music."}, status_code=500)

# Serve static files like MIDI and generated images
app.mount("/static", StaticFiles(directory="app/static"), name="static")
