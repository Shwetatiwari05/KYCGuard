import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv("backend/.env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)

image_path = Path("/Users/shwetatiwari/Documents/Projects/finshield_model1/output_pan/fake/pan_fake_0219.jpg")

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

with open(image_path, "rb") as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        ),
        """
Extract all visible text from this Aadhaar card image.

Return ONLY the OCR text.
Preserve the text as accurately as possible.
Do not explain anything.
Do not guess missing text.
""",
    ],
)

print("\n========== GEMINI OCR OUTPUT ==========\n")
print(response.text)
print("\n========================================\n")
