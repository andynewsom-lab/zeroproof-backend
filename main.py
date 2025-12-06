import os
import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import anthropic

# Load local .env in dev
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
APP_SHARED_KEY = os.getenv("ZEROPROOF_APP_KEY")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not set")
if not APP_SHARED_KEY:
    raise RuntimeError("ZEROPROOF_APP_KEY not set")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI(title="ZeroProof Backend", version="1.0.0")


# ---------- Models ----------

class UserProfile(BaseModel):
    favoriteFlavors: List[str] = []
    dislikedFlavors: List[str] = []
    preferredSweetness: Optional[str] = None
    dietaryRestrictions: List[str] = []
    complexityPreference: Optional[str] = None


class DrinkRequest(BaseModel):
    profile: UserProfile
    mood: Optional[str] = None
    ingredients: Optional[List[str]] = None
    persona: Optional[str] = "Classic"


class Drink(BaseModel):
    name: str
    description: str
    ingredients: List[str]
    instructions: str
    garnish: Optional[str] = None
    glassware: Optional[str] = None
    tags: Optional[List[str]] = []


class DrinkResponse(BaseModel):
    drinks: List[Drink]


# ---------- Prompt builder ----------

def build_prompt(req: DrinkRequest) -> str:
    p = req.profile
    return f"""
You are an expert non-alcoholic mixologist creating drinks for a mobile app called ZeroProof.

User profile:
- Favorite flavors: {', '.join(p.favoriteFlavors) or 'none specified'}
- Disliked flavors: {', '.join(p.dislikedFlavors) or 'none specified'}
- Preferred sweetness: {p.preferredSweetness or 'unspecified'}
- Dietary restrictions: {', '.join(p.dietaryRestrictions) or 'none specified'}
- Complexity preference: {p.complexityPreference or 'unspecified'}

Context:
- Mood: {req.mood or 'unspecified'}
- Available ingredients: {', '.join(req.ingredients or []) or 'unspecified'}
- Persona style: {req.persona or 'Classic'}

Task:
- Propose 1–2 realistic, makeable, fully non-alcoholic drink recipes.
- Respect dietary restrictions strictly.
- Prefer common supermarket ingredients.
- Adjust sweetness and complexity to preferences.
- If ingredients are supplied, use them as the base when possible.

Respond with ONLY valid JSON:

{{
  "drinks": [
    {{
      "name": "string",
      "description": "string",
      "ingredients": ["string", "..."],
      "instructions": "string",
      "garnish": "string or null",
      "glassware": "string or null",
      "tags": ["string", "..."]
    }}
  ]
}}

No commentary or markdown, only the JSON object.
""".strip()


# ---------- Endpoint ----------

@app.post("/generate-drinks", response_model=DrinkResponse)
async def generate_drinks(request: DrinkRequest, x_app_key: str = Header(default=None)):
    if x_app_key != APP_SHARED_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = build_prompt(request)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",  # adjust model as needed
            max_tokens=800,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    # Extract text block
    try:
      text_block = next(
        block for block in message.content
        if getattr(block, "type", None) == "text"
      )
      raw_json = text_block.text
    except StopIteration:
      raise HTTPException(status_code=502, detail="No text content in LLM response")
      
    # Try to salvage a JSON object from the text
    clean = raw_json.strip()
  
    # If there's extra text, attempt to isolate the first {...} block
    if not clean.startswith("{"):
      start = clean.find("{")
      end = clean.rfind("}")
      if start == -1 or end == -1:
        # For debugging, you could log clean somewhere
        raise HTTPException(
          status_code=502,
          detail=f"Invalid JSON from model (no braces found)"
        )
      clean = clean[start : end + 1]
      
    try:
      data = json.loads(clean)
    except json.JSONDecodeError as e:
      # Again, you could log `clean` here if needed
      raise HTTPException(status_code=502, detail=f"Invalid JSON from model: {e}")
      
    # Let Pydantic validate and coerce
    try:
      response = DrinkResponse(**data)
    except Exception as e:
      raise HTTPException(status_code=502, detail=f"Response validation error: {e}")
      
    return response