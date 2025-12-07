"""
ZeroProof Backend API
Handles AI-powered non-alcoholic drink recommendations using Anthropic Claude.

Environment Variables Required:
- ANTHROPIC_API_KEY: Your Anthropic API key
- ZEROPROOF_APP_KEY: Shared secret for iOS app authentication
"""

import os
import json
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import anthropic

# ---------------------------------------------------------------------------
# Logging & env setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load local .env in dev
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
APP_SHARED_KEY = os.getenv("ZEROPROOF_APP_KEY")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
if not APP_SHARED_KEY:
    raise RuntimeError("ZEROPROOF_APP_KEY not set in environment")

logger.info(f"Anthropic API key configured: {ANTHROPIC_API_KEY[:12]}...")
logger.info("ZeroProof app key configured")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Model configuration (update if you change your Anthropic model)
MODEL_ID = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 1200
TEMPERATURE = 0.8  # slightly higher for more variety

app = FastAPI(
    title="ZeroProof Backend",
    description="AI-powered non-alcoholic drink recommendation service",
    version="1.0.0",
)

# CORS – open for now (you can tighten this later if you want)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    """User's drink preference profile."""
    favoriteFlavors: List[str] = Field(default_factory=list, description="Flavors user enjoys")
    dislikedFlavors: List[str] = Field(default_factory=list, description="Flavors to avoid")
    preferredSweetness: Optional[str] = Field(None, description="Sweetness level preference")
    dietaryRestrictions: List[str] = Field(default_factory=list, description="Dietary constraints")
    complexityPreference: Optional[str] = Field(None, description="Simple, moderate, or complex")
    drinkStyles: List[str] = Field(default_factory=list, description="Preferred drink styles")
    usualMoods: List[str] = Field(default_factory=list, description="Typical moods when drinking")


class DrinkRequest(BaseModel):
    """Request body for drink recommendations."""
    profile: UserProfile
    mood: Optional[str] = Field(None, description="Current mood")
    ingredients: Optional[List[str]] = Field(None, description="Available ingredients")
    persona: Optional[str] = Field("Classic", description="Bartender persona style")
    flavorOverride: Optional[List[str]] = Field(None, description="Priority flavors for this request")
    numberOfSuggestions: int = Field(default=1, ge=1, le=5, description="Requested number of drinks")

    # History/context from the client (for variety)
    previousDrinkNames: Optional[List[str]] = Field(
        default=None,
        description="Names of drinks the user has already seen; avoid repeating these or trivial variants.",
    )
    overusedIngredients: Optional[List[str]] = Field(
        default=None,
        description="Ingredients that have been used too often recently; de-prioritize as primary flavors.",
    )


class Drink(BaseModel):
    """
    A complete drink recipe as sent to the iOS app.

    NOTE: ingredients are normalized to [String] so they decode cleanly on iOS.
    """
    name: str
    description: str
    ingredients: List[str]
    steps: List[str] = Field(description="Step-by-step instructions")
    variations: List[str] = Field(default_factory=list)
    mood: Optional[str] = None
    flavorProfile: List[str] = Field(default_factory=list)
    estimatedPrepTime: Optional[str] = None
    servings: Optional[int] = 1
    garnish: Optional[str] = None
    glassware: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class ResponseMetadata(BaseModel):
    """Metadata about the AI response."""
    model: str
    source: str = "anthropic"


class DrinkResponse(BaseModel):
    """Response containing drink recommendations."""
    drinks: List[Drink]
    reasoning: Optional[str] = None
    metadata: ResponseMetadata


# ---------------------------------------------------------------------------
# Persona prompts
# ---------------------------------------------------------------------------

PERSONA_PROMPTS = {
    "Classic": "Create traditional, balanced mocktails that are approachable and familiar. Focus on timeless flavor combinations.",
    "Mixologist": "Be creative and avant-garde. Experiment with unexpected flavor combinations and modern techniques. Think molecular mixology without the alcohol.",
    "Barista": "Focus on coffee and tea-based beverages. Incorporate espresso, matcha, chai, and other café-style ingredients.",
    "SpaChef": "Create wellness-focused drinks with superfoods, adaptogens, and health benefits. Think spa menus and wellness retreats.",
    "Tropical": "Use exotic fruits and tropical flavors. Create drinks that transport the user to a beach paradise.",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a world-class zero-proof mixologist.

You will be given:
- A user profile with flavor preferences.
- An optional mood.
- A list of previousDrinkNames (drinks the user has already seen this session).
- Optionally, a list of overusedIngredients (flavors that have been used too often recently).
- The desired numberOfSuggestions (the user-facing count; you may internally generate a few more options for variety).

Your job is to return an array of NEW non-alcoholic drink ideas as structured JSON.

IMPORTANT CONSTRAINTS:

1. NO REPEATS
    - Do NOT return any drink whose name or concept is the same as any in previousDrinkNames.
    - Do NOT return small variants of those drinks (e.g., "Ginger Fire Tonic", "Ginger Fire Fizz",
      "Spicy Ginger Mule" if "Spiced Ginger Mule" is in the list).
    - If a proposed drink would be too similar in name or core ingredients to something in previousDrinkNames,
      discard it and come up with something different.

2. VARY PRIMARY FLAVOR BASES
    - Avoid relying on the same primary flavor base too many times.
    - If overusedIngredients is provided (e.g., ["ginger", "jalapeño"]), then:
        - Do NOT make these the star or dominant flavor of new drinks.
        - You may include them in subtle supporting roles at most, but should strongly prefer other bases.
    - Aim to include a variety of bases over time: citrus, berry, stone fruit, tropical fruit, herbal,
      floral, tea-based, coffee-based, cola/soda, creamy/dessert, bitter, etc.
    - If several recent drinks clearly share a similar base (for example, multiple peppery citrus coolers
      in a row), deliberately switch to a noticeably different base for the new suggestions.

3. DISTINCT SUGGESTIONS PER CALL
    - Within a single response, each drink must be clearly distinct from the others in:
        - Primary flavor base
        - Structure (sour, spritz, tall highball, dessert-style, etc.)
    - Do NOT return several ginger-heavy drinks together.
    - Do NOT return trivial variants like Mule / Fizz / Tonic of the same flavor base in the same batch.

4. MATCH USER BUT AVOID RUTS
    - It’s OK if the user likes bold, spicy flavors, but do NOT get stuck on only ginger, jalapeño,
      or citrus+pepper riffs.
    - Explore that preference through many different flavor families:
        - smoky teas, peppercorns, chilies other than jalapeño, bitter citrus, herbal bitters, etc.

You must return ONLY valid JSON with this exact shape:

{
    "drinks": [
        {
            "name": string,
            "description": string,
            "ingredients": [string, ...],
            "instructions": string,
            "garnish": string,
            "glassware": string,
            "tags": [string, ...]
        }
    ],
    "reasoning": string
}

- "ingredients" must be an array of human-readable text strings like
    "2 oz fresh orange juice (freshly squeezed)".
- Do NOT make "ingredients" an array of objects or dictionaries.
- Do NOT include fields like "amount", "unit", or "notes" inside "ingredients".
Do NOT include any backticks.
Do NOT include the word "json".
Do NOT wrap the JSON in a code block.
Do NOT add extra top-level fields.
Return exactly one JSON object in this schema.

Rules:
- All ingredients must be non-alcoholic
- Include realistic measurements and clear steps
- Match the user's flavor preferences and mood
- Respect all dietary restrictions strictly
- Be creative but practical with common ingredients
- Each drink should be unique and memorable
"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_prompt(req: DrinkRequest) -> str:
    """Build the user prompt from the request."""
    p = req.profile
    sections: List[str] = []

    # Ask the model for at least 3 options for variety, even if user only sees 1
    effective_n = max(req.numberOfSuggestions, 3)

    sections.append(
        f"Please suggest {effective_n} non-alcoholic drink(s) based on the following preferences:"
    )
    sections.append("")
    sections.append("USER TASTE PROFILE:")
    sections.append(f"- Favorite flavors: {', '.join(p.favoriteFlavors) if p.favoriteFlavors else 'not specified'}")
    sections.append(f"- Disliked flavors: {', '.join(p.dislikedFlavors) if p.dislikedFlavors else 'none'}")
    sections.append(f"- Preferred sweetness: {p.preferredSweetness or 'not specified'}")
    sections.append(f"- Complexity preference: {p.complexityPreference or 'not specified'}")
    sections.append(f"- Drink styles: {', '.join(p.drinkStyles) if p.drinkStyles else 'not specified'}")
    sections.append(f"- Usual moods: {', '.join(p.usualMoods) if p.usualMoods else 'not specified'}")
    sections.append(f"- Dietary restrictions: {', '.join(p.dietaryRestrictions) if p.dietaryRestrictions else 'none'}")

    if req.mood:
        sections.append(f"\nCURRENT MOOD: {req.mood}")

    if req.flavorOverride:
        sections.append(f"\nFLAVOR OVERRIDE (prioritize these): {', '.join(req.flavorOverride)}")

    if req.ingredients:
        sections.append("\nAVAILABLE INGREDIENTS (create drinks using primarily these):")
        sections.append(", ".join(req.ingredients))
        sections.append("Please only suggest drinks that can be made with these available ingredients.")

    if req.previousDrinkNames:
        sections.append("\nPREVIOUS DRINK NAMES (avoid repeats or trivial variants of these):")
        for name in req.previousDrinkNames:
            sections.append(f"- {name}")

    if req.overusedIngredients:
        sections.append("\nOVERUSED INGREDIENTS (avoid making these the main flavor):")
        sections.append(", ".join(req.overusedIngredients))

    persona = req.persona or "Classic"
    persona_instruction = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["Classic"])
    sections.append(f"\nSTYLE INSTRUCTION: {persona_instruction}")

    sections.append("\nRespond with valid JSON only.")

    return "\n".join(sections)


def extract_json(text: str) -> str:
    """Extract JSON from response text, handling markdown code blocks."""
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    return cleaned


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check():
    """Simple health check."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/generate-drinks", response_model=DrinkResponse)
async def generate_drinks(
    request: DrinkRequest,
    x_app_key: Optional[str] = Header(None, alias="x-app-key"),
):
    """Generate non-alcoholic drink recommendations based on user preferences."""
    # Auth
    if x_app_key != APP_SHARED_KEY:
        logger.warning("Unauthorized request - invalid app key")
        raise HTTPException(status_code=401, detail="Unauthorized - invalid app key")

    prompt = build_prompt(request)
    logger.info(
        f"Generating {request.numberOfSuggestions} drink(s) with persona: {request.persona or 'Classic'}; "
        f"previous drinks: {request.previousDrinkNames or []}; "
        f"overused ingredients: {request.overusedIngredients or []}"
    )

    # Call Anthropic
    try:
        message = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error calling Anthropic: {e}")
        raise HTTPException(status_code=502, detail="AI service unavailable")

    # Extract text content
    try:
        text_block = next(block for block in message.content if getattr(block, "type", None) == "text")
        raw_json = text_block.text
    except StopIteration:
        logger.error("No text content in Anthropic response")
        raise HTTPException(status_code=502, detail="Invalid response from AI service")

    clean_json = extract_json(raw_json)

    # Parse JSON
    try:
        data = json.loads(clean_json)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Raw response (first 500 chars): {raw_json[:500]}...")
        raise HTTPException(status_code=502, detail="Failed to parse AI response")

    # Transform into our models
    try:
        drinks: List[Drink] = []

        for drink_data in data.get("drinks", []):
            # INGREDIENTS – normalize everything to strings
            raw_ingredients = drink_data.get("ingredients", [])
            ingredients: List[str] = []

            for ing in raw_ingredients:
                if isinstance(ing, str):
                    ingredients.append(ing)
                elif isinstance(ing, dict):
                    name = ing.get("name") or ing.get("ingredient") or ""
                    amount = ing.get("amount") or ing.get("quantity") or ""
                    unit = ing.get("unit") or ""
                    notes = ing.get("notes") or ""

                    parts = [p.strip() for p in [amount, unit, str(name)] if p and str(p).strip()]
                    line = " ".join(parts) if parts else str(name).strip()

                    if notes:
                        line = f"{line} ({notes})"

                    if line:
                        ingredients.append(line)

            # INSTRUCTIONS / STEPS – extremely defensive
            raw_instructions = (
                drink_data.get("instructions")
                or drink_data.get("steps")
                or drink_data.get("directions")
                or drink_data.get("method")
                or drink_data.get("preparation")
                or []
            )

            logger.info(
                f"Raw instructions for '{drink_data.get('name', 'Unnamed Drink')}': "
                f"type={type(raw_instructions)} value={raw_instructions}"
            )

            if isinstance(raw_instructions, str):
                # Split on periods and newlines into separate steps
                parts = raw_instructions.replace("\n", ".").split(".")
                steps = [s.strip() for s in parts if s.strip()]
            elif isinstance(raw_instructions, list):
                steps = [str(s).strip() for s in raw_instructions if str(s).strip()]
            else:
                steps = []

            # Absolute guarantee: never send an empty steps array to the app
            if not steps:
                steps = [
                    "Combine all ingredients in a glass with ice.",
                    "Stir or shake until well chilled and fully mixed.",
                    "Garnish as described and serve immediately.",
                ]

            logger.info(
                f"Final normalized steps for '{drink_data.get('name', 'Unnamed Drink')}': {steps}"
            )

            drink = Drink(
                name=drink_data.get("name", "Unnamed Drink"),
                description=drink_data.get("description", ""),
                ingredients=ingredients,
                steps=steps,
                variations=drink_data.get("variations", []),
                mood=drink_data.get("mood"),
                flavorProfile=drink_data.get("flavorProfile", []),
                estimatedPrepTime=drink_data.get("estimatedPrepTime"),
                servings=drink_data.get("servings", 1),
                garnish=drink_data.get("garnish"),
                glassware=drink_data.get("glassware"),
                tags=drink_data.get("tags", []),
            )

            drinks.append(drink)

        if not drinks:
            logger.error("No drinks in parsed response")
            raise HTTPException(status_code=502, detail="AI returned no drink recommendations")

        response = DrinkResponse(
            drinks=drinks,
            reasoning=data.get("reasoning"),
            metadata=ResponseMetadata(model=MODEL_ID, source="anthropic"),
        )

        logger.info(f"Successfully generated {len(drinks)} drink(s)")

        # Extra logging for debugging the shape we send back to iOS
        response_json = response.model_dump()
        logger.info(f"Response JSON top-level keys: {list(response_json.keys())}")
        if drinks:
            first = response_json["drinks"][0]
            logger.info(f"First drink keys: {list(first.keys())}")
            logger.info(f"First drink steps: {first.get('steps')}")
            logger.info(f"First drink ingredients: {first.get('ingredients')}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Response validation error: {e}")
        raise HTTPException(status_code=502, detail=f"Response validation error: {str(e)}")


# Legacy path for backwards compatibility
@app.post("/api/drinks/recommend", response_model=DrinkResponse)
async def recommend_drinks(
    request: DrinkRequest,
    x_app_key: Optional[str] = Header(None, alias="x-app-key"),
):
    return await generate_drinks(request, x_app_key)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)