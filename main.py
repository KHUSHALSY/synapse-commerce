import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize Gemini Client
# The Client() automatically looks for the GEMINI_API_KEY variable in your terminal
client = genai.Client()

# 2. Frontend Request Model
class VideoRequest(BaseModel):
    videoUrl: str

# 3. AI Structured Output Model
class AIStrategyResult(BaseModel):
    audience_intent: str = Field(description="A 3 to 5 word summary of the psychological reason a user watches this.")
    recommended_niche: str = Field(description="A specific, highly profitable e-commerce niche to target based on the video.")
    engagement_score: int = Field(description="An estimated virality and engagement score from 1 to 100.")

@app.post("/api/analyze-video")
async def analyze_video(request: VideoRequest):
    url = request.videoUrl
    print(f"\n[SYNAPSE LOG] Pinging Gemini Engine (3.6-flash) for URL: {url}...\n")
    
    prompt = f"Analyze this viral video URL: {url}. Predict the audience intent, suggest an optimal e-commerce niche for affiliate marketing, and calculate a predicted engagement score."

    try:
        # 4. Call the latest Gemini 3.6 Flash model as required by Google
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AIStrategyResult,
            )
        )
        
        # 5. Extract the parsed Pydantic object and convert it to a dictionary
        ai_data = response.parsed.model_dump()
        
        return {
            "status": "success",
            "processed_url": url,
            "ai_analysis": ai_data,
            "action": "Display dashboard"
        }

    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        return {"status": "error", "message": "API Connection Failed. Check server logs."}

# 6. Production-Ready Boot Sequence
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)