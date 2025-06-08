from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline
from .bias_analysis import analyze_and_summarize, extract_article_info_fallback

import uvicorn
import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):  # Convert tuple to list
        return [convert_numpy_types(i) for i in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return float(obj)
    else:
        return obj

app = FastAPI(title="News Bias & Summary Analyzer")

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class AnalyzeRequest(BaseModel):
    article: str = None
    source: str = None
    url: str = None  # Optional URL field

class AnalyzeResponse(BaseModel):
    bias_report: dict
    summary: str

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest):
    article = request.article
    source = request.source

    # If URL is provided, extract article and source from it
    if request.url:
        extracted = extract_article_info_fallback(request.url)
        article = extracted['text']
        source = extracted['source']
        print(f"[URL Extracted] Source: {source} | Title: {extracted['title']}")

    # Ensure article and source are available
    if not article or not source:
        return {"bias_report": {}, "summary": "Invalid input: article or source missing."}

    # Run bias analysis & get result
    result = analyze_and_summarize(article, source)
    print("analyze_article result:", result)

    if isinstance(result, tuple) and len(result) == 2:
        bias_report_raw, bias_summary_from_analysis = result
    else:
        bias_report_raw = result
        bias_summary_from_analysis = None

    bias_report_clean = convert_numpy_types(bias_report_raw)

    # Use summary from analysis if present; fallback to model-generated summary
    if bias_summary_from_analysis and len(bias_summary_from_analysis.strip()) > 0:
        summary = bias_summary_from_analysis
    else:
        summary_result = summarizer(article, max_length=130, min_length=60, do_sample=False)
        summary = summary_result[0]['summary_text']

    return {
        "bias_report": bias_report_clean,
        "summary": summary
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
