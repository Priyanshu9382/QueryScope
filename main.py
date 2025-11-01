from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Query
from semanticscholar import AsyncSemanticScholar
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage



api_key = os.getenv('GOOGLE_API_KEY')

model = init_chat_model("google_genai:gemini-2.5-flash-lite", api_key=api_key)


client = AsyncSemanticScholar()

app = FastAPI()

async def search_paper(query : str):
    print("In process-query")
    if not query :
        raise HTTPException(400, "Query is required")
    papers =  await client.search_paper(query = query, limit= 10)
    print('paper fetching done')
    return papers.items

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/process-query")
async def generate_summary(query:str):
    papers = await search_paper(query=query)
    print(papers)
    conversations = [
        SystemMessage("You are a research assistant helping a scientist understand the current state of research on a given topic.The scientist provides you with abstracts of the 10 most recent research papers retrieved from Semantic Scholar related to a specific question.Your task is to carefully read all the abstracts and provide a clear, structured summary that includes:1. Overall research direction – what is the main focus of current research on this topic?2.Key findings or conclusions – what are the most important discoveries or consensus points?3.Gaps or limitations – what questions remain unanswered or underexplored?Trends and emerging ideas - what new methods, perspectives, or technologies are being explored?Concise synthesis – summarize everything in a coherent narrative that a researcher can quickly understand.Keep the explanation factual and analytical, not speculative. Avoid simply rephrasing abstracts — instead, synthesize insights across multiple papers."),
        HumanMessage(f"Give a summary of {papers}")
    ]
    summary = None
    for chunk in model.stream(conversations):
        summary = chunk if summary is None else summary + chunk
        print(summary)
    return {"summaries": summary}