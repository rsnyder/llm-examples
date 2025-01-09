#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, secrets
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware

LLM_MODEL = 'gpt-4o'

llm = ChatOpenAI(model=LLM_MODEL)

def do_chat(prompt: str):
  
  messages = [
    SystemMessage(content='You are a helpful assistant.'),
    HumanMessage(content=prompt)
  ]

  return llm(messages)

app = FastAPI()

# Mount static files
class CacheControlStaticFiles(StaticFiles):
  def file_response(self, *args, **kwargs) -> Response:
    response = super().file_response(*args, **kwargs)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

app.mount("/static", CacheControlStaticFiles(directory="static"), name="static")
app.mount('/static', StaticFiles(directory='static'), name='static')

# CORS
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

@app.get('/')
async def root():
  return FileResponse('index.html')

@app.get('/chat/{prompt}')
@app.post('/chat')
async def chat(request: Request, prompt: Optional[str] = None, session_id: Optional[str] = None, stream: Optional[bool] = False): 
  
  if request.method == 'POST':
    body = await request.body()
    payload = json.loads(body)
    prompt = payload.get('prompt', '')
    sessionid = payload.get('sessionid', secrets.token_hex(4))
    stream = payload.get('stream', False)
    
  print(f'prompt={prompt}, sessionid={sessionid}, stream={stream}')

  resp = do_chat(prompt)

  return Response(content=resp.content, media_type='text/plain')  

if __name__ == '__main__':
  import uvicorn

  uvicorn.run(app, host='0.0.0.0', port=8080)