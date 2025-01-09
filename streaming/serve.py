#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, json
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response, FileResponse

LLM_MODEL = 'gpt-4o'

llm = ChatOpenAI(model=LLM_MODEL)

def get_chain():
  prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('user', '{input}')
  ])
  chain = prompt | llm 
  return chain

def do_chat(input: str):
  return get_chain.invoke({'input': input})

async def generate_chat_events(input):
  async for evt in get_chain().astream_events(input, version='v1' ):
    if evt['event'] == 'on_chain_stream':
      chunk = evt['data']['chunk'].content if evt.get('data').get('chunk') else None
      if chunk:
        yield chunk

app = FastAPI()

@app.get('/')
async def root():
  return FileResponse('../index.html')

@app.get('/chat/{prompt}')
@app.post('/chat')
async def chat(request: Request, prompt: Optional[str] = None, session_id: Optional[str] = None, stream: Optional[bool] = False): 
  
  if request.method == 'POST':
    body = await request.body()
    payload = json.loads(body)
    prompt = payload.get('prompt', '')
    stream = payload.get('stream', False)
    
  print(f'prompt={prompt}, stream={stream}')

  if stream:
    return StreamingResponse(generate_chat_events({'input': prompt}), media_type='text/event-stream')
  else:
    resp = do_chat(prompt)
    return Response(content=resp.content, media_type='text/plain')  

if __name__ == '__main__':
  import uvicorn

  uvicorn.run(app, host='0.0.0.0', port=8080)