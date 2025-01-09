#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is a simple FastAPI server that uses the langchain_openai library to chat with an OpenAI model.
It is a basic example that does not include any streaming capabilities or chat history.

'''

import json
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from fastapi import FastAPI, Request
from fastapi.responses import Response, FileResponse

LLM_MODEL = 'gpt-4o'

llm = ChatOpenAI(model=LLM_MODEL)

def do_chat(input: str):
  prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant.'),
    ('user', '{input}')
  ])
  chain = prompt | llm 
  return chain.invoke({'input': input})

app = FastAPI()

@app.get('/')
async def root():
  return FileResponse('../index.html')

@app.get('/chat/{prompt}')
@app.post('/chat')
async def chat(request: Request, prompt: Optional[str] = None): 
  
  if request.method == 'POST':
    body = await request.body()
    payload = json.loads(body)
    prompt = payload.get('prompt', '')

  return Response(content=do_chat(prompt).content, media_type='text/plain')  

if __name__ == '__main__':
  import uvicorn

  uvicorn.run(app, host='0.0.0.0', port=8080)