#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, json, secrets
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response, FileResponse

LLM_MODEL = 'gpt-4o'

llm = ChatOpenAI(model=LLM_MODEL)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id] = ChatMessageHistory()
  return store[session_id]
  
prompt = ChatPromptTemplate.from_messages([
  ('system', 'You are a helpful assistant.'),
  MessagesPlaceholder(variable_name='chat_history'),
  ('user', '{input}')
])
chain = prompt | llm

conversational_chain = RunnableWithMessageHistory(
  chain,
  get_session_history,  
  input_messages_key='input',
  history_messages_key='chat_history'
)

async def generate_chat_events(input, session_id):
  async for evt in conversational_chain.astream_events(input, version='v1', config={'configurable': {'session_id': session_id}} ):
    if evt['event'] == 'on_chain_stream' and 'seq:step:2' in evt['tags']:
      yield evt['data']['chunk'].content

app = FastAPI()

@app.get('/')
async def root():
  return FileResponse('../index.html')

@app.get('/chat/{prompt}')
@app.post('/chat')
async def chat(request: Request, prompt: Optional[str] = None, sessionid: Optional[str] = None, stream: Optional[bool] = False): 
  
  if request.method == 'POST':
    body = await request.body()
    payload = json.loads(body)
    prompt = payload.get('prompt', '')
    stream = payload.get('stream', False)
    sessionid = payload.get('sessionid', secrets.token_hex(4))
    
  if stream:
    return StreamingResponse(generate_chat_events({'input': prompt, 'chat_history': []}, sessionid), media_type='text/event-stream')
  else:
    resp = conversational_chain.invoke({'input': prompt}, config={'configurable': {'session_id': sessionid}})
    return Response(content=resp.content, media_type='text/plain')

if __name__ == '__main__':
  import uvicorn

  uvicorn.run(app, host='0.0.0.0', port=8080)