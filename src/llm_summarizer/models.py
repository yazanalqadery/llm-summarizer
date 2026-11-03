from pydantic import BaseModel

class Document(BaseModel):
    id: int
    title: str
    text: str

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    message: Message

class Usage(BaseModel):
    total_tokens: int = 0

class ChatCompletion(BaseModel):
    choices :list[Choice]
    usage: Usage = Usage()

class Summary(BaseModel):
    id: int
    title: str
    summary: str
    