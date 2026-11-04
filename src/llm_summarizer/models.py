from pydantic import BaseModel, ConfigDict, Field


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
    choices :list[Choice] = Field(min_length=1)
    usage: Usage = Usage()

class Summary(BaseModel):
    id: int
    title: str
    summary: str

class AnalyzedDoc(BaseModel):
    id: int
    title: str
    summary: str
    keywords: list[str]
    sentiment: str
    
class Analysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    keywords: list[str]
    sentiment: str
