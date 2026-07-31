"""Request/response schemas for the DomainBot gateway.

Validation is a security boundary, not a formality. The rules here defend against:
  - oversized prompts (a cheap DoS)
  - unbounded conversation history (memory + cost blowup)
  - clients injecting their own `role: system` message (prompt-injection: a client
    trying to overwrite the server's persona / safety instructions)

The server owns the system prompt. Clients send only user/assistant turns.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    # NOTE: 'system' is deliberately NOT allowed here. The server injects the
    # single system prompt; a client-supplied system message is rejected.
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=512)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def last_must_be_user(cls, v: list[Message]) -> list[Message]:
        if v[-1].role != "user":
            raise ValueError("the last message must have role 'user'")
        return v


class ChatResponse(BaseModel):
    content: str
    model: str
    revision: str
    usage: dict


class ModelInfo(BaseModel):
    repo_id: str
    revision: str
    engine_url: str
    engine_model_name: str


class HealthResponse(BaseModel):
    status: str
    detail: str | None = None
