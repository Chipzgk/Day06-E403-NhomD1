from __future__ import annotations
from pydantic import BaseModel, Field


class MappingItem(BaseModel):
    technical_component: str
    analogy_element: str


class AnalogyResult(BaseModel):
    concept: str
    interest: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    narrative: str
    mapping_table: list[MappingItem]
    is_fallback: bool = False
    fallback_reason: str = ""
    suggestions: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""