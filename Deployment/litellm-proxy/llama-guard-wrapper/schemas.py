from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class ModerationRequest(BaseModel):
    input: str

# Not used for OpenAI response schema because wrapper returns OpenAI format directly
