from pydantic import BaseModel

class InputPrompt(BaseModel):
    user_english_query: str