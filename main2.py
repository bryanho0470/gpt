from fastapi import FastAPI
from pydantic import BaseModel, Field

class Quote(BaseModel):
    quote : str = Field(description="The quote that Nicolacus Maximus said")
    year : int = Field(description="the year when Nicolacus Maximus said the quote")

app = FastAPI(
    title= " NICOLOACUS MAXIMUS QUOTE GIVER",
    description="Get a real quote said by Nicolacus Maximus himself",
    servers=[
        {"url" : "https://jesse-seconds-broad-recognised.trycloudflare.com"}
    ]
)

@app.get("/quote", 
         summary="Returns a random quote by Nicolacus        Maximus", description="Upon receiving a GET request this endpoint will return a real quote said by Nicolacus Maximus himself.", response_description="A quote object that contains the quote said by Nicolacus Maximus and the date when the quote was said",
         response_model=Quote,
         openapi_extra={
             "x-openai-isConsequential":True,
         }
         )

def get_quote():
    return {"quote":"Life is short. so eat it all","year":2025}