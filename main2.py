import stat
from fastapi import Body, Form, FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, Any

class Quote(BaseModel):
    quote : str = Field(description="The quote that Nicolacus Maximus said")
    year : int = Field(description="the year when Nicolacus Maximus said the quote")

app = FastAPI(
    title= " NICOLOACUS MAXIMUS QUOTE GIVER",
    description="Get a real quote said by Nicolacus Maximus himself",
    servers=[
        {"url" : "https://vaccine-concepts-such-penguin.trycloudflare.com"}
    ]
)

@app.get("/quote", 
         summary="Returns a random quote by Nicolacus Maximus", 
         description="Upon receiving a GET request this endpoint will return a real quote said by Nicolacus Maximus himself.", 
         response_description="A quote object that contains the quote said by Nicolacus Maximus and the date when the quote was said",
         response_model=Quote,
         openapi_extra={
             "x-openai-isConsequential":True,}
         )


def get_quote(request : Request):
    print(request.headers)
    return {"quote":"Life is short. so eat it all","year":2025}


user_token_db = { "ABCDEF": "Park"}


@app.get("/authorize", response_class=HTMLResponse)
def handle_authorize(response_type: str, client_id: str, redirect_uri: str, scope: str, state: str):
    html_contents = f"""
    <html>
        <head>
            <title>Nicolas Maximus</title>
        </head>
        <body>
            <h1>Log into Nicolas Maximus</h1>
            <a href="{redirect_uri}?code=ABCDEF&state={state}">Authorize Nicolas Maximus GPT</a>
        </body>
    </html>
    """
    return HTMLResponse(content=html_contents) 

@app.post("/token")
def handle_token(code = Form(...)):
    return {
        "access_token":user_token_db[code],
    }
