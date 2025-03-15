from fastapi import Form, FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import getpass
import time
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4

from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "recipes"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 

vector_store = PineconeVectorStore.from_existing_index(
    
    embedding=embeddings,
    index_name=index_name,
)

app = FastAPI(
    title= "ChefGPT. The best provider of Indian Recipes in the world.",
    description="Give ChefGPT a couple of ingredients and it will give you recipes in return.",
    servers=[
        {"url" : "https://networking-prostate-steam-conferences.trycloudflare.com"}
    ]
)

class Document(BaseModel):
    page_content: str

@app.get("/recipes", 
         summary="Return a list of recipes.", 
         description="Upon receiving an ingredient, this endpoint will return a list of recipes that contain that ingredient", 
         response_description="A document object that contains the recipe and preparetion instructions",
         response_model=list[Document],
         openapi_extra={
             "x-openai-isConsequential":False,}
         )

def get_recipes(ingredient: str):
    search = vector_store.similarity_search(ingredient)
    return search

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
