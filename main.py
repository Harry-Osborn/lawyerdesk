# from fastapi import FastAPI, Body
# import random

# app = FastAPI()

# secret_number = random.randint(1, 100)
# attempts = 0

# def guess_number(guess: int):
#   global attempts, secret_number
#   attempts += 1
#   if guess == secret_number:
#     return {"message": f"Congratulations! You guessed the number in {attempts} attempts!"}
#   elif guess < secret_number:
#     return {"message": "Too low! Try again."}
#   else:
#     return {"message": "Too high! Try again."}

# @app.post("/guess")
# async def guess(guess: int = Body(...)):
#   """
#   This endpoint takes a guess as input and returns a message indicating if the guess is correct, too low, or too high.
#   """
#   response = guess_number(guess)
#   return response





from fastapi import FastAPI
from starlette.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
app = FastAPI()

# app.mount("/static", StaticFiles(directory="./"), name="static")

@app.get("/")
async def root():
    return {"message": "Hello World"}

class RequestSample(BaseModel):
    text:str
    

# Redirect to the welcome page for /welcome
@app.get("/welcome")
async def welcome(request:RequestSample):
    data = await request.json()
    text_ex = data["text"]
    return {
        "message":f"{text_ex}-----added"
    }
    # return FileResponse("static/welcome.html")


