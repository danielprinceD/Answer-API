from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import chatbot
app = FastAPI()

app.add_middleware(
    CORSMiddleware ,
    allow_origins= ["*"],
    allow_methods = ["*"]
)

class Message(BaseModel) :
    message : str


@app.get('/')
async def status():
    return { "status" : 200  }


@app.post('/api/health/post')
async def status(message : Message="None"):
    if message != "None":
        return {"answer" : chatbot.process_answer(message.message)}
    else :
        return {"message" : "Cant Answer !"}
    
@app.get('/api/get/medicine/list')
def list_medicine():
    medicine_string = '''1.Paracetamol (Acetaminophen) 
            2.Ibuprofen 
            3.Aspirin
            4.Amoxicillin
            5.Omeprazole
            6.Loratadine
            7.Metformin
            8.Prednisone
            9.Ciprofloxacin
            10.Ranitidine
            11.Metoprolol
            12.Levothyroxine
            13.Lisinopril
            14.Fluoxetine
            15.Amlodipine
            16.Losartan
            17.Albuterol
            18.Warfarin
            19.Citalopram
            20.Furosemide  
            Which medicine Detail you want ? '''
    medicine_list = [ _.strip() for _ in medicine_string.split("\n") ]
    return {"list" : medicine_list}

@app.post('/api/post/medicine')
def post_medicine(message : Message = "None"):
    if message != "None":
        return {"answer" : chatbot.getMedicine(message.message) }
    else :
        return {"message" : "Cant Answer !"}
    