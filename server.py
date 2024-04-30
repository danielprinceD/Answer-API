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
    medicine_list = '''1 . Paracetamol (Acetaminophen) \n
            2.Ibuprofen \n
         3.Aspirin\n
            4.Amoxicillin\n
            5.Omeprazole\n
            6.Loratadine\n
            7.Metformin\n
            8.Prednisone\n
            9.Ciprofloxacin\n
            10.Ranitidine\n
            11.Metoprolol\n
            12.Levothyroxine\n
            13.Lisinopril\n
            14.Fluoxetine\n
            15.Amlodipine\n
            16.Losartan\n
            17.Albuterol\n
            18.Warfarin\n
            19.Citalopram\n
            20.Furosemide \n 
            Which medicine Detail you want ? '''
    return {"list" : medicine_list}

@app.post('/api/medicine/post')
def post_medicine(message : Message = "None"):
    if message != "None":
        
        return {"answer" : chatbot.getMedicine(message.message) }
    else :
        return {"message" : "Cant Answer !"}
    