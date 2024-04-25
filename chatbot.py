import os
from pinecone import Pinecone
from dotenv import load_dotenv
import numpy as np
from langchain.llms import HuggingFacePipeline
import base64
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import SentenceTransformerEmbeddings 

load_dotenv()


os.environ['CURL_CA_BUNDLE'] = ''
pc = Pinecone(api_key=os.getenv('PINECONE_API'))
index_name = "health-bot"

device = torch.device('cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)


def llm_pipeline(meta , text):
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.2,
        top_p= 0.20,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    chain = load_qa_chain(local_llm , chain_type="stuff")
    ans = chain.run(input_documents = meta , question = text)
    return ans


def process_answer(instruction):


    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    index = pc.Index(index_name)
    text = instruction

    text_embed = embeddings.embed_query(text)
    get_response = index.query(
        namespace = "np10",
        vector = text_embed,
        top_k =  5,
        includeMetadata = True

    )

    meta = [ i.metadata['text'] for i in  get_response.matches]
    # result = ""
    # for i in get_response.matches :
    #     result = result + " " + i.metadata['text']
    result = {
        "page_content" : meta ,
        "metadata" : []
    }
    
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.20,
    )
    
    local_llm = HuggingFacePipeline(pipeline=pipe)
    chain = load_qa_chain(local_llm , chain_type="stuff")
    ans = chain.run(input_documents = result  , question = text)
    print(text)
    print(ans)
    return ans

# print(process_answer("what is depression ? "))