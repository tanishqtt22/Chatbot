from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize FastAPI app
app = FastAPI(title="Ollama Chatbot API")

# Initialize LLM and memory
llm = Ollama(model="llama3")
memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Request schema
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_input = request.message.strip()

    if user_input.lower() in ["exit", "quit", "bye"]:
        return {"response": "Goodbye! 👋"}

    if user_input.lower() == "show memory":
        history = memory.load_memory_variables({})["history"]
        return {"history": [str(msg) for msg in history]}

    response = conversation.invoke(user_input)
    return {"response": response["response"]}

@app.get("/")
def root():
    return {"message": "🤖 Chatbot API is running! Send POST /chat with { 'message': 'your text' }"}
