from langchain.chains.question_answering.map_reduce_prompt import messages
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes

load_dotenv()

model = ChatOpenAI( model="gpt-4", temperature=0.1)
#tempertature 0-1 arası değer alır. 1'e yaklaştıkça alaka oranı düşer.

#messages =[
#    SystemMessage(content="Translate the following from English to Spanish"),
#    HumanMessage(content="Hi!"),]
#Yukarıdaki işlem sadece bir dili çeviriyordu. Aşağıda bunun dinamik halini, birden fazla dile çevrilmişini yapalım.

system_prompt = "Translate the following message into {language}"
promt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),("user", "{text}")
])

parser = StrOutputParser()

chain = promt_template | model | parser

app = FastAPI(
    title="Simple Message Chain",
    description="Simple Message Chain Description",
    version="1.0",
    )

add_routes(app, chain, path="/chain")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
