from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

#Define the model
llm=HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.18-Chat-v1.0",
    #"google/gemma-2-2b-it"
    task="text-generation"
)
#This model won't give with_structured_output function
#so we are using output parsers here

model=ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name='fact_1',description="Fact1"),
    ResponseSchema(name='fact_2',description="Fact2"),
    ResponseSchema(name='fact_3',description="Fact3"),
    ResponseSchema(name='fact_4',description="Fact4"),
    ResponseSchema(name='fact_5',description="Fact5")
]

parser=StructuredOutputParser.from_response_schema(schema)

template=PromptTemplate(
    template="Give 5 fact about {topic}.\n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain=template|model|parser
res=chain.invoke({'topic':'blackhole'})
print(res)

#Diadvantage:
#Data Validation
