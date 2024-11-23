
from pydantic import BaseModel, Field, field_validator

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


class Hi(BaseModel):
    class Config:
        title = "title for Hi"
        descripiton="just example"

    message: str = Field(description="simple message")


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
    recursive: Hi = Field(descripnt="another pydantic class")

    @field_validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field

joke_query = "Tell me a joke."


parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

x = prompt.invoke({
    "query": joke_query,
})

print(x)