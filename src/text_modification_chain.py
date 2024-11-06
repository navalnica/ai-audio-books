from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel

from src.prompts import ModifyTextPrompt
from src.utils import GPTModels, get_chat_llm


class ModifiedTextOutput(BaseModel):
    text_raw: str
    text_modified: str


def modify_text_chain(llm_model: GPTModels):
    llm = get_chat_llm(llm_model=llm_model, temperature=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(ModifyTextPrompt.SYSTEM),
            HumanMessagePromptTemplate.from_template(ModifyTextPrompt.USER),
        ]
    )

    chain = RunnablePassthrough.assign(text_modified=prompt | llm | StrOutputParser()) | (
        lambda inputs: ModifiedTextOutput(
            text_raw=inputs["text"], text_modified=inputs["text_modified"]
        )
    )
    return chain
