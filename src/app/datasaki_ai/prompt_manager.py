from pydantic import BaseModel, ValidationError, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from abc import ABC, abstractmethod
from typing import Dict, Any,Union, List
from app.datasaki_ai.prompt_templates import prompts
import json

class PromptOutputSchema(ABC):
    @abstractmethod
    def set_parser(self, schema:object, parser_type:str = None) -> None:
        """
        Parse the output into the required format.
        :param schema: Object of schema.
        :param parser_type: Type of Parser
        :return: Parsed output (string, JSON, or list).
        """
        pass

    @abstractmethod
    def get_schema(self) -> object:
        """
        Parse the output into the required format.
        :return: Parsed output (string, JSON, or list).
        """
        pass

    @abstractmethod
    def get_parser(self) -> object:
        """
        Parse the output into the required format.
        :return: Parsed output (string, JSON, or list).
        """
        pass

class PydanticSchemaOutput(PromptOutputSchema):
    def __init__(self):
        self.schema = None
        self.parser = None
        self.schema_prompt = None

    def set_parser(self, schema: BaseModel, parser_type:str = None) -> None:
        self.schema = schema
        if parser_type == 'with_structured_output':
            self.parser = 'with_structured_output'
        else:
            self.parser = PydanticOutputParser(pydantic_object=schema)
            self.schema_prompt = self.parser.get_format_instructions()

    def get_schema(self) -> object:
         return self.schema

    def get_parser(self) -> object:
        return self.parser

    def get_schema_prompt(self):
        return self.schema_prompt

class ResponseSchemaOutput(PromptOutputSchema):
    def __init__(self):
        self.schema = None
        self.parser = None
        self.schema_prompt = None

    def set_parser(self, schema: List[ResponseSchema], parser_type: str = None) -> None:
        self.schema = schema
        if parser_type == 'with_structured_output':
            self.parser = 'with_structured_output'
        else:
            self.parser = StructuredOutputParser.from_response_schemas(schema)
            self.schema_prompt = self.parser.get_format_instructions()

    def get_schema(self) -> object:
        return self.schema

    def get_parser(self) -> object:
        return self.parser

    def get_schema_prompt(self):
        return self.schema_prompt

class JsonSchemaOutput(PromptOutputSchema):
    def __init__(self):
        self.schema = None
        self.parser = None
        self.schema_prompt = None

    def set_parser(self, schema: dict, parser_type: str = 'with_structured_output') -> None:
        self.schema = {
            "title": "Response",
            "description": "A simple answer to the cleaning question",
            "properties": schema,
            "required": list(schema.keys())
        }
        self.parser = 'with_structured_output'

    def get_schema(self) -> object:
        return self.schema

    def get_parser(self) -> object:
        return self.parser

    def get_schema_prompt(self):
        return self.schema_prompt

class PromptInput(BaseModel):
    system_prompt: str
    user_query: str
    context: str = None
    prompt_args: Dict[str,str] = None

class PromptTemplate:
    def __init__(self, title: str, description:str, template: list[tuple[str,str]],
                 output_parser: PromptOutputSchema,input_var=Dict[str,str]):
        """
        Initialize the prompt template with a title, template string, output type, and output parser.
        :param title: The title of the prompt.
        :param description: The description of the prompt.
        :param template: The string template for the prompt.
        :param prompt_input: The OutputParser to process the result.
        :param output_parser: The OutputParser to process the result.
        """
        self.title = title
        self.template = template
        self.description = description
        self.output_schema = output_parser.get_schema()
        self.output_parser = output_parser.get_parser()

    def create_prompt(self,**input_data):
        """
        Create a final prompt by filling in placeholders.
        :param system_prompt: The query from the user to be inserted into the prompt.
        :param prompt_input: The query from the user to be inserted into the prompt.
        :param context: The optional context for the LLM to use.
        :return: The fully generated prompt string.
        """
        chat = ChatPromptTemplate.from_messages(self.template)
        return chat


class PromptDB:
    prompts = {}
    def __init__(self,prompts:dict={}):
        for title,prompt in prompts.items():
            self.add_prompt(title,prompt)

    def add_prompt(self,title,template):
        parser = PydanticSchemaOutput()
        parser.set_parser(template["prompt_parser"])
        prompt_template = PromptTemplate(
            title = title,
            description=template["description"],
            template=template["template"],
            output_parser = parser
        )
        self.prompts[title] = { "prompt_template":prompt_template}

    def get_prompt(self,title:str) -> PromptTemplate:
        return self.prompts[title]["prompt_template"]

    # def get_system_message(self,title:str) -> PromptTemplate:
    #     return self.prompts[title]["system_message"]

prompts_db = PromptDB(prompts)





