from enum import Enum
import io
from dotenv import load_dotenv
import openai
import requests
import simplejson
import time
from enum import Enum
import os

from utils.logger import LOG


# Load .env file into environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MAX_TOKENS = 8192 # MAximum number of tokens for GPT-4



class Role(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class Message():
    
    def __init__(self,role: Role, content: str):
        self.role = role
        self.content = content
        
    def to_dict(self):
        return {
            "role": self.role.value,
            "content": self.content
        }
    @staticmethod
    def from_tuple(t):
        return Message(t[0], t[1])
    
    def __repr__(self) -> str:
        return f"{self.role.value}: {self.content}"
    
    @staticmethod
    def from_tuples(tuples) -> list:
        return [Message.from_tuple(t) for t in tuples]
    
    @staticmethod
    def to_dict_list(messages: list):
        return [m.to_dict() for m in messages]
    
 
class OpenAIModel(Enum):
    GPT4 = "gpt-4"
    GPT3_5_16k = "gpt-3.5-turbo-16k"
    GPT3_5_TURBO = "gpt-3.5-turbo"
    WHISPER1 = "whisper-1"


class GPT():
    def __init__(self, model: OpenAIModel = OpenAIModel.GPT4):
        self.model = model

    def ask(self, messages: [Message], temperature=0.5, max_tokens=MAX_TOKENS):
        attempts = 0
        messages_dict = Message.to_dict_list(messages)
        while attempts < 3:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model.value,
                    messages=messages_dict,
                    max_tokens=max_tokens,
                    temperature=temperature  # between 0 and 2
                )
                return response['choices'][0]['message']

            except openai.error.RateLimitError:
                attempts += 1
                if attempts < 3:
                    LOG.warning(
                        "Rate limit reached. Waiting for 60 seconds before retrying.")
                    time.sleep(60)
                else:
                    raise Exception(
                        "Rate limit reached. Maximum attempts exceeded.")
            except requests.exceptions.RequestException as e:
                raise Exception(f"Request exception: {e}")
            except requests.exceptions.Timeout as e:
                raise Exception(f"Request timeout: {e}")
            except simplejson.errors.JSONDecodeError as e:
                raise Exception("Error: response is not valid JSON format.")
            except Exception as e:
                raise Exception(f"An unknown error occurred: {e}")
        return "", False
    
    def transcribe(self,audio_file_path:str, prompt:str=None):
        LOG.debug("Start sending request to OpenAI API.")
        audio_file = open(audio_file_path, "rb")
        transcript_response = openai.Audio.transcribe(OpenAIModel.WHISPER1.value, audio_file, prompt=prompt)
        LOG.debug(transcript_response)
        transcript = transcript_response['text']
        return transcript 
    
    def load_models(self):
        result = openai.Model.list()
        return result


# if __name__ == "__main__":
#     # gpt = GPT(OpenAIModel.GPT4)
#     # result = gpt.transcribe("/var/folders/_2/9xt4t4fx2578_2pc0ymcw65m0000gn/T/temp.mp3")
#     # print(result)
#     audio_file = open("/var/folders/_2/9xt4t4fx2578_2pc0ymcw65m0000gn/T/temp.mp3", "rb")
#     transcript_response = openai.Audio.transcribe(OpenAIModel.WHISPER1.value, audio_file)
#     print(transcript_response)
    
    


   
# if __name__ == "__main__":
#     m = Message(Role.USER, "hello")
#     print(m.to_dict())
#     print(Message.to_dict_list([m]))
    
#     n = Message.from_tuple((Role.USER, "hello"))
#     print(n.to_dict())
#     print(Message.to_dict_list([m,n]))
    
#     print(Message.from_tuple((Role.USER, "hello")))
#     print(Message.from_tuples([(Role.USER, "hello"), (Role.SYSTEM, "hi")]))
