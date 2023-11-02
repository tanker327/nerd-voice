# __all__ = ["gpt","message"]

from .gpt import OpenAIModel, GPT, Message, Role

GPT4 = GPT(OpenAIModel.GPT4)
GPT3_5_16k = GPT(OpenAIModel.GPT3_5_16k)
