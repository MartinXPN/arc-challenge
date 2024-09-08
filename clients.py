import os
import time
from abc import abstractmethod, ABC

import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from gradio_client import Client as GradioClient
from openai import OpenAI
from transformers import pipeline

load_dotenv()


class BaseClient(ABC):
    def __init__(self, api_key):
        self.api_key = api_key

    @abstractmethod
    def gen_responses(self, messages: list[str], responses: list[str], nb_responses: int, retries: int = 10) -> list[str]:
        ...


class AnthropicClient(BaseClient):
    def __init__(self, api_key: str | None = None):
        if api_key is None:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
        if api_key is None:
            raise ValueError('ANTHROPIC_API_KEY is required either as an environment variable or as an argument')
        super().__init__(api_key)
        self.client = Anthropic(api_key=self.api_key)

    def gen_response(self, messages: list[str], responses: list[str], retries: int) -> str:
        if retries == 0:
            return ''

        conversation = []
        for m, r in zip(messages, responses):
            conversation.append({'role': 'user',        'content': [{'type': 'text', 'text': m, 'cache_control': {'type': 'ephemeral'}}]})
            conversation.append({'role': 'assistant',   'content': [{'type': 'text', 'text': r}]})
        conversation.append({'role': 'user',            'content': [{'type': 'text', 'text': messages[-1], 'cache_control': {'type': 'ephemeral'}}]})

        try:
            message = self.client.messages.create(
                model='claude-3-5-sonnet-20240620',
                max_tokens=4096,
                extra_headers={'anthropic-beta': 'prompt-caching-2024-07-31'},
                messages=conversation,
            )
            print('Cache hits:', getattr(message.usage, 'cache_read_input_tokens', 0))
            return message.content[0].text
        except Exception as e:
            print(e, f'Retrying {retries - 1} more times after waiting...')
            time.sleep(1)
            return self.gen_response(messages, responses, retries - 1)


    def gen_responses(self, messages: list[str], responses: list[str], nb_responses: int, retries: int = 10) -> list[str]:
        assert len(messages) == len(responses) + 1, 'We should respond to the latest message'
        assert nb_responses >= 1, 'We need at least one response'
        assert retries >= 0, 'We need to retry at least once'

        all_responses = [self.gen_response(messages, responses, retries) for _ in range(nb_responses)]
        return all_responses


class OpenAIClient(BaseClient):
    def __init__(self, api_key: str | None = None):
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError('OPENAI_API_KEY is required either as an environment variable or as an argument')
        super().__init__(api_key)
        self.client = OpenAI(api_key=self.api_key)


    def gen_responses(self, messages: list[str], responses: list[str], nb_responses: int, retries: int = 10) -> list[str]:
        assert len(messages) == len(responses) + 1, 'We should respond to the latest message'
        assert nb_responses >= 1, 'We need at least one response'

        if retries <= 0:
            return [''] * nb_responses

        conversations = []
        for m, r in zip(messages, responses):
            conversations.append({'role': 'user', 'content': m})
            conversations.append({'role': 'assistant', 'content': r})
        conversations.append({'role': 'user', 'content': messages[-1]})

        try:
            response = self.client.chat.completions.create(
                model='gpt-4o-2024-08-06',
                messages=conversations,
                n=nb_responses,
            )
            results = [choice.message.content for choice in response.choices]
            return results
        except Exception as e:
            print(e, f'Retrying {retries - 1} more times after waiting...')
            time.sleep(1)
            return self.gen_responses(messages, responses, nb_responses, retries - 1)


class ReflectionLlamaClient(BaseClient):
    """
    https://huggingface.co/mattshumer/Reflection-Llama-3.1-70B
    """
    BASE_URL = 'https://api-inference.huggingface.co/models/mattshumer/Reflection-Llama-3.1-70B'

    def __init__(self, api_key: str | None = None, run_locally: bool = False):
        self.run_locally = run_locally
        if api_key is None:
            api_key = os.environ.get('HUGGINGFACE_KEY')

        super().__init__(api_key)
        if run_locally:
            self.pipe = pipeline('text-generation', model='mattshumer/Reflection-Llama-3.1-70B')
        else:
            self.client = GradioClient('gokaygokay/Reflection-70B-llamacpp')

    def gen_responses(self, messages: list[str], responses: list[str], nb_responses: int, retries: int = 10) -> list[str]:
        assert len(messages) == len(responses) + 1, 'We should respond to the latest message'
        assert nb_responses >= 1, 'We need at least one response'

        if retries <= 0:
            return [''] * nb_responses

        conversations = [
            {'role': 'system', 'content': 'You are a world-class AI system, capable of complex reasoning and reflection. Reason through the query inside <thinking> tags, and then provide your final response inside <output> tags. If you detect that you made a mistake in your reasoning at any point, correct yourself inside <reflection> tags.'},
        ]
        for m, r in zip(messages, responses):
            conversations.append({'role': 'user', 'content': m})
            conversations.append({'role': 'assistant', 'content': r})
        conversations.append({'role': 'user', 'content': messages[-1]})

        try:
            if self.run_locally:
                res = self.pipe(conversations, max_length=128000, temperature=0.7, top_p=0.95, num_return_sequences=nb_responses)
            else:
                response = requests.post(self.BASE_URL, headers={'Authorization': f'Bearer {self.api_key}'}, json={
                    'inputs': conversations,
                    'parameters': {
                        'max_length': 128000,
                        'temperature': 0.7,
                        'top_p': 0.95,
                        'num_return_sequences': nb_responses,
                    },
                })
                res = response.json()

            print('RES:', res)
            return [choice['generated_text'] for choice in res]
        except Exception as e:
            print(e, f'Retrying {retries - 1} more times after waiting...')
            time.sleep(1)
            return self.gen_responses(messages, responses, nb_responses, retries - 1)
