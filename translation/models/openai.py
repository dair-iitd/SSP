import openai
import logging
import tiktoken
import time
import backoff
import os, pdb

def backoff_hdlr(details):
    print ("Backing off {wait:0.1f} seconds after {tries} tries "
           "calling function {target} with args {args} and kwargs "
           "{kwargs}".format(**details))
    print(details)

from openai import OpenAI
client = OpenAI(api_key="")
@backoff.on_exception(backoff.expo, Exception, max_tries=10, on_backoff=backoff_hdlr, max_time=600)
@backoff.on_predicate(backoff.expo, lambda response: response is None)
def get_response(prompt, args, chat=False):
    #pdb.set_trace()
    if chat:
        response = client.chat.completions.create(**{'messages': prompt, **args})
    else:
        response = client.completions.create(**{'prompt': prompt, **args})

    return response

def setup_api_key(api_key):

    openai.api_key = os.getenv("OPENAI_API_KEY")
    #openai.api_key = os.getenv("AZURE_OPENAI_KEY")
    #openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    #openai.api_type = 'azure'
    #openai.api_version = '2023-05-15'

class ChatGPT:

    DEFAULT_ARGS = {
        'model': 'gpt-3.5-turbo',
        'max_tokens': 1024,
        'stop': None,
        'temperature': 0.0,
        'timeout': 20
    }

    def __init__(self, default_args=None):

        self.encoding = tiktoken.encoding_for_model("gpt-4")

        if default_args:
            self.default_args = default_args
        else:
            self.default_args = ChatGPT.DEFAULT_ARGS

        self.logger = logging.getLogger('GPT')

        self.chat_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    def complete(self, prompt, args=None):

        response = None

        if not args:
            args = self.default_args

        self.chat_history.append({
            'role': 'user',
            'content': prompt
        })
        self.logger.info(f'Giving the following prompt:{prompt}')
        n_tokens = len(self.encoding.encode(prompt))
        #if n_tokens + args['max_tokens'] > 8100:
        #    args['max_tokens'] = abs(8100 - n_tokens)
        if args['model'] == 'gpt-35-turbo':
            if n_tokens+args['max_tokens'] > 4096:
                print("WARN: prompt exceeds gpt-35-turbo's prompt length. Truncating")
                self.chat_history = self.chat_history[1:]
                self.chat_history[0]['content'] = self.chat_history[0]['content'][:246] + self.chat_history[0]['content'][1000:]
        try:
            response = get_response(self.chat_history, args, chat=True)

        except Exception as e:
            # TODO exponential backoff and termination
            print(e)
        finally:
            self.logger.info('Received the following response:')
            self.logger.info(response)
            #pdb.set_trace()
            if not response:
                return ""
            #pdb.set_trace()
            if 'content' not in response.choices[0].message.to_dict():
                print("An error occured, could not get response (probably content filter")
                return ""
            #pdb.set_trace()
            return response.choices[0].message.content.strip()

    def cleanup(self):

        self.chat_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]


