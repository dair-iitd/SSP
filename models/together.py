import together
import logging
import tiktoken
import time
import backoff
import os

def backoff_hdlr(details):
    print ("Backing off {wait:0.1f} seconds after {tries} tries "
           "calling function {target} with args {args} and kwargs "
           "{kwargs}".format(**details))
    print(details)


@backoff.on_exception(backoff.expo, Exception, max_tries=4, on_backoff=backoff_hdlr)
@backoff.on_predicate(backoff.expo, lambda response: response is None)
def get_response(prompt, args):
    response = together.Complete.create(**{'prompt': prompt, **args})
    return response

def setup_api_key():

    # openai.api_key = os.getenv("OPENAI_API_KEY")
    together.api_key = os.getenv("TOGETHER_API_KEY")

class CompletionModel:

    DEFAULT_ARGS = {
        'model': 'meta-llama/llama-2-70b-hf',
        'max_tokens': 1024,
        'stop': ['```', 'Premise'],
        'temperature': 0.0,
        'top_k': 60,
        'top_p': 1.0,
        'repetition_penalty': 1.1
    }

    def __init__(self, default_args=None):

        if default_args:
            self.default_args = default_args
        else:
            self.default_args = CompletionModel.DEFAULT_ARGS

        self.logger = logging.getLogger('Together')

    def complete(self, prompt, args=None):

        response = None

        if not args:
            args = self.default_args

        self.logger.info(f'Giving the following prompt:{prompt}')
        try:
            response = get_response(prompt, args)

        except Exception as e:
            print(e)
        finally:
            self.logger.info('Received the following response:')
            self.logger.info(response)
            if not response:
                return ""
            return response['output']['choices'][0]['text'].strip()

    def cleanup(self):
        pass

