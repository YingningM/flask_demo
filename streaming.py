# This class would allow the chat interface to display the AI's responses as they are being generated, 
# token by token, providing a smooth and interactive user experience.

from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        print(f"Received token: {token}")
        self.text += token
        self.container.markdown(self.text)
