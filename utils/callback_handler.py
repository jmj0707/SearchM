from langchain.callbacks.base import BaseCallbackHandler

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, update_fn):
        self.update_fn = update_fn

    def on_text(self, data):
        self.update_fn(data)