import os
import openai
from typing import List, Dict, Any


class OpenAI:
    def __init__(self,
                 model: str = "gpt-4",
                 temperature: float = 0.0,
                 seed: int = 42,
                 *args,
                 **kwargs):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.args = args
        self.kwargs = kwargs

    def get_response(self, chat_history: List[Dict[str, Any]]):
        """
        Generates a response from the given input.

        Args:
            chat_history (List[Dict[str, Any]]): A list of message dictionaries to be processed by the model.

        Returns:
            The response from the ChatCompletion API.
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
            messages=chat_history,
            *self.args,
            **self.kwargs
        )
        return response

    def get_response_message(self, chat_history: List[Dict[str, Any]]) -> str:
        """
        Generates a chatbot message from the given input.

        Args:
            chat_history (List[Dict[str, Any]]): A list of message dictionaries to be processed by the model.

        Returns:
            str: The message from the chatbot.
        """
        return self.get_response(chat_history).choices[0].message["content"]
