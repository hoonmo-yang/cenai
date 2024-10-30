from typing import Optional

import http.client
import json
import os
import requests

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from cenai_core.system import load_dotenv


class HyperCLOVAXChatModel(BaseChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        load_dotenv()

        self._api_url = os.environ["X_NCP_CLOVASTUDIO_MODEL_API_URL"]
        self._api_key = os.environ["X_NCP_CLOVASTUDIO_API_KEY"]
        self._apigw_api_key = os.environ["X_NCP_CLOVASTUDIO_APIGW_API_KEY"]

    def _generate(self,
                  messages: list[BaseMessage],
                  stop: Optional[list[str]] = None,
                  **kwargs
                  ) -> ChatResult:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._apigw_api_key,
        }

        stop = [] if stop is None else stop
        api_messages = self._convert_messages(messages)

        request_data = {
            "messages": api_messages,
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 2500,
            "temperature": 0.5,
            "repeatPenalty": 5.0,
            "stopBefore": stop,
            "includeAiFilters": False,
            "seed": 0,
        }

        response = requests.post(
            self._api_url,
            headers=headers,
            json=request_data,
        )

        response_json = response.json()

        content=(
            response_json["result"]["message"]["content"]
            if response_json["status"]["code"] == "20000" else
            "LLM Runtime error"
        )

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @staticmethod
    def _convert_messages(
        langchain_messages: list[BaseMessage]
    ) -> list[dict[str,str]]:
        api_messages = [
            {
                "role": "user" if message.type == "human" else
                        "assistant" if message.type == "ai" else
                        message.type,

                "content": message.content,
            }
            for message in langchain_messages
        ]
        return api_messages

    @property
    def _llm_type(self) -> str:
        return "hyperclovax-hcx-003"

    @property
    def model_name(self) -> str:
        return self._llm_type


class HyperCLOVAXEmbeddings(Embeddings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        load_dotenv()

        self._api_host = os.environ["X_NCP_CLOVASTUDIO_EMBED_API_HOST"]
        self._api_path = os.environ["X_NCP_CLOVASTUDIO_EMBED_API_PATH"]
        self._api_key = os.environ["X_NCP_CLOVASTUDIO_API_KEY"]
        self._apigw_api_key = os.environ["X_NCP_CLOVASTUDIO_APIGW_API_KEY"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeds = []
        for text in texts:
            embed = self.embed_query(text)
            if not embed:
                return []
            
            embeds.append(embed)

        return embeds

    def embed_query(self, query: str) -> Optional[list[float]]:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._apigw_api_key,
        }

        request_data = {
            "text": query
        }

        conn = http.client.HTTPSConnection(self._api_host)

        conn.request(
            method="POST",
            url=self._api_path,
            body=json.dumps(request_data),
            headers=headers,
        )

        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding="utf-8"))
        conn.close()

        return (
            result["result"]["embedding"]
            if result["status"]["code"] == "20000" else
            None
        )


class HyperCLOVAXSummarizer(BaseChatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        load_dotenv()

        self._api_host = os.environ["X_NCP_CLOVASTUDIO_SUMMARY_API_HOST"]
        self._api_path = os.environ["X_NCP_CLOVASTUDIO_SUMMARY_API_PATH"]
        self._api_key = os.environ["X_NCP_CLOVASTUDIO_API_KEY"]
        self._apigw_api_key = os.environ["X_NCP_CLOVASTUDIO_APIGW_API_KEY"]

    def _generate(self,
                  messages: list[BaseMessage],
                  **kwargs
                  ) -> ChatResult:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-NCP-CLOVASTUDIO-API-KEY": self._api_key,
            "X-NCP-APIGW-API-KEY": self._apigw_api_key,
        }

        api_messages = self._convert_messages(messages)

        request_data = {
            "texts": api_messages,
            "segMinSize": 300,
            "includeAiFilters": True,
            "autoSentenceSplitter": True,
            "segCount": -1,
            "segMaxSize": 1000,
        }

        conn = http.client.HTTPSConnection(self._api_host)

        conn.request(
            method="POST",
            url=self._api_path,
            body=json.dumps(request_data),
            headers=headers,
        )

        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding="utf-8"))
        conn.close()

        content=(
            result["result"]["message"]["text"]
            if result["status"]["code"] == "20000" else
            "LLM Runtime error"
        )

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @staticmethod
    def _convert_messages(
        langchain_messages: list[BaseMessage]
    ) -> list[str]:
        api_messages = [
            message.content
            for message in langchain_messages
            if message.type == "human"
        ]
        return api_messages

    @property
    def _llm_type(self) -> str:
        return "hyperclovax-summarizer"

    @property
    def model_name(self) -> str:
        return self._llm_type
