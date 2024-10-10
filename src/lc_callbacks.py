import typing as t

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.messages import BaseMessage

from src.config import logger


class LCMessageLoggerAsync(AsyncCallbackHandler):
    """Custom callback to make Langchain logs easy to read"""

    @staticmethod
    def langchain_msg_2_role_content(msg: BaseMessage):
        return {"role": msg.type, "content": msg.content}

    def __init__(self, log_raw_llm_response=True):
        super().__init__()
        self._log_raw_llm_response = log_raw_llm_response

    def on_chat_model_start(
        self,
        serialized: dict[str, t.Any],
        messages: list[list[BaseMessage]],
        **kwargs: t.Any,
    ) -> t.Any:
        """Run when Chat Model starts running."""
        if len(messages) != 1:
            raise ValueError(f'expected "messages" to have len 1, got: {len(messages)}')

        kwargs = serialized["kwargs"]
        model_name = kwargs.get("model_name")
        if not model_name:
            model_name = kwargs.get("deployment_name")
        if not model_name:
            model_name = "<failed to determine LLM>"

        msgs_list = list(map(self.langchain_msg_2_role_content, messages[0]))
        msgs_str = "\n".join(map(str, msgs_list))

        logger.info(f"call to {model_name} with {len(msgs_list)} messages:\n{msgs_str}")

    def on_llm_end(self, response: LLMResult, **kwargs: t.Any) -> t.Any:
        """Run when LLM ends running."""
        generations = response.generations
        if len(generations) != 1:
            raise ValueError(
                f'expected "generations" to have len 1, got: {len(generations)}'
            )
        if len(generations[0]) != 1:
            raise ValueError(
                f'expected "generations[0]" to have len 1, got: {len(generations[0])}'
            )

        if self._log_raw_llm_response is True:
            gen: ChatGeneration = generations[0][0]
            ai_msg = gen.message
            logger.info(f'raw LLM response: "{ai_msg.content}"')
