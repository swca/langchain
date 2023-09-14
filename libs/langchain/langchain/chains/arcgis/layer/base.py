from __future__ import annotations

from langchain import LLMChain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from pydantic import Extra
from typing import Any, Dict, List, Optional, Sequence

from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseDocumentTransformer
from langchain.schema import Document

from langchain.chains.arcgis.layer.prompts import PROMPT


class ArcGISLayerSummaryChain(LLMChain):
    """
    Represents a custom chain to generate overall layer summaries for geospatial data layers using an LLM.

    Attributes:
        llm (BaseChatModel): The Large Language Model used for text generation.
        prompt (ChatPromptTemplate): The template to guide the LLM's responses.
        output_key (str): The key used to extract the output from the LLM's response.
    """

    llm: BaseChatModel
    prompt: ChatPromptTemplate = PROMPT
    output_key: str = "text"

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        prompt_value = self.prompt.format_prompt(**inputs)

        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # if run_manager:
        #     run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        prompt_value = self.prompt.format_prompt(**inputs)

        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # if run_manager:
        #     await run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "ArcGISLayerSummaryChain"
