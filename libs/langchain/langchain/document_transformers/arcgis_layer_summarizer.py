from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from langchain import LLMChain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseDocumentTransformer
from langchain.schema import Document
from pydantic import Extra

from AI_GIS_language.summarize.row import RowSummarizer

from libs.langchain.langchain.chains.arcgis.layer.prompts import PROMPT


class LayerSummarizer(BaseDocumentTransformer):
    """
    A class to produce overall layer summaries for geospatial data layers using a Large Language Model.

    Attributes:
        chain (ArcGISLayerSummaryChain): The chain of LLM models used for generating summaries.
    """

    def __init__(
        self,
        chain: ArcGISLayerSummaryChain,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the LayerSummarizer class.

        Args:
            chain (ArcGISLayerSummaryChain): The chain of LLM models.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.chain = chain

    @staticmethod
    def summaries_str(docs: Sequence[Document]) -> str:
        """
        Converts a list of summaries into a formatted string.

        Args:
            docs (list[str]): A list of Documents.

        Returns:
            str: A formatted string containing the summaries.
        """
        summaries = [doc.metadata["summary"] for doc in docs]
        sub_sums = "\n".join(
            f"<row_summary>\n{summary}\n</row_summary>" for summary in summaries
        )
        return f"<layer_summary>\n{sub_sums}\n</layer_summary>"

    def summarize(self, docs: list[Document]) -> str:
        """
        Produces bullet-point summaries for given geospatial data rows.

        Args:
            docs (list[Document]): List of documents representing geospatial data rows.

        Returns:
            list[str]: A list of bullet-point summaries.
        """
        input = self.summaries_str(docs)
        output = self.chain.run(input)
        return output["text"]

    async def asummarize(self, docs: list[Document]) -> str:
        """
        Asynchronously produces bullet-point summaries for given geospatial data rows.

        Args:
            docs (list[Document]): List of documents representing geospatial data rows.

        Returns:
            list[str]: A list of bullet-point summaries.
        """
        input = self.summaries_str(docs)
        output = await self.chain.arun(input)
        return output["text"]

    def transform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Document:
        summary = self.summarize(list(documents))
        return Document(
            page_content=summary,
            metadata={"input_docs": documents}
        )

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Document:
        summary = await self.asummarize(list(documents))
        return Document(
            page_content=summary,
            metadata={"input_docs": documents}
        )
