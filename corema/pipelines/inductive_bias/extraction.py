"""Module for extracting inductive bias information from scientific papers."""

from typing import List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
import logging

from corema.storage import LocalStorage
from corema.utils.openai_api import get_structured_output
from corema.pipelines.common import PipelineBase, get_openai_pipeline_params

logger = logging.getLogger(__name__)


class InductiveBiasEvidence(BaseModel):
    """A piece of evidence for an inductive bias from a paper."""

    text_passage: str = Field(
        description="The relevant text passage discussing the bias"
    )
    bias_type: str = Field(
        description="Category of the inductive bias (e.g., 'Physics.ConservationLaws')"
    )
    summary: str = Field(
        description="Brief summary of how the bias is encoded in the model"
    )
    confidence: float = Field(
        description="Confidence score for this evidence", ge=0, le=1
    )


class InductiveBiasResult(BaseModel):
    """Collection of inductive biases found for a model."""

    biases: List[InductiveBiasEvidence] = Field(default_factory=list)


BIAS_EXTRACTION_SYSTEM_PROMPT = """You are an expert at identifying domain-specific inductive biases encoded into scientific foundation models. These are the prior assumptions, constraints, or domain knowledge explicitly built into the model architecture, loss function, or training process.

Focus on identifying SPECIFIC architectural choices and constraints that encode domain knowledge, such as:
1. Custom layer types or modifications that reflect domain structure
   - e.g. k-mer convolutions for genomics
   - e.g. physics-informed layers that solve PDEs
   - e.g. graph convolutions modified for molecular bonds
2. Architectural constraints that enforce domain rules
   - e.g. symmetry-preserving layers for physics
   - e.g. conservation law constraints in the architecture
   - e.g. anatomical priors in medical imaging
3. Domain-specific attention mechanisms
   - e.g. attention modified for protein contacts
   - e.g. cross-attention between modalities
   - e.g. spatial attention for geospatial data
4. Loss terms that encode domain knowledge
   - e.g. physics-based regularization
   - e.g. chemical validity constraints
   - e.g. biological feasibility terms

DO NOT include vague statements about the model being "designed for" a domain or "incorporating domain knowledge". Only extract specific architectural choices, constraints, or loss terms that explicitly encode domain knowledge.

For each bias found:
1. Extract the relevant passage that describes the specific architectural choice
2. Categorize the bias type (e.g., Physics.ConservationLaws, Biology.ProteinStructure)
3. Briefly summarize HOW it's encoded (the specific architectural mechanism)
4. Assign a confidence score (0-1) based on how explicitly the mechanism is described

Return your response as a JSON object with the following structure:
- biases: array of objects (max 10 items) containing evidence for inductive biases with:
  - text_passage: The relevant text passage describing the specific architectural choice
  - bias_type: Category of the inductive bias (e.g., 'Physics.ConservationLaws')
  - summary: Brief summary of the specific architectural mechanism that encodes this bias
  - confidence: Confidence score for this evidence (between 0 and 1)

Only include biases where you can identify the specific architectural mechanism (confidence >= 0.7)."""

BIAS_EXTRACTION_USER_PROMPT = """Analyze the following text from a scientific paper and identify any explicit inductive biases encoded into the foundation model. Focus on specific architectural choices, constraints, or loss terms that explicitly encode domain knowledge.

Text to analyze:
{text}

Return your response as a JSON object."""


class InductiveBiasPipeline(PipelineBase[InductiveBiasResult]):
    """Pipeline for analyzing inductive biases in foundation models."""

    PIPELINE_NAME = "inductive_bias"

    def __init__(self, storage: LocalStorage):
        super().__init__(self.PIPELINE_NAME, storage)

    def _flatten_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten biases into separate records."""
        flattened = []
        for bias in results["biases"]:
            record = {
                "bias_type": bias["bias_type"],
                "summary": bias["summary"],
                "confidence": bias["confidence"],
                "text_passage": bias["text_passage"],
            }
            flattened.append(record)
        return flattened

    def process_paper(
        self, project_id: str, project_name: str, paper_path: Path
    ) -> InductiveBiasResult:
        """Process a single paper to extract inductive biases."""
        # Read paper text
        with open(paper_path, "r") as f:
            text = f.read()

        # Get OpenAI parameters from pipeline config
        openai_params = get_openai_pipeline_params("inductive_bias")

        # Process text with LLM
        result = get_structured_output(
            system_prompt=BIAS_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=BIAS_EXTRACTION_USER_PROMPT.format(text=text),
            response_model=InductiveBiasResult,
            reduce_op=combine_biases,
            **openai_params,
        )

        return result


def combine_biases(
    a: InductiveBiasResult, b: InductiveBiasResult
) -> InductiveBiasResult:
    """Combine two InductiveBiasResult objects, merging their evidence lists."""
    # Combine evidence lists, using text_passage as key to avoid duplicates
    evidence_map = {e.text_passage: e for e in a.biases}
    for e in b.biases:
        if (
            e.text_passage not in evidence_map
            or e.confidence > evidence_map[e.text_passage].confidence
        ):
            evidence_map[e.text_passage] = e

    return InductiveBiasResult(biases=list(evidence_map.values()))
