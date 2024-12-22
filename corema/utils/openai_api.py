"""Utilities for interacting with the OpenAI API."""

import logging
from functools import reduce
from typing import TypeVar, Type, List, Iterator, Callable, cast
from pydantic import BaseModel
from openai import OpenAI
import tiktoken
from ..config import get_config
from .openai_models import get_model_info, DEFAULT_MODEL

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in a text string.

    Args:
        text: The text to count tokens for
        model: The model to use for counting tokens

    Returns:
        The number of tokens in the text
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def count_message_tokens(messages: List[dict], model: str) -> int:
    """Count the number of tokens in a list of chat messages.

    Args:
        messages: The messages to count tokens for
        model: The model to use for counting tokens

    Returns:
        The number of tokens in the messages

    Note:
        This implements the token counting logic from:
        https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    encoding = tiktoken.encoding_for_model(model)

    # Every message follows <|start|>{role/name}\n{content}<|end|>\n
    tokens_per_message = 3

    # If there's a name, the role is omitted
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name

    # Every reply is primed with <|start|>assistant<|end|>
    num_tokens += 3

    return num_tokens


def chunk_text(
    text: str, model: str, max_chunk_tokens: int, overlap_tokens: int = 100
) -> Iterator[str]:
    """Split text into chunks that fit within token limits.

    Args:
        text: The text to chunk
        model: The model to use for token counting
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks for context

    Yields:
        Text chunks that fit within token limits

    Note:
        The overlap helps maintain context between chunks. Adjust overlap_tokens
        based on your needs - larger values help maintain more context but
        increase total tokens processed.
    """
    if not text:
        return

    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    if len(tokens) <= max_chunk_tokens:
        yield text
        return

    # Ensure overlap is not larger than chunk size
    overlap_tokens = min(overlap_tokens, max_chunk_tokens - 1)

    current_index = 0
    while current_index < len(tokens):
        # Get chunk of tokens
        chunk_end = min(current_index + max_chunk_tokens, len(tokens))
        chunk_tokens = tokens[current_index:chunk_end]

        # Decode chunk back to text
        chunk_text = encoding.decode(chunk_tokens)
        yield chunk_text

        # If we've reached the end, break
        if chunk_end == len(tokens):
            break

        # Move to next chunk with overlap, ensuring we make forward progress
        next_index = current_index + max_chunk_tokens - overlap_tokens
        # If we wouldn't make progress, force moving forward
        if next_index <= current_index:
            next_index = current_index + 1
        current_index = next_index


def _get_structured_output(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    model: str = DEFAULT_MODEL,
) -> T:
    """Get a structured output from the OpenAI API using a Pydantic model.

    Args:
        system_prompt: The system prompt to send to the API
        user_prompt: The user prompt to send to the API
        response_model: The Pydantic model class to validate and parse the response
        temperature: The temperature to use for sampling (default: 0.0)
        max_tokens: The maximum number of tokens to generate (default: None, uses model's default)
        model: The model to use (default: gpt-4-turbo-preview)

    Returns:
        The parsed and validated response as an instance of response_model

    Raises:
        ValueError: If the OpenAI API key is not configured or if inputs exceed model limits
        RuntimeError: If the API request fails
    """
    api_key = get_config().get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not configured")

    # Get model information
    model_info = get_model_info(model)
    client = OpenAI(api_key=api_key)

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        input_tokens = count_message_tokens(messages, model)
        max_output = max_tokens or model_info.max_output_tokens

        if input_tokens > model_info.context_window:
            raise ValueError(
                f"Input tokens ({input_tokens}) exceeds model context window ({model_info.context_window})"
            )

        if max_output > model_info.max_output_tokens:
            raise ValueError(
                f"Requested max_tokens ({max_output}) exceeds model maximum ({model_info.max_output_tokens})"
            )

        logger.info(
            f"Sending OpenAI request: model={model}, input_tokens={input_tokens}, "
            f"max_output_tokens={max_output}, response_model={response_model.__name__}"
        )

        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=temperature,
            max_tokens=max_output,
        )

        logger.info(
            f"OpenAI request tokens - Input: {input_tokens}, Completion: {response.usage.completion_tokens}"
        )
        logger.debug(f"Response: {response}")

        try:
            return cast(
                T,
                response_model.model_validate_json(response.choices[0].message.content),
            )
        except Exception as e:
            logger.error(f"JSON validation error: {str(e)}")
            logger.error("Raw response content:")
            logger.error(response.choices[0].message.content)
            raise

    except Exception as e:
        logger.exception("Error during OpenAI API request")
        raise RuntimeError(f"OpenAI API request failed: {str(e)}") from e


def get_structured_output(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    reduce_op: Callable[[T, T], T] | None = None,
    *,
    chunk_size: int | None = None,
    overlap_tokens: int = 100,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    model: str = DEFAULT_MODEL,
) -> T:
    """Get a structured output from the OpenAI API by processing text in chunks and combining results.

    Args:
        system_prompt: The system prompt to send to the API
        user_prompt: The user prompt to send to the API
        response_model: The Pydantic model class to validate and parse the response
        reduce_op: Function that takes two instances of response_model and combines them.
                  Must be provided if input requires chunking.
        chunk_size: Size of each chunk in tokens
        overlap_tokens: Number of tokens to overlap between chunks for context
        temperature: The temperature to use for sampling (default: 0.0)
        max_tokens: The maximum number of tokens to generate (default: None)
        model: The model to use (default: gpt-4-turbo-preview)

    Returns:
        Combined results from all chunks as an instance of response_model

    Raises:
        ValueError: If input requires chunking but no reduce_op is provided
    """
    if chunk_size is None:
        # If no chunk size specified, just make one request
        return _get_structured_output(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

    # Get chunks
    chunks = list(chunk_text(user_prompt, model, chunk_size, overlap_tokens))

    if not chunks:
        raise ValueError("No text chunks generated")

    # Process all chunks and collect results
    all_results = [
        _get_structured_output(
            system_prompt=system_prompt,
            user_prompt=chunk,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        for chunk in chunks
    ]

    # If only one result, return it
    if len(all_results) == 1:
        return all_results[0]

    # Multiple chunks but no reducer provided
    if reduce_op is None:
        raise ValueError(
            "Input required chunking but no reduce_op was provided. "
            "Provide a reduce_op function to combine results from multiple chunks."
        )

    # Reduce all results using the reduce operator
    return reduce(reduce_op, all_results[1:], all_results[0])
