"""The seam between this application and whichever LLM is behind it.

Both Gemini and Ollama reduce to the same operation once you look past the
SDK differences:

    Gemini:  response_format={"schema": Model.model_json_schema()}
             -> Model.model_validate_json(interaction.output_text)

    Ollama:  format=Model.model_json_schema()
             -> Model.model_validate_json(response.message.content)

Same input (a prompt plus a JSON Schema), same output (a JSON string). So
that is the whole interface. Everything else -- writing the prompt, running
Pydantic validation, deciding whether a failure is worth retrying -- is
provider-independent and lives exactly once, in workers/extract.py.

Getting this boundary in the right place is the design decision. Put it too
high (a `extract_profile()` method on each provider) and the prompt and the
validation logic get duplicated and drift. Put it too low (a `post_json()`
method) and every provider's auth and request shape leaks upward.
"""

from typing import Any, Protocol, runtime_checkable


class LLMError(Exception):
    """Base for every failure originating from the LLM provider."""


class LLMTransientError(LLMError):
    """Worth retrying: timeout, rate limit, 5xx, connection reset.

    In M5 this is what earns a job an exponential-backoff retry.
    """


class LLMPermanentError(LLMError):
    """Never worth retrying: bad API key, unknown model, malformed request.

    Retrying burns a worker slot on something that cannot succeed. This is
    the same permanent/transient split the spec describes in section 6.4 --
    classify by what the caller should do, not by where it happened.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can return schema-constrained JSON.

    A Protocol, not an ABC: providers do not inherit from this, they just
    happen to match its shape. That means a test can pass a plain object with
    one method and never touch the network -- structural typing, checked by
    the type checker rather than enforced by an import.
    """

    name: str

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> str:
        """Return a JSON string conforming to `schema`.

        Args:
            prompt: The full instruction, including the source document.
            schema: A JSON Schema dict, from `SomeModel.model_json_schema()`.

        Returns:
            The model's raw response text, expected to be JSON. Callers
            validate it -- providers must not.

        Raises:
            LLMTransientError: The call failed in a way worth retrying.
            LLMPermanentError: The call failed in a way that never will.
        """
        ...
