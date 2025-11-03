from typing import Any

from udspy.signature import Signature


class AdapterParseError(Exception):
    """Exception raised when adapter cannot parse the LM response."""

    def __init__(
        self,
        adapter_name: str,
        signature: Signature | None,
        lm_response: str,
        message: str | None = None,
        parsed_result: dict[str, Any] | None = None,
    ):
        self.adapter_name = adapter_name
        self.signature = signature
        self.lm_response = lm_response
        self.parsed_result = parsed_result

        message = f"{message}\n\n" if message else ""
        base_message = (
            f"{message}"
            f"Adapter {adapter_name} failed to parse the LM response. \n\n"
            f"LM Response: {lm_response} \n\n"
        )

        if signature is not None:
            base_message += f"Expected to find output fields in the LM response: [{', '.join(signature.get_output_fields().keys())}] \n\n"

        if parsed_result is not None:
            base_message += f"Actual output fields parsed from the LM response: [{', '.join(parsed_result.keys())}] \n\n"

        super().__init__(base_message)
