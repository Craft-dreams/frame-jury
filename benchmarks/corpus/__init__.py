"""Build and validate the frame-jury benchmark corpus."""

from .schema import CASE_SCHEMA_VERSION, CaseValidationError, validate_case

__all__ = ["CASE_SCHEMA_VERSION", "CaseValidationError", "validate_case"]
