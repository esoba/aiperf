# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Declarative cross-field validation via annotations.

This module provides constraint annotations that can be used with Pydantic models
to enforce cross-field validation rules declaratively.

Example:
    from aiperf.config.constraints import (
        ConstraintsMixin, ConflictsWith, AtLeastOne, AllOrNone,
        Requires, GreaterThan, RequiredIf,
    )

    class MyConfig(BaseModel, ConstraintsMixin):
        option_a: Annotated[str | None, ConflictsWith("opts")] = None
        option_b: Annotated[str | None, ConflictsWith("opts")] = None
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    get_type_hints,
)

from pydantic import BaseModel, model_validator

from aiperf.common.enums import CaseInsensitiveStrEnum

if TYPE_CHECKING:
    from typing_extensions import Self


# =============================================================================
# ENUMS
# =============================================================================


class ConstraintType(CaseInsensitiveStrEnum):
    """Types of constraints that can be applied to fields."""

    # Group-based constraints (fields share a group identifier)
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    TRUTHY_MUTUALLY_EXCLUSIVE = "truthy_mutually_exclusive"
    AT_LEAST_ONE = "at_least_one"
    AT_LEAST_ONE_TRUTHY = "at_least_one_truthy"
    EXACTLY_ONE = "exactly_one"
    EXACTLY_ONE_TRUTHY = "exactly_one_truthy"
    ALL_OR_NONE = "all_or_none"
    MIN_FIELDS_SET = "min_fields_set"
    MAX_FIELDS_SET = "max_fields_set"
    FIELD_COUNT_RANGE = "field_count_range"
    CONDITIONAL_AT_LEAST_ONE = "conditional_at_least_one"
    GROUP_MUTUALLY_EXCLUSIVE = "group_mutually_exclusive"

    # Field-specific constraints
    CONFLICTS_WITH = "conflicts_with"
    REQUIRES_FIELD = "requires_field"
    REQUIRES_FIELD_WHEN_TRUTHY = "requires_field_when_truthy"
    REQUIRES_ALL_OF = "requires_all_of"
    REQUIRES_ANY_OF = "requires_any_of"
    FORBIDS_FIELD = "forbids_field"
    FORBIDS_FIELD_WHEN_TRUTHY = "forbids_field_when_truthy"

    # Comparison constraints
    COMPARE_TO = "compare_to"
    MATCHES_FIELD = "matches_field"
    NOT_EQUAL_TO_FIELD = "not_equal_to_field"

    # Conditional constraints
    CONDITIONAL_REQUIREMENT = "conditional_requirement"
    CONDITIONAL_REQUIREMENT_IN = "conditional_requirement_in"
    FORBIDDEN_WITH = "forbidden_with"
    FORBIDDEN_WITH_IN = "forbidden_with_in"
    ALLOWED_ONLY_WITH = "allowed_only_with"


class ComparisonOp(CaseInsensitiveStrEnum):
    """Comparison operators for CompareTo constraint."""

    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="


_OP_DESCRIPTIONS: dict[ComparisonOp, str] = {
    ComparisonOp.GT: "greater than",
    ComparisonOp.GE: "greater than or equal to",
    ComparisonOp.LT: "less than",
    ComparisonOp.LE: "less than or equal to",
    ComparisonOp.EQ: "equal to",
    ComparisonOp.NE: "not equal to",
}


@dataclass(frozen=True)
class ConstraintGroup:
    """Represents a collected constraint with its associated fields.

    The constraint instance itself contains all metadata needed for validation.
    """

    constraint_type: ConstraintType
    group_name: str
    fields: tuple[str, ...]
    constraint: BaseConstraint


# =============================================================================
# CONSTRAINT VIOLATION AND ERROR
# =============================================================================


@dataclass(frozen=True)
class ConstraintViolation:
    """Details about a single constraint validation failure.

    Attributes:
        constraint_class: The constraint class that failed (e.g., ConflictsWith).
        constraint_type: The constraint type enum value.
        message: Human-readable error message.
        fields: The field(s) involved in the violation.
        group_name: The group name for group-based constraints, or field name for others.
    """

    constraint_class: type[BaseConstraint]
    constraint_type: ConstraintType
    message: str
    fields: tuple[str, ...]
    group_name: str


@dataclass
class ValidationContext:
    """Context passed to constraint validate methods.

    Consolidates all inputs needed for validation, making the validate()
    signature cleaner and easier to extend in the future.

    Attributes:
        model: The Pydantic model instance being validated.
        fields: The field(s) this constraint applies to.
        display_names: Mapping of field names to CLI display names.
        cli_to_field: Mapping of CLI names to field names (for reverse lookup).
        all_groups: All constraint groups (needed by GroupConflictsWith).
    """

    model: Any
    fields: tuple[str, ...]
    display_names: dict[str, str]
    cli_to_field: dict[str, str]
    all_groups: dict[ConstraintType, list[ConstraintGroup]]

    @property
    def source_field(self) -> str:
        """The primary field this constraint is attached to."""
        return self.fields[0]

    def resolve_field(self, field_or_cli: str) -> str:
        """Resolve a field reference, which may contain CLI arg names.

        Handles:
            - Simple field names: 'request_count' -> 'request_count'
            - CLI argument names: '--request-count' -> 'request_count'
            - Dot-paths with CLI names: 'inner.--max-value' -> 'inner.max_val'
            - Mixed paths: 'config.--timeout.value' -> 'config.timeout.value'

        Args:
            field_or_cli: Field path that may contain CLI arg names at any segment.

        Returns:
            The resolved field path with all CLI names converted to field names.

        Raises:
            ValueError: If a CLI arg name is not found in the mapping.
        """
        # Fast path: no CLI args in the path
        if "-" not in field_or_cli:
            return field_or_cli

        # Simple CLI arg (no dots)
        if "." not in field_or_cli and "[" not in field_or_cli:
            if field_or_cli.startswith("-"):
                if field_or_cli in self.cli_to_field:
                    return self.cli_to_field[field_or_cli]
                raise ValueError(
                    f"Unknown CLI argument '{field_or_cli}'. "
                    f"Available: {list(self.cli_to_field.keys())}"
                )
            return field_or_cli

        # Path with potential CLI args - resolve each segment
        result_parts: list[str] = []
        i = 0
        while i < len(field_or_cli):
            # Skip dots
            if field_or_cli[i] == ".":
                result_parts.append(".")
                i += 1
                continue

            # Handle bracket notation (pass through as-is)
            if field_or_cli[i] == "[":
                end = field_or_cli.index("]", i) + 1
                result_parts.append(field_or_cli[i:end])
                i = end
                continue

            # Find end of segment (next dot, bracket, or end)
            end = i
            while end < len(field_or_cli) and field_or_cli[end] not in ".[]":
                end += 1
            segment = field_or_cli[i:end]

            # Resolve CLI name if it looks like one
            if segment.startswith("-"):
                if segment in self.cli_to_field:
                    segment = self.cli_to_field[segment]
                else:
                    raise ValueError(
                        f"Unknown CLI argument '{segment}' in path '{field_or_cli}'. "
                        f"Available: {list(self.cli_to_field.keys())}"
                    )

            result_parts.append(segment)
            i = end

        return "".join(result_parts)

    def get_value(self, field: str) -> Any:
        """Get a field's value from the model, resolving CLI names and dot-paths.

        Supports:
            - Simple field names: "request_count"
            - CLI argument names: "--request-count"
            - Dot-notation paths: "load.warmup.enabled"
            - Array indexing: "phases[0].duration"
            - Mixed: "load.phases[0].concurrency"
        """
        resolved = self.resolve_field(field)
        return _get_nested_value(self.model, resolved)

    def get_set_fields(self) -> list[str]:
        """Return fields that are set (not None)."""
        return [f for f in self.fields if _is_set(self.get_value(f))]

    def get_truthy_fields(self) -> list[str]:
        """Return fields that are truthy."""
        return [f for f in self.fields if _is_truthy(self.get_value(f))]

    def is_explicitly_set(self, field: str) -> bool:
        """Check if a field was explicitly provided by the user.

        Uses model_fields_set to determine if the user actually passed the field,
        regardless of its value. Works correctly for fields with non-None defaults
        (e.g., int=0) where _is_set always returns True.
        """
        resolved = self.resolve_field(field)
        return _is_explicitly_set(self.model, resolved)

    def get_explicitly_set_fields(self) -> list[str]:
        """Return fields that were explicitly provided by the user."""
        return [f for f in self.fields if _is_explicitly_set(self.model, f)]

    def format_field(self, field: str) -> str:
        """Format field name for display (uses CLI name if available)."""
        resolved = self.resolve_field(field)
        return _format_field_name(resolved, self.display_names)

    def format_fields(self, fields: list[str] | tuple[str, ...]) -> str:
        """Format list of fields for display."""
        return _format_field_list(fields, self.display_names)


class ConstraintsError(ValueError):
    """Error raised when one or more constraint validations fail.

    Provides detailed information about each violation for programmatic access.

    Attributes:
        violations: List of ConstraintViolation objects describing each failure.

    Example:
        try:
            config = MyConfig(option_a="x", option_b="y")  # Both set but mutually exclusive
        except ConstraintsError as e:
            for violation in e.violations:
                print(f"{violation.constraint_class.__name__}: {violation.message}")
                print(f"  Fields: {violation.fields}")
    """

    def __init__(self, violations: list[ConstraintViolation]) -> None:
        self.violations = violations
        super().__init__(str(self))

    def __str__(self) -> str:
        """Return newline-separated error messages for all violations."""
        return "\n".join(v.message for v in self.violations)

    def __repr__(self) -> str:
        """Return detailed representation including all violation details."""
        violation_reprs = [
            f"ConstraintViolation("
            f"constraint_class={v.constraint_class.__name__}, "
            f"fields={v.fields}, "
            f"message={v.message!r})"
            for v in self.violations
        ]
        return f"ConstraintsError(violations=[{', '.join(violation_reprs)}])"


# =============================================================================
# HELPER FUNCTIONS (used by constraint validate methods)
# =============================================================================


def _is_set(value: Any) -> bool:
    """Check if a value is considered 'set' (not None)."""
    return value is not None


def _is_truthy(value: Any) -> bool:
    """Check if a value is truthy (bool(value) is True)."""
    return bool(value)


def _is_explicitly_set(model: Any, field_name: str) -> bool:
    """Check if a field was explicitly provided by the user (in model_fields_set).

    Unlike _is_set (not None) or _is_truthy (bool), this checks whether the
    user actually passed the field, regardless of its value. Works correctly
    for fields with non-None defaults (e.g., int=0) where _is_set always
    returns True.
    """
    return field_name in model.model_fields_set


def _get_nested_value(obj: Any, path: str) -> Any:
    """Get a value from a nested object using dot-notation path.

    Supports:
        - Dot notation: "load.warmup.enabled"
        - Array indexing: "phases[0].duration"
        - Mixed: "load.phases[0].concurrency"

    Args:
        obj: The root object to traverse.
        path: The dot-notation path to the value.

    Returns:
        The value at the specified path.

    Raises:
        AttributeError: If a path segment doesn't exist.
        IndexError: If an array index is out of bounds.
        TypeError: If trying to index a non-sequence.
    """
    # Fast path: no dots or brackets means simple attribute access
    if "." not in path and "[" not in path:
        return getattr(obj, path)

    current = obj

    # Parse path into segments
    # "load.phases[0].duration" -> [("load", None), ("phases", None), (None, "0"), ("duration", None)]
    pos = 0
    while pos < len(path):
        # Skip dots
        if path[pos] == ".":
            pos += 1
            continue

        # Check for bracket index
        if path[pos] == "[":
            end = path.index("]", pos)
            index = int(path[pos + 1 : end])
            current = current[index]
            pos = end + 1
        else:
            # Find end of attribute name (next dot, bracket, or end)
            end = pos
            while end < len(path) and path[end] not in ".[]":
                end += 1
            attr = path[pos:end]
            current = getattr(current, attr)
            pos = end

    return current


# =============================================================================
# BASE CONSTRAINT CLASS
# =============================================================================


@dataclass(frozen=True)
class BaseConstraint:
    """Base class for all constraint annotations.

    All constraint subclasses should implement the `validate` method
    to perform their specific validation logic.
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType]

    def get_group_key(self) -> str | None:
        """Return the group key for group-based constraints, or None for field-specific.

        Constraints with a `group` attribute return that value; others return None.
        """
        return getattr(self, "group", None)

    def validate(self, ctx: ValidationContext) -> str | None:
        """Validate the constraint against the model.

        Subclasses must implement this method.

        Args:
            ctx: Validation context containing model, fields, display names, etc.

        Returns:
            An error message string if validation fails, or None if valid.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate()"
        )


# =============================================================================
# GROUP-BASED CONSTRAINTS (fields share a group identifier)
# =============================================================================


@dataclass(frozen=True)
class ConflictsWith(BaseConstraint):
    """This field cannot be set together with the specified field(s).

    Direct field reference - specify exactly which fields conflict.
    If this field is set, none of the specified fields can be set.

    Args:
        *fields: Field names (or CLI args) that conflict with each other.
        truthy: If True, use truthy check (for booleans). If False, use is-set check (not None).

    Example::

        option_a: Annotated[str | None, ConflictsWith("option_b")] = None
        option_b: str | None = None  # No annotation needed (asymmetric)

        # For boolean fields, use truthy=True:
        @wired_constraints(ConflictsWith("load.enabled", "debug.enabled", truthy=True))
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.CONFLICTS_WITH

    other_fields: tuple[str, ...]
    truthy: bool

    def __init__(self, *fields: str, truthy: bool = False) -> None:
        object.__setattr__(self, "other_fields", fields)
        object.__setattr__(self, "truthy", truthy)

    def validate(self, ctx: ValidationContext) -> str | None:
        check_fn = _is_truthy if self.truthy else _is_set
        source = ctx.source_field
        if not check_fn(ctx.get_value(source)):
            return None

        # Exclude source from conflict check (for wired constraints where source is in other_fields)
        conflicts = [
            f for f in self.other_fields if f != source and check_fn(ctx.get_value(f))
        ]
        if conflicts:
            source_display = ctx.format_field(source)
            conflicts_display = ctx.format_fields(conflicts)
            return f"{source_display} cannot be used with {conflicts_display}"
        return None


@dataclass(frozen=True)
class ExclusiveGroup(BaseConstraint):
    """Only one field in the group can be set (not None).

    Group-based - multiple fields join the same named group.
    Use when you have N fields that are all mutually exclusive.

    Example::

        option_a: Annotated[str | None, ExclusiveGroup("opts")] = None
        option_b: Annotated[str | None, ExclusiveGroup("opts")] = None
        option_c: Annotated[str | None, ExclusiveGroup("opts")] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.MUTUALLY_EXCLUSIVE

    group: str

    def validate(self, ctx: ValidationContext) -> str | None:
        set_fields = ctx.get_set_fields()
        if len(set_fields) > 1:
            set_display = ctx.format_fields(set_fields)
            all_display = ctx.format_fields(ctx.fields)
            return f"Only one of {all_display} can be specified. Got multiple: {set_display}"
        return None


@dataclass(frozen=True)
class TruthyExclusiveGroup(BaseConstraint):
    """Only one field in the group can be truthy (handles booleans, empty collections).

    Unlike ExclusiveGroup which checks `is not None`, this checks truthiness.
    Use for boolean flags where `False` should not count as "set".

    Example:
        use_ssl: Annotated[bool, TruthyExclusiveGroup("mode")] = False
        use_plaintext: Annotated[bool, TruthyExclusiveGroup("mode")] = False
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.TRUTHY_MUTUALLY_EXCLUSIVE

    group: str

    def validate(self, ctx: ValidationContext) -> str | None:
        truthy_fields = ctx.get_truthy_fields()
        if len(truthy_fields) > 1:
            truthy_display = ctx.format_fields(truthy_fields)
            all_display = ctx.format_fields(ctx.fields)
            return f"Only one of {all_display} can be enabled. Got multiple: {truthy_display}"
        return None


@dataclass(frozen=True)
class AtLeastOne(BaseConstraint):
    """At least one field in the group must be set (not None).

    Example::

        username: Annotated[str | None, AtLeastOne("auth")] = None
        token: Annotated[str | None, AtLeastOne("auth")] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.AT_LEAST_ONE

    group: str

    def validate(self, ctx: ValidationContext) -> str | None:
        set_fields = ctx.get_set_fields()
        if len(set_fields) == 0:
            all_display = ctx.format_fields(ctx.fields)
            return f"At least one of {all_display} must be specified"
        return None


@dataclass(frozen=True)
class AtLeastOneTruthy(BaseConstraint):
    """At least one field in the group must be truthy.

    Unlike AtLeastOne which checks `is not None`, this checks truthiness.
    Use when empty strings or False values should not count as "set".

    Example::

        name: Annotated[str, AtLeastOneTruthy("identity")] = ""
        id: Annotated[int, AtLeastOneTruthy("identity")] = 0
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.AT_LEAST_ONE_TRUTHY

    group: str

    def validate(self, ctx: ValidationContext) -> str | None:
        truthy_fields = ctx.get_truthy_fields()
        if len(truthy_fields) == 0:
            all_display = ctx.format_fields(ctx.fields)
            return f"At least one of {all_display} must have a non-empty value"
        return None


@dataclass(frozen=True)
class AllOrNone(BaseConstraint):
    """All fields in the group must be set together, or none at all.

    Example::

        ssl_cert: Annotated[str | None, AllOrNone("ssl")] = None
        ssl_key: Annotated[str | None, AllOrNone("ssl")] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.ALL_OR_NONE

    group: str

    def validate(self, ctx: ValidationContext) -> str | None:
        set_fields = ctx.get_set_fields()
        if 0 < len(set_fields) < len(ctx.fields):
            unset_fields = [f for f in ctx.fields if not _is_set(ctx.get_value(f))]
            all_display = ctx.format_fields(ctx.fields)
            set_display = ctx.format_fields(set_fields)
            unset_display = ctx.format_fields(unset_fields)
            return (
                f"Fields {all_display} must all be specified together or none at all. "
                f"Got: {set_display}, missing: {unset_display}"
            )
        return None


@dataclass(frozen=True)
class ExactlyOne(BaseConstraint):
    """Exactly one field in the group must be set (not None).

    Unlike ConflictsWith which allows zero, this requires exactly one.
    Common for CLI tools where one option must be chosen.

    Example::

        output_file: Annotated[str | None, ExactlyOne("output")] = None
        output_stdout: Annotated[bool | None, ExactlyOne("output")] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.EXACTLY_ONE

    group: str

    def validate(self, ctx: ValidationContext) -> str | None:
        set_fields = ctx.get_set_fields()
        all_display = ctx.format_fields(ctx.fields)
        if len(set_fields) == 0:
            return f"Exactly one of {all_display} must be specified, but none were set"
        if len(set_fields) > 1:
            set_display = ctx.format_fields(set_fields)
            return f"Exactly one of {all_display} must be specified. Got multiple: {set_display}"
        return None


@dataclass(frozen=True)
class ExactlyOneTruthy(BaseConstraint):
    """Exactly one field in the group must be truthy.

    Unlike TruthyExclusiveGroup which allows zero truthy, this requires exactly one.

    Example::

        use_http: Annotated[bool, ExactlyOneTruthy("protocol")] = False
        use_grpc: Annotated[bool, ExactlyOneTruthy("protocol")] = False
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.EXACTLY_ONE_TRUTHY

    group: str

    def validate(self, ctx: ValidationContext) -> str | None:
        truthy_fields = ctx.get_truthy_fields()
        all_display = ctx.format_fields(ctx.fields)
        if len(truthy_fields) == 0:
            return (
                f"Exactly one of {all_display} must be enabled, but none were enabled"
            )
        if len(truthy_fields) > 1:
            truthy_display = ctx.format_fields(truthy_fields)
            return f"Exactly one of {all_display} must be enabled. Got multiple: {truthy_display}"
        return None


@dataclass(frozen=True)
class MinFieldsSet(BaseConstraint):
    """At least N fields in the group must be set (not None).

    Generalization of AtLeastOne (which is MinFieldsSet with count=1).

    Example::

        # At least 2 of these must be provided
        name: Annotated[str | None, MinFieldsSet("identity", 2)] = None
        email: Annotated[str | None, MinFieldsSet("identity", 2)] = None
        phone: Annotated[str | None, MinFieldsSet("identity", 2)] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.MIN_FIELDS_SET

    group: str
    count: int

    def validate(self, ctx: ValidationContext) -> str | None:
        set_fields = ctx.get_set_fields()
        if len(set_fields) < self.count:
            all_display = ctx.format_fields(ctx.fields)
            set_display = ctx.format_fields(set_fields)
            return (
                f"At least {self.count} of {all_display} must be specified. "
                f"Only {len(set_fields)} set: {set_display}"
            )
        return None


@dataclass(frozen=True)
class MaxFieldsSet(BaseConstraint):
    """At most N fields in the group can be set (not None).

    Generalization of ConflictsWith (which is MaxFieldsSet with count=1).

    Example::

        # At most 2 of these can be set
        opt_a: Annotated[str | None, MaxFieldsSet("options", 2)] = None
        opt_b: Annotated[str | None, MaxFieldsSet("options", 2)] = None
        opt_c: Annotated[str | None, MaxFieldsSet("options", 2)] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.MAX_FIELDS_SET

    group: str
    count: int

    def validate(self, ctx: ValidationContext) -> str | None:
        set_fields = ctx.get_set_fields()
        if len(set_fields) > self.count:
            all_display = ctx.format_fields(ctx.fields)
            set_display = ctx.format_fields(set_fields)
            return (
                f"At most {self.count} of {all_display} can be specified. "
                f"Got {len(set_fields)}: {set_display}"
            )
        return None


@dataclass(frozen=True)
class FieldCountRange(BaseConstraint):
    """Between min_count and max_count fields in the group must be set.

    Combines MinFieldsSet and MaxFieldsSet into a single constraint.

    Example::

        # Between 1 and 3 of these must be set
        opt_a: Annotated[str | None, FieldCountRange("options", 1, 3)] = None
        opt_b: Annotated[str | None, FieldCountRange("options", 1, 3)] = None
        opt_c: Annotated[str | None, FieldCountRange("options", 1, 3)] = None
        opt_d: Annotated[str | None, FieldCountRange("options", 1, 3)] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.FIELD_COUNT_RANGE

    group: str
    min_count: int
    max_count: int

    def validate(self, ctx: ValidationContext) -> str | None:
        set_fields = ctx.get_set_fields()
        count = len(set_fields)
        all_display = ctx.format_fields(ctx.fields)
        if count < self.min_count:
            set_display = ctx.format_fields(set_fields)
            return (
                f"At least {self.min_count} of {all_display} must be specified. "
                f"Only {count} set: {set_display}"
            )
        if count > self.max_count:
            set_display = ctx.format_fields(set_fields)
            return (
                f"At most {self.max_count} of {all_display} can be specified. "
                f"Got {count}: {set_display}"
            )
        return None


@dataclass(frozen=True)
class AtLeastOneUnless(BaseConstraint):
    """At least one field in the group must be set, unless a condition is met.

    Use when a group of fields requires at least one to be set, but this
    requirement is waived when another field has a specific value.

    Example::

        # At least one stop condition required, unless fixed_schedule is True
        requests: Annotated[int | None, AtLeastOneUnless("stop", "fixed_schedule", True)] = None
        duration: Annotated[float | None, AtLeastOneUnless("stop", "fixed_schedule", True)] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.CONDITIONAL_AT_LEAST_ONE

    group: str
    unless_field: str
    unless_value: Any

    def validate(self, ctx: ValidationContext) -> str | None:
        # If the unless condition is met, skip the validation
        unless_actual = ctx.get_value(self.unless_field)
        if unless_actual == self.unless_value:
            return None
        # Check that at least one field is set
        set_fields = ctx.get_set_fields()
        if len(set_fields) == 0:
            all_display = ctx.format_fields(ctx.fields)
            unless_display = ctx.format_field(self.unless_field)
            return (
                f"At least one of {all_display} must be specified "
                f"when {unless_display} is not {self.unless_value!r}"
            )
        return None


@dataclass(frozen=True)
class GroupConflictsWith(BaseConstraint):
    """Two groups of fields are mutually exclusive.

    If any field in this group is set, no field in the conflicting group can be set.
    Both groups must use the same `conflicts_with` value pointing to each other.

    Args:
        group: The group name this field belongs to.
        conflicts_with: The group name that conflicts with this group.
        explicit: If True, use ``model_fields_set`` to determine if fields are set.
            This correctly handles fields with non-None defaults (e.g., ``int = 0``)
            that would always appear "set" with the default ``is not None`` check.

    Example::

        # Fields with None defaults (explicit=False is fine)
        option_a: Annotated[str | None, GroupConflictsWith("group_a", "group_b")] = None
        option_b: Annotated[str | None, GroupConflictsWith("group_b", "group_a")] = None

        # Fields with non-None defaults (need explicit=True)
        num_prefix_prompts: Annotated[int, GroupConflictsWith("legacy", "context", explicit=True)] = 0
        shared_prompt_length: Annotated[int | None, GroupConflictsWith("context", "legacy", explicit=True)] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.GROUP_MUTUALLY_EXCLUSIVE

    group: str
    conflicts_with: str
    explicit: bool = False

    def validate(self, ctx: ValidationContext) -> str | None:
        # Get fields in this group that are set (using appropriate check)
        if self.explicit:
            set_in_this = ctx.get_explicitly_set_fields()
        else:
            set_in_this = ctx.get_set_fields()
        if not set_in_this:
            return None

        # Find the conflicting group
        conflicting_groups = ctx.all_groups.get(
            ConstraintType.GROUP_MUTUALLY_EXCLUSIVE, []
        )
        conflicting_fields: list[str] = []
        for other_group in conflicting_groups:
            if other_group.group_name == self.conflicts_with:
                conflicting_fields = list(other_group.fields)
                break

        if not conflicting_fields:
            return None

        # Check if any field in the conflicting group is set
        if self.explicit:
            set_in_other = [
                f for f in conflicting_fields if _is_explicitly_set(ctx.model, f)
            ]
        else:
            set_in_other = [f for f in conflicting_fields if _is_set(ctx.get_value(f))]
        if set_in_other:
            this_display = ctx.format_fields(set_in_this)
            other_display = ctx.format_fields(set_in_other)
            return (
                f"Cannot use {this_display} together with {other_display}. "
                f"Fields in '{self.group}' and '{self.conflicts_with}' are mutually exclusive."
            )
        return None


# =============================================================================
# FIELD-SPECIFIC CONSTRAINTS (constraint metadata stored per-field)
# =============================================================================


@dataclass(frozen=True)
class Requires(BaseConstraint):
    """If this field is set (not None), the specified field is also required.

    Args:
        required_field: Field name (or CLI arg) that is required.
        explicit: If True, check that the required field is in ``model_fields_set``
            instead of just ``is not None``. Use for fields with non-None defaults
            (e.g., ``int = 0``) where the user must explicitly pass the value.

    Example:
        cert_path: Annotated[str | None, Requires("key_path")] = None
        key_path: str | None = None

        # For fields with non-None defaults, use explicit=True:
        context_length: Annotated[int | None, Requires("--num-entries", explicit=True)] = None
        num_entries: int = 100  # Has non-None default
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.REQUIRES_FIELD

    required_field: str
    explicit: bool = False

    def validate(self, ctx: ValidationContext) -> str | None:
        if not _is_set(ctx.get_value(ctx.source_field)):
            return None

        if self.explicit:
            target_met = ctx.is_explicitly_set(self.required_field)
        else:
            target_met = _is_set(ctx.get_value(self.required_field))

        if not target_met:
            source_display = ctx.format_field(ctx.source_field)
            required_display = ctx.format_field(self.required_field)
            return f"{required_display} is required when {source_display} is specified"
        return None


@dataclass(frozen=True)
class RequiresIfTruthy(BaseConstraint):
    """If this field is truthy, the specified field is required.

    Unlike Requires which checks `is not None`, this checks truthiness.
    Use for boolean flags where `True` triggers the requirement.

    Example:
        enable_ssl: Annotated[bool, RequiresIfTruthy("cert_path")] = False
        cert_path: str | None = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = (
        ConstraintType.REQUIRES_FIELD_WHEN_TRUTHY
    )

    required_field: str

    def validate(self, ctx: ValidationContext) -> str | None:
        source_val = ctx.get_value(ctx.source_field)
        if _is_truthy(source_val) and not _is_set(ctx.get_value(self.required_field)):
            source_display = ctx.format_field(ctx.source_field)
            required_display = ctx.format_field(self.required_field)
            return f"{required_display} is required when {source_display} is {source_val!r}"
        return None


@dataclass(frozen=True)
class RequiresAllOf(BaseConstraint):
    """If this field is set, ALL specified fields are also required.

    Example:
        enable_auth: Annotated[bool, RequiresAllOf(["username", "password"])] = False
        username: str | None = None
        password: str | None = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.REQUIRES_ALL_OF

    required_fields: tuple[str, ...]

    def __init__(self, required_fields: list[str] | tuple[str, ...]) -> None:
        object.__setattr__(self, "required_fields", tuple(required_fields))

    def validate(self, ctx: ValidationContext) -> str | None:
        if not _is_set(ctx.get_value(ctx.source_field)):
            return None
        missing = [f for f in self.required_fields if not _is_set(ctx.get_value(f))]
        if missing:
            source_display = ctx.format_field(ctx.source_field)
            required_display = ctx.format_fields(self.required_fields)
            missing_display = ctx.format_fields(missing)
            return f"When {source_display} is specified, {required_display} are also required. Missing: {missing_display}"
        return None


@dataclass(frozen=True)
class RequiresAnyOf(BaseConstraint):
    """If this field is set, at least one of the specified fields must also be set.

    Example:
        enable_notifications: Annotated[bool, RequiresAnyOf(["email", "phone", "webhook"])] = False
        email: str | None = None
        phone: str | None = None
        webhook: str | None = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.REQUIRES_ANY_OF

    required_fields: tuple[str, ...]

    def __init__(self, required_fields: list[str] | tuple[str, ...]) -> None:
        object.__setattr__(self, "required_fields", tuple(required_fields))

    def validate(self, ctx: ValidationContext) -> str | None:
        if not _is_set(ctx.get_value(ctx.source_field)):
            return None
        set_required = [f for f in self.required_fields if _is_set(ctx.get_value(f))]
        if not set_required:
            source_display = ctx.format_field(ctx.source_field)
            required_display = ctx.format_fields(self.required_fields)
            return f"When {source_display} is specified, at least one of {required_display} must also be set"
        return None


@dataclass(frozen=True)
class Forbids(BaseConstraint):
    """If this field is set (not None), the specified field must NOT be set.

    Inverse of Requires. Use when two fields are incompatible.

    Example:
        use_cache: Annotated[bool | None, Forbids("cache_bypass")] = None
        cache_bypass: bool | None = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.FORBIDS_FIELD

    forbidden_field: str

    def validate(self, ctx: ValidationContext) -> str | None:
        if _is_set(ctx.get_value(ctx.source_field)) and _is_set(
            ctx.get_value(self.forbidden_field)
        ):
            source_display = ctx.format_field(ctx.source_field)
            forbidden_display = ctx.format_field(self.forbidden_field)
            return (
                f"{forbidden_display} cannot be used when {source_display} is specified"
            )
        return None


@dataclass(frozen=True)
class ForbidsIfTruthy(BaseConstraint):
    """If this field is truthy, the specified field must NOT be set.

    Inverse of RequiresIfTruthy.

    Example:
        debug_mode: Annotated[bool, ForbidsIfTruthy("production_key")] = False
        production_key: str | None = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.FORBIDS_FIELD_WHEN_TRUTHY

    forbidden_field: str

    def validate(self, ctx: ValidationContext) -> str | None:
        source_val = ctx.get_value(ctx.source_field)
        if _is_truthy(source_val) and _is_set(ctx.get_value(self.forbidden_field)):
            source_display = ctx.format_field(ctx.source_field)
            forbidden_display = ctx.format_field(self.forbidden_field)
            return f"{forbidden_display} cannot be used when {source_display} is {source_val!r}"
        return None


# =============================================================================
# COMPARISON CONSTRAINTS
# =============================================================================

_COMPARISON_OPS: dict[ComparisonOp, Callable[[Any, Any], bool]] = {
    ComparisonOp.GT: lambda a, b: a > b,
    ComparisonOp.GE: lambda a, b: a >= b,
    ComparisonOp.LT: lambda a, b: a < b,
    ComparisonOp.LE: lambda a, b: a <= b,
    ComparisonOp.EQ: lambda a, b: a == b,
    ComparisonOp.NE: lambda a, b: a != b,
}


def _validate_comparison(
    ctx: ValidationContext,
    other_field: str,
    op: ComparisonOp,
) -> str | None:
    """Shared validation logic for comparison constraints."""
    return _validate_comparison_with_source(ctx, ctx.source_field, other_field, op)


def _validate_comparison_with_source(
    ctx: ValidationContext,
    source_field: str,
    other_field: str,
    op: ComparisonOp,
) -> str | None:
    """Shared validation logic for comparison constraints with explicit source."""
    source_val = ctx.get_value(source_field)
    other_val = ctx.get_value(other_field)

    # Skip comparison if either value is None
    if not _is_set(source_val) or not _is_set(other_val):
        return None

    if not _COMPARISON_OPS[op](source_val, other_val):
        source_display = ctx.format_field(source_field)
        other_display = ctx.format_field(other_field)
        return (
            f"{source_display} ({source_val}) must be {_OP_DESCRIPTIONS[op]} "
            f"{other_display} ({other_val})"
        )
    return None


@dataclass(frozen=True)
class CompareTo(BaseConstraint):
    """This field must satisfy a comparison with the specified field.

    Example:
        min_val: int = 0
        max_val: Annotated[int, CompareTo("min_val", ComparisonOp.GE)] = 100
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.COMPARE_TO

    other_field: str
    op: ComparisonOp

    def validate(self, ctx: ValidationContext) -> str | None:
        return _validate_comparison(ctx, self.other_field, self.op)


@dataclass(frozen=True)
class BaseComparisonConstraint(BaseConstraint):
    """Base class for comparison shortcut constraints (GT, GE, LT, LE).

    Subclasses only need to set the `OP` ClassVar.

    For annotated use (source determined by annotation):
        max_val: Annotated[int, GreaterThan("min_val")] = 100

    For wired use (source specified explicitly):
        @wired_constraints(GreaterThan("min_val", source="max_val"))
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.COMPARE_TO
    OP: ClassVar[ComparisonOp]

    other_field: str
    source: str | None = None  # Explicit source for wired constraints

    def validate(self, ctx: ValidationContext) -> str | None:
        # Use explicit source if provided, otherwise use context's source_field
        source_field = self.source if self.source is not None else ctx.source_field
        return _validate_comparison_with_source(
            ctx, source_field, self.other_field, self.OP
        )


@dataclass(frozen=True)
class GreaterThan(BaseComparisonConstraint):
    """This field must be greater than the specified field.

    Example:
        min_val: int = 0
        max_val: Annotated[int, GreaterThan("min_val")] = 100
    """

    OP: ClassVar[ComparisonOp] = ComparisonOp.GT


@dataclass(frozen=True)
class GreaterThanOrEqual(BaseComparisonConstraint):
    """This field must be greater than or equal to the specified field.

    Example:
        min_val: int = 0
        max_val: Annotated[int, GreaterThanOrEqual("min_val")] = 100
    """

    OP: ClassVar[ComparisonOp] = ComparisonOp.GE


@dataclass(frozen=True)
class LessThan(BaseComparisonConstraint):
    """This field must be less than the specified field.

    Example:
        start: int = 0
        end: Annotated[int, LessThan("start")] = -1
    """

    OP: ClassVar[ComparisonOp] = ComparisonOp.LT


@dataclass(frozen=True)
class LessThanOrEqual(BaseComparisonConstraint):
    """This field must be less than or equal to the specified field.

    Example:
        start_offset: int = 0
        end_offset: Annotated[int, LessThanOrEqual("start_offset")] = 100
    """

    OP: ClassVar[ComparisonOp] = ComparisonOp.LE


@dataclass(frozen=True)
class Matches(BaseConstraint):
    """This field must equal the specified field.

    Useful for confirmation fields like password confirmation.

    Example:
        password: str = ""
        password_confirm: Annotated[str, Matches("password")] = ""
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.MATCHES_FIELD

    other_field: str

    def validate(self, ctx: ValidationContext) -> str | None:
        source_val = ctx.get_value(ctx.source_field)
        other_val = ctx.get_value(self.other_field)
        # Skip if either is None
        if not _is_set(source_val) or not _is_set(other_val):
            return None
        if source_val != other_val:
            source_display = ctx.format_field(ctx.source_field)
            other_display = ctx.format_field(self.other_field)
            return (
                f"{source_display} ({source_val!r}) must match "
                f"{other_display} ({other_val!r})"
            )
        return None


@dataclass(frozen=True)
class NotEqualTo(BaseConstraint):
    """This field must NOT equal the specified field.

    Useful for ensuring two fields have different values.

    Example:
        primary_endpoint: str = ""
        backup_endpoint: Annotated[str, NotEqualTo("primary_endpoint")] = ""
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.NOT_EQUAL_TO_FIELD

    other_field: str

    def validate(self, ctx: ValidationContext) -> str | None:
        source_val = ctx.get_value(ctx.source_field)
        other_val = ctx.get_value(self.other_field)
        # Skip if either is None
        if not _is_set(source_val) or not _is_set(other_val):
            return None
        if source_val == other_val:
            source_display = ctx.format_field(ctx.source_field)
            other_display = ctx.format_field(self.other_field)
            return (
                f"{source_display} ({source_val!r}) must be different from "
                f"{other_display} ({other_val!r})"
            )
        return None


# =============================================================================
# CONDITIONAL CONSTRAINTS
# =============================================================================


@dataclass(frozen=True)
class RequiredIf(BaseConstraint):
    """This field is required when another field equals a specific value.

    Example:
        auth_type: str = "none"
        cert: Annotated[str | None, RequiredIf("auth_type", "ssl")] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.CONDITIONAL_REQUIREMENT

    trigger_field: str
    trigger_value: Any

    def validate(self, ctx: ValidationContext) -> str | None:
        trigger_actual = ctx.get_value(self.trigger_field)
        if trigger_actual == self.trigger_value and not _is_set(
            ctx.get_value(ctx.source_field)
        ):
            source_display = ctx.format_field(ctx.source_field)
            trigger_display = ctx.format_field(self.trigger_field)
            return f"{source_display} is required when {trigger_display} is {self.trigger_value!r}"
        return None


@dataclass(frozen=True)
class RequiredIfIn(BaseConstraint):
    """This field is required when another field equals one of several values.

    Example:
        phase_type: str = "concurrency"
        rate: Annotated[float | None, RequiredIfIn("phase_type", ["poisson", "gamma"])] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = (
        ConstraintType.CONDITIONAL_REQUIREMENT_IN
    )

    trigger_field: str
    trigger_values: tuple[Any, ...]

    def __init__(
        self, trigger_field: str, trigger_values: list[Any] | tuple[Any, ...]
    ) -> None:
        object.__setattr__(self, "trigger_field", trigger_field)
        object.__setattr__(self, "trigger_values", tuple(trigger_values))

    def validate(self, ctx: ValidationContext) -> str | None:
        trigger_actual = ctx.get_value(self.trigger_field)
        if trigger_actual in self.trigger_values and not _is_set(
            ctx.get_value(ctx.source_field)
        ):
            source_display = ctx.format_field(ctx.source_field)
            trigger_display = ctx.format_field(self.trigger_field)
            return f"{source_display} is required when {trigger_display} is {trigger_actual!r}"
        return None


@dataclass(frozen=True)
class ForbiddenWith(BaseConstraint):
    """This field is forbidden (must be None/unset) when another field equals a specific value.

    Example:
        phase_type: str = "concurrency"
        rate: Annotated[float | None, ForbiddenWith("phase_type", "concurrency")] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.FORBIDDEN_WITH

    trigger_field: str
    trigger_value: Any

    def validate(self, ctx: ValidationContext) -> str | None:
        trigger_actual = ctx.get_value(self.trigger_field)
        if trigger_actual == self.trigger_value and _is_set(
            ctx.get_value(ctx.source_field)
        ):
            source_display = ctx.format_field(ctx.source_field)
            trigger_display = ctx.format_field(self.trigger_field)
            return f"{source_display} cannot be used when {trigger_display} is {self.trigger_value!r}"
        return None


@dataclass(frozen=True)
class ForbiddenWithIn(BaseConstraint):
    """This field is forbidden when another field equals one of several values.

    Example:
        phase_type: str = "fixed_schedule"
        rate: Annotated[float | None, ForbiddenWithIn("phase_type", ["concurrency", "fixed_schedule"])] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.FORBIDDEN_WITH_IN

    trigger_field: str
    trigger_values: tuple[Any, ...]

    def __init__(
        self, trigger_field: str, trigger_values: list[Any] | tuple[Any, ...]
    ) -> None:
        object.__setattr__(self, "trigger_field", trigger_field)
        object.__setattr__(self, "trigger_values", tuple(trigger_values))

    def validate(self, ctx: ValidationContext) -> str | None:
        trigger_actual = ctx.get_value(self.trigger_field)
        if trigger_actual in self.trigger_values and _is_set(
            ctx.get_value(ctx.source_field)
        ):
            source_display = ctx.format_field(ctx.source_field)
            trigger_display = ctx.format_field(self.trigger_field)
            return f"{source_display} cannot be used when {trigger_display} is {trigger_actual!r}"
        return None


@dataclass(frozen=True)
class AllowedOnlyWith(BaseConstraint):
    """This field is only allowed (can be set) when another field equals a specific value.

    Inverse of ForbiddenWith. Use when a field is only valid for one specific
    mode/type and should be rejected for all others.

    Example:
        phase_type: str = "concurrency"
        smoothness: Annotated[float | None, AllowedOnlyWith("phase_type", "gamma")] = None
    """

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.ALLOWED_ONLY_WITH

    trigger_field: str
    trigger_value: Any

    def validate(self, ctx: ValidationContext) -> str | None:
        if not _is_set(ctx.get_value(ctx.source_field)):
            return None
        trigger_actual = ctx.get_value(self.trigger_field)
        if trigger_actual != self.trigger_value:
            source_display = ctx.format_field(ctx.source_field)
            trigger_display = ctx.format_field(self.trigger_field)
            return (
                f"{source_display} can only be used when {trigger_display} "
                f"is {self.trigger_value!r}, got {trigger_actual!r}"
            )
        return None


# =============================================================================
# SHORTHAND CONSTRAINTS CLASS
# =============================================================================


@dataclass(frozen=True)
class Constraints:
    """Shorthand for declaring multiple constraints on a single field.

    Provides concise field names for common constraints. All fields are optional;
    set only the ones you need. The actual constraint objects are generated in
    __post_init__ and stored in _constraints.

    Example:
        class MyConfig(BaseModel, ConstraintsMixin):
            min_val: int = 0
            max_val: Annotated[int, Constraints(gt="min_val")] = 100

            # Direct field conflict - option_a cannot be used with option_b
            option_a: Annotated[str | None, Constraints(conflicts_with="option_b")] = None
            option_b: str | None = None

            # Group-based exclusivity - only one of these can be set
            mode_x: Annotated[str | None, Constraints(exclusive_group="modes")] = None
            mode_y: Annotated[str | None, Constraints(exclusive_group="modes")] = None

            cert: Annotated[str | None, Constraints(requires="key")] = None
            key: str | None = None

            # Multiple constraints on one field
            value: Annotated[int, Constraints(gt="min_val", requires="enabled")] = 0

    Shorthand Reference:
        Comparison (field name to compare against):
            gt: str          -> GreaterThan(other_field)
            gte: str         -> GreaterThanOrEqual(other_field)
            lt: str          -> LessThan(other_field)
            lte: str         -> LessThanOrEqual(other_field)
            eq: str          -> Matches(other_field)
            ne: str          -> NotEqualTo(other_field)

        Dependency (field names):
            requires: str                -> Requires(field)
            requires_truthy: str         -> RequiresIfTruthy(field)
            requires_explicit: str       -> Requires(field, explicit=True)
            requires_all: tuple[str,...] -> RequiresAllOf(fields)
            requires_any: tuple[str,...] -> RequiresAnyOf(fields)
            forbids: str                 -> Forbids(field)
            forbids_truthy: str          -> ForbidsIfTruthy(field)

        Field conflicts (field names this field conflicts with):
            conflicts_with: str | tuple[str,...] -> ConflictsWith(*fields)

        Group-based (group name):
            exclusive_group: str           -> ExclusiveGroup(group)
            exclusive_group_nonempty: str  -> TruthyExclusiveGroup(group)
            one_of: str             -> AtLeastOne(group)
            one_of_truthy: str      -> AtLeastOneTruthy(group)
            exactly_one: str        -> ExactlyOne(group)
            exactly_one_truthy: str -> ExactlyOneTruthy(group)
            all_or_none: str        -> AllOrNone(group)

        Group with count (group, count):
            min_set: tuple[str, int]       -> MinFieldsSet(group, count)
            max_set: tuple[str, int]       -> MaxFieldsSet(group, count)
            count_range: tuple[str, int, int] -> FieldCountRange(group, min, max)

        Group mutually exclusive (group, conflicts_with):
            conflicting_groups: tuple[str, str] -> GroupConflictsWith(group, conflicts_with)
            explicit_conflicting_groups: tuple[str, str] -> GroupConflictsWith(group, conflicts_with, explicit=True)

        Conditional at least one (group, unless_field, unless_value):
            one_of_unless: tuple[str, str, Any] -> AtLeastOneUnless(...)

        Conditional (trigger_field, trigger_value):
            required_if: tuple[str, Any]              -> RequiredIf(...)
            required_if_in: tuple[str, tuple[Any,...]] -> RequiredIfIn(...)
            forbidden_if: tuple[str, Any]             -> ForbiddenWith(...)
            forbidden_if_in: tuple[str, tuple[Any,...]] -> ForbiddenWithIn(...)
            allowed_only_if: tuple[str, Any]          -> AllowedOnlyWith(...)
    """

    # Comparison constraints (field name to compare against)
    gt: str | None = None
    gte: str | None = None
    lt: str | None = None
    lte: str | None = None
    eq: str | None = None
    ne: str | None = None

    # Dependency constraints
    requires: str | None = None
    requires_truthy: str | None = None
    requires_explicit: str | None = None
    requires_all: tuple[str, ...] | None = None
    requires_any: tuple[str, ...] | None = None
    forbids: str | None = None
    forbids_truthy: str | None = None

    # Field conflicts (field names this field conflicts with)
    conflicts_with: str | tuple[str, ...] | None = None

    # Group-based constraints (group name)
    exclusive_group: str | None = None
    exclusive_group_nonempty: str | None = None
    one_of: str | None = None
    one_of_truthy: str | None = None
    exactly_one: str | None = None
    exactly_one_truthy: str | None = None
    all_or_none: str | None = None

    # Group with count: (group_name, count)
    min_set: tuple[str, int] | None = None
    max_set: tuple[str, int] | None = None
    # Group with range: (group_name, min_count, max_count)
    count_range: tuple[str, int, int] | None = None

    # Group mutually exclusive: (group_name, conflicts_with_group)
    conflicting_groups: tuple[str, str] | None = None
    explicit_conflicting_groups: tuple[str, str] | None = None

    # Conditional at least one: (group_name, unless_field, unless_value)
    one_of_unless: tuple[str, str, Any] | None = None

    # Conditional constraints: (trigger_field, trigger_value)
    required_if: tuple[str, Any] | None = None
    # Conditional constraints: (trigger_field, trigger_values)
    required_if_in: tuple[str, tuple[Any, ...]] | None = None
    # Forbidden constraints: (trigger_field, trigger_value)
    forbidden_if: tuple[str, Any] | None = None
    # Forbidden constraints: (trigger_field, trigger_values)
    forbidden_if_in: tuple[str, tuple[Any, ...]] | None = None
    # Allowed only when: (trigger_field, trigger_value)
    allowed_only_if: tuple[str, Any] | None = None

    # Generated constraints (populated in __post_init__)
    _constraints: tuple[BaseConstraint, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        """Generate actual constraint objects from shorthand fields."""
        constraints: list[BaseConstraint] = []

        # Comparison constraints
        if self.gt is not None:
            constraints.append(GreaterThan(self.gt))
        if self.gte is not None:
            constraints.append(GreaterThanOrEqual(self.gte))
        if self.lt is not None:
            constraints.append(LessThan(self.lt))
        if self.lte is not None:
            constraints.append(LessThanOrEqual(self.lte))
        if self.eq is not None:
            constraints.append(Matches(self.eq))
        if self.ne is not None:
            constraints.append(NotEqualTo(self.ne))

        # Dependency constraints
        if self.requires is not None:
            constraints.append(Requires(self.requires))
        if self.requires_truthy is not None:
            constraints.append(RequiresIfTruthy(self.requires_truthy))
        if self.requires_explicit is not None:
            constraints.append(Requires(self.requires_explicit, explicit=True))
        if self.requires_all is not None:
            constraints.append(RequiresAllOf(self.requires_all))
        if self.requires_any is not None:
            constraints.append(RequiresAnyOf(self.requires_any))
        if self.forbids is not None:
            constraints.append(Forbids(self.forbids))
        if self.forbids_truthy is not None:
            constraints.append(ForbidsIfTruthy(self.forbids_truthy))

        # Field conflicts
        if self.conflicts_with is not None:
            if isinstance(self.conflicts_with, str):
                constraints.append(ConflictsWith(self.conflicts_with))
            else:
                constraints.append(ConflictsWith(*self.conflicts_with))

        # Group-based constraints
        if self.exclusive_group is not None:
            constraints.append(ExclusiveGroup(self.exclusive_group))
        if self.exclusive_group_nonempty is not None:
            constraints.append(TruthyExclusiveGroup(self.exclusive_group_nonempty))
        if self.one_of is not None:
            constraints.append(AtLeastOne(self.one_of))
        if self.one_of_truthy is not None:
            constraints.append(AtLeastOneTruthy(self.one_of_truthy))
        if self.exactly_one is not None:
            constraints.append(ExactlyOne(self.exactly_one))
        if self.exactly_one_truthy is not None:
            constraints.append(ExactlyOneTruthy(self.exactly_one_truthy))
        if self.all_or_none is not None:
            constraints.append(AllOrNone(self.all_or_none))

        # Group with count
        if self.min_set is not None:
            group, count = self.min_set
            constraints.append(MinFieldsSet(group, count))
        if self.max_set is not None:
            group, count = self.max_set
            constraints.append(MaxFieldsSet(group, count))
        if self.count_range is not None:
            group, min_count, max_count = self.count_range
            constraints.append(FieldCountRange(group, min_count, max_count))

        # Group mutually exclusive
        if self.conflicting_groups is not None:
            group, conflicts_with = self.conflicting_groups
            constraints.append(GroupConflictsWith(group, conflicts_with))
        if self.explicit_conflicting_groups is not None:
            group, conflicts_with = self.explicit_conflicting_groups
            constraints.append(GroupConflictsWith(group, conflicts_with, explicit=True))

        # Conditional at least one
        if self.one_of_unless is not None:
            group, unless_field, unless_value = self.one_of_unless
            constraints.append(AtLeastOneUnless(group, unless_field, unless_value))

        # Conditional constraints
        if self.required_if is not None:
            trigger_field, trigger_value = self.required_if
            constraints.append(RequiredIf(trigger_field, trigger_value))
        if self.required_if_in is not None:
            trigger_field, trigger_values = self.required_if_in
            constraints.append(RequiredIfIn(trigger_field, trigger_values))
        if self.forbidden_if is not None:
            trigger_field, trigger_value = self.forbidden_if
            constraints.append(ForbiddenWith(trigger_field, trigger_value))
        if self.forbidden_if_in is not None:
            trigger_field, trigger_values = self.forbidden_if_in
            constraints.append(ForbiddenWithIn(trigger_field, trigger_values))
        if self.allowed_only_if is not None:
            trigger_field, trigger_value = self.allowed_only_if
            constraints.append(AllowedOnlyWith(trigger_field, trigger_value))

        object.__setattr__(self, "_constraints", tuple(constraints))


# =============================================================================
# FIELD DISPLAY NAME EXTRACTION
# =============================================================================


def _extract_cli_name(metadata: tuple[Any, ...]) -> str | None:
    """Extract CLI parameter name from field metadata if present.

    Looks for cyclopts.Parameter or subclasses in the metadata and extracts
    the first name (e.g., '--request-count').

    Args:
        metadata: The __metadata__ tuple from an Annotated type.

    Returns:
        The CLI flag name (e.g., '--request-count') or None if not found.
    """
    for meta in metadata:
        # Check if it looks like a cyclopts Parameter (has 'name' attribute that's a tuple)
        if hasattr(meta, "name") and isinstance(getattr(meta, "name", None), tuple):
            names = meta.name
            if names:
                return names[0]  # Return first name (e.g., '--request-count')
    return None


def collect_field_display_names(
    model_cls: type[BaseModel],
) -> dict[str, str]:
    """
    Extract display names for fields from CLI parameter annotations.

    Traverses the MRO to collect CLI parameter names from parent classes.

    Args:
        model_cls: The Pydantic model class to extract display names from.

    Returns:
        A dictionary mapping field names to their CLI display names.
        Fields without CLI parameters are not included.
    """
    display_names: dict[str, str] = {}

    # Traverse MRO to collect from parent classes
    for cls in model_cls.__mro__:
        if cls is object or not hasattr(cls, "__annotations__"):
            continue

        try:
            hints = get_type_hints(cls, include_extras=True)
        except (NameError, TypeError, AttributeError):
            # NameError: forward reference can't be resolved
            # TypeError: invalid type annotation
            # AttributeError: malformed annotation missing expected attributes
            continue

        for field_name in cls.__annotations__:
            if field_name in display_names:
                continue  # Already found from a subclass
            if field_name not in hints:
                continue

            hint = hints[field_name]
            if not hasattr(hint, "__metadata__"):
                continue

            cli_name = _extract_cli_name(hint.__metadata__)
            if cli_name:
                display_names[field_name] = cli_name

    return display_names


def collect_cli_to_field_mapping(
    model_cls: type[BaseModel],
) -> dict[str, str]:
    """
    Build a reverse mapping from CLI argument names to field names.

    This enables constraints to reference fields by their CLI names
    (e.g., '--request-count' instead of 'request_count').

    Args:
        model_cls: The Pydantic model class to extract CLI mappings from.

    Returns:
        A dictionary mapping CLI names to field names.
        All CLI names for a field are mapped (e.g., both '--request-count'
        and '--num-requests' map to 'request_count').
    """
    cli_to_field: dict[str, str] = {}

    # Traverse MRO to collect from parent classes
    for cls in model_cls.__mro__:
        if cls is object or not hasattr(cls, "__annotations__"):
            continue

        try:
            hints = get_type_hints(cls, include_extras=True)
        except (NameError, TypeError, AttributeError):
            continue

        for field_name in cls.__annotations__:
            if field_name not in hints:
                continue

            hint = hints[field_name]
            if not hasattr(hint, "__metadata__"):
                continue

            # Extract all CLI names for this field
            cli_names = _extract_all_cli_names(hint.__metadata__)
            for cli_name in cli_names:
                if cli_name not in cli_to_field:
                    cli_to_field[cli_name] = field_name

    return cli_to_field


def _extract_all_cli_names(metadata: tuple[Any, ...]) -> list[str]:
    """Extract all CLI parameter names from field metadata.

    Unlike _extract_cli_name which returns only the first name,
    this returns all names for reverse lookup purposes.

    Args:
        metadata: The __metadata__ tuple from an Annotated type.

    Returns:
        List of all CLI flag names (e.g., ['--request-count', '--num-requests']).
    """
    for meta in metadata:
        # Check if it looks like a cyclopts Parameter (has 'name' attribute that's a tuple)
        if hasattr(meta, "name") and isinstance(getattr(meta, "name", None), tuple):
            names = meta.name
            if names:
                return list(names)
    return []


def _extract_group(metadata: tuple[Any, ...]) -> Any | None:
    """Extract cyclopts Group from field metadata if present.

    Looks for cyclopts.Parameter or subclasses in the metadata and extracts
    the group. The group attribute is stored as a tuple after cyclopts conversion.

    Args:
        metadata: The __metadata__ tuple from an Annotated type.

    Returns:
        The cyclopts Group object, or None if not found.
    """
    for meta in metadata:
        if hasattr(meta, "group") and isinstance(getattr(meta, "group", None), tuple):
            groups = meta.group
            if groups:
                return groups[0]
    return None


def collect_group_fields(
    model_cls: type[BaseModel],
) -> dict[Any, list[str]]:
    """
    Build a lookup table from cyclopts Group to field names.

    Traverses the MRO to collect group assignments from CLI parameter annotations.
    Each entry maps a cyclopts Group instance to the list of field names assigned to it.

    Args:
        model_cls: The Pydantic model class to extract group mappings from.

    Returns:
        A dictionary mapping cyclopts Group objects to lists of field names.
    """
    group_fields: dict[Any, list[str]] = defaultdict(list)

    for cls in model_cls.__mro__:
        if cls is object or not hasattr(cls, "__annotations__"):
            continue

        try:
            hints = get_type_hints(cls, include_extras=True)
        except (NameError, TypeError, AttributeError):
            continue

        for field_name in cls.__annotations__:
            if field_name not in hints:
                continue

            hint = hints[field_name]
            if not hasattr(hint, "__metadata__"):
                continue

            group = _extract_group(hint.__metadata__)
            if group is not None and field_name not in group_fields[group]:
                group_fields[group].append(field_name)

    return dict(group_fields)


def _format_field_name(field_name: str, display_names: dict[str, str]) -> str:
    """Format a field name for display, using CLI name if available."""
    if field_name in display_names:
        return display_names[field_name]
    # Don't quote CLI argument names (already display-ready)
    if field_name.startswith("-"):
        return field_name
    return f"'{field_name}'"


def _format_field_list(
    fields: list[str] | tuple[str, ...], display_names: dict[str, str]
) -> str:
    """Format a list of field names for display."""
    formatted = [_format_field_name(f, display_names) for f in fields]
    if len(formatted) == 1:
        return formatted[0]
    return "[" + ", ".join(formatted) + "]"


# =============================================================================
# CONSTRAINT COLLECTION
# =============================================================================


def collect_constraints(
    model_cls: type[BaseModel],
) -> dict[ConstraintType, list[ConstraintGroup]]:
    """
    Extract constraints from Annotated[] metadata in a model class.

    Traverses the MRO to collect constraints from parent classes as well.

    Args:
        model_cls: The Pydantic model class to extract constraints from.

    Returns:
        A dictionary mapping constraint types to lists of ConstraintGroup objects.
    """
    # Use defaultdict to group constraints by type and then by group key
    grouped: dict[ConstraintType, dict[str, list[tuple[str, BaseConstraint]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    # For non-group constraints, collect directly
    non_grouped: dict[ConstraintType, list[ConstraintGroup]] = defaultdict(list)

    # Traverse MRO to collect from parent classes
    for cls in model_cls.__mro__:
        if cls is object or not hasattr(cls, "__annotations__"):
            continue

        # Get type hints for this specific class (not inherited)
        try:
            hints = get_type_hints(cls, include_extras=True)
        except (NameError, TypeError, AttributeError):
            # NameError: forward reference can't be resolved
            # TypeError: invalid type annotation
            # AttributeError: malformed annotation missing expected attributes
            continue

        # Only process annotations defined directly on this class
        for field_name in cls.__annotations__:
            if field_name not in hints:
                continue

            hint = hints[field_name]

            # Check if it's an Annotated type
            if not hasattr(hint, "__metadata__"):
                continue

            # Extract constraint annotations from metadata
            for meta in hint.__metadata__:
                # Handle Constraints shorthand by expanding to individual constraints
                if isinstance(meta, Constraints):
                    constraint_list = meta._constraints
                elif isinstance(meta, BaseConstraint):
                    constraint_list = (meta,)
                else:
                    continue

                for constraint in constraint_list:
                    constraint_type = constraint.CONSTRAINT_TYPE
                    group_key = constraint.get_group_key()

                    if group_key is not None:
                        # Group-based constraint
                        grouped[constraint_type][group_key].append(
                            (field_name, constraint)
                        )
                    else:
                        # Field-specific constraint
                        non_grouped[constraint_type].append(
                            ConstraintGroup(
                                constraint_type=constraint_type,
                                group_name=field_name,
                                fields=(field_name,),
                                constraint=constraint,
                            )
                        )

    # Convert grouped constraints to ConstraintGroup objects
    result: dict[ConstraintType, list[ConstraintGroup]] = defaultdict(list)

    for constraint_type, groups in grouped.items():
        for group_name, field_constraints in groups.items():
            fields = tuple(fc[0] for fc in field_constraints)
            # Use the first constraint (all in same group share the same constraint type)
            first_constraint = field_constraints[0][1]
            result[constraint_type].append(
                ConstraintGroup(
                    constraint_type=constraint_type,
                    group_name=group_name,
                    fields=fields,
                    constraint=first_constraint,
                )
            )

    # Add non-grouped constraints
    for constraint_type, constraint_groups in non_grouped.items():
        result[constraint_type].extend(constraint_groups)

    return dict(result)


# =============================================================================
# WIRED CONSTRAINTS SUPPORT
# =============================================================================


def _get_constraint_fields(constraint: BaseConstraint) -> tuple[str, ...]:
    """Extract all field paths referenced by a constraint.

    Used by wired constraints to determine which fields to include in the context.

    Args:
        constraint: The constraint to extract fields from.

    Returns:
        Tuple of all field paths referenced by the constraint.
    """
    fields: list[str] = []

    # Check for common field attributes
    if hasattr(constraint, "other_fields"):
        # ConflictsWith, RequiresAllOf, etc.
        fields.extend(constraint.other_fields)  # type: ignore[attr-defined]
    if hasattr(constraint, "other_field"):
        # Comparison constraints
        fields.append(constraint.other_field)  # type: ignore[attr-defined]
    if hasattr(constraint, "source") and constraint.source is not None:  # type: ignore[attr-defined]
        # Explicit source for wired comparison constraints
        fields.insert(0, constraint.source)  # type: ignore[attr-defined]
    if hasattr(constraint, "condition_field"):
        # Conditional constraints
        fields.append(constraint.condition_field)  # type: ignore[attr-defined]
    if hasattr(constraint, "required_field"):
        # Requires
        fields.append(constraint.required_field)  # type: ignore[attr-defined]
    if hasattr(constraint, "forbidden_field"):
        # Forbids
        fields.append(constraint.forbidden_field)  # type: ignore[attr-defined]

    return tuple(fields)


def wired_constraints(
    *constraints: BaseConstraint,
) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """Decorator to define cross-model constraints at the parent level.

    Use this to define constraints that reference fields in nested child models
    using dot-notation paths. The recommended approach is using the fluent F() API.

    Example with F() (recommended):
        from aiperf.config.constraints import F, wired_constraints

        @wired_constraints(
            F('benchmark.max_requests') > F('benchmark.min_requests'),
            F('load.enabled').conflicts_with(F('debug.enabled'), truthy=True),
            F('ssl.cert').requires(F('ssl.key')),
        )
        class AIPerfConfig(BaseModel, ConstraintsMixin):
            benchmark: BenchmarkConfig
            load: LoadConfig
            ssl: SSLConfig

    Example with constraint classes (alternative):
        @wired_constraints(
            ConflictsWith("load.enabled", "debug.enabled", truthy=True),
            GreaterThan("benchmark.min_requests", source="benchmark.max_requests"),
        )
        class AIPerfConfig(BaseModel, ConstraintsMixin):
            ...

    Supports:
        - Dot notation: "load.warmup.enabled"
        - Array indexing: "phases[0].duration"
        - Mixed: "load.phases[0].concurrency"

    Args:
        *constraints: Constraint instances to apply at the top level.

    Returns:
        A class decorator that attaches the constraints to the model.
    """

    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        # Store constraints on the class for the mixin to pick up
        existing = getattr(cls, "_wired_constraints", [])
        cls._wired_constraints = list(existing) + list(constraints)  # type: ignore[attr-defined]
        return cls

    return decorator


# =============================================================================
# FLUENT FIELD API (F)
# =============================================================================


class F:
    """Fluent field reference for wired constraints.

    Provides a Pythonic way to define cross-model constraints using
    comparison operators and method chaining.

    Example:
        from aiperf.config.constraints import F, wired_constraints

        @wired_constraints(
            F('benchmark.max_requests') > F('benchmark.min_requests'),
            F('load.enabled').conflicts_with(F('debug.enabled'), truthy=True),
            F('ssl.cert').requires(F('ssl.key')),
        )
        class Config(BaseModel, ConstraintsMixin):
            benchmark: BenchmarkConfig
            load: LoadConfig
            ssl: SSLConfig

    Supports:
        - Comparison operators: >, <, >=, <=, ==, !=
        - Relationship methods: conflicts_with(), requires(), requires_all()
    """

    __slots__ = ("path",)

    def __init__(self, path: str) -> None:
        """Create a field reference.

        Args:
            path: Dot-notation path to the field (e.g., "load.warmup.enabled").
        """
        self.path = path

    def __repr__(self) -> str:
        return f"F({self.path!r})"

    # -------------------------------------------------------------------------
    # Comparison operators
    # -------------------------------------------------------------------------

    def __gt__(self, other: F) -> _FComparison:
        """Create a greater-than constraint: self > other."""
        return _FComparison(self.path, other.path, ComparisonOp.GT)

    def __ge__(self, other: F) -> _FComparison:
        """Create a greater-than-or-equal constraint: self >= other."""
        return _FComparison(self.path, other.path, ComparisonOp.GE)

    def __lt__(self, other: F) -> _FComparison:
        """Create a less-than constraint: self < other."""
        return _FComparison(self.path, other.path, ComparisonOp.LT)

    def __le__(self, other: F) -> _FComparison:
        """Create a less-than-or-equal constraint: self <= other."""
        return _FComparison(self.path, other.path, ComparisonOp.LE)

    def __eq__(self, other: object) -> _FComparison:  # type: ignore[override]
        """Create an equality constraint: self == other."""
        if not isinstance(other, F):
            return NotImplemented
        return _FComparison(self.path, other.path, ComparisonOp.EQ)

    def __ne__(self, other: object) -> _FComparison:  # type: ignore[override]
        """Create a not-equal constraint: self != other."""
        if not isinstance(other, F):
            return NotImplemented
        return _FComparison(self.path, other.path, ComparisonOp.NE)

    # -------------------------------------------------------------------------
    # Relationship methods
    # -------------------------------------------------------------------------

    def conflicts_with(self, *others: F, truthy: bool = False) -> _FConflicts:
        """Create a mutual exclusivity constraint.

        Args:
            *others: Other field references that conflict with this one.
            truthy: If True, use truthy check. If False, use is-set (not None) check.

        Returns:
            A constraint that ensures these fields are mutually exclusive.
        """
        other_paths = tuple(o.path for o in others)
        return _FConflicts((self.path,) + other_paths, truthy=truthy)

    def requires(self, other: F) -> _FRequires:
        """Create a dependency constraint: if self is set, other must be set.

        Args:
            other: The field that is required when this field is set.

        Returns:
            A constraint that ensures other is set when self is set.
        """
        return _FRequires(self.path, other.path)

    def requires_all(self, *others: F) -> _FRequiresAll:
        """Create a dependency constraint: if self is set, all others must be set.

        Args:
            *others: Fields that are required when this field is set.

        Returns:
            A constraint that ensures all others are set when self is set.
        """
        other_paths = tuple(o.path for o in others)
        return _FRequiresAll(self.path, other_paths)


@dataclass(frozen=True)
class _FComparison(BaseConstraint):
    """Comparison constraint created by F() operator."""

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.COMPARE_TO

    source: str
    other_field: str
    op: ComparisonOp

    def validate(self, ctx: ValidationContext) -> str | None:
        return _validate_comparison_with_source(
            ctx, self.source, self.other_field, self.op
        )


@dataclass(frozen=True)
class _FConflicts(BaseConstraint):
    """Mutual exclusivity constraint created by F().conflicts_with()."""

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.CONFLICTS_WITH

    other_fields: tuple[str, ...]
    truthy: bool = False

    def validate(self, ctx: ValidationContext) -> str | None:
        check_fn = _is_truthy if self.truthy else _is_set

        # Find all fields that are set/truthy
        set_fields = [f for f in self.other_fields if check_fn(ctx.get_value(f))]

        if len(set_fields) > 1:
            fields_display = ctx.format_fields(set_fields)
            return f"Only one of {fields_display} can be set, but multiple are set"
        return None


@dataclass(frozen=True)
class _FRequires(BaseConstraint):
    """Dependency constraint created by F().requires()."""

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.REQUIRES_FIELD

    source: str
    required_field: str

    def validate(self, ctx: ValidationContext) -> str | None:
        if not _is_set(ctx.get_value(self.source)):
            return None

        if not _is_set(ctx.get_value(self.required_field)):
            source_display = ctx.format_field(self.source)
            required_display = ctx.format_field(self.required_field)
            return f"{source_display} requires {required_display} to be set"
        return None


@dataclass(frozen=True)
class _FRequiresAll(BaseConstraint):
    """Dependency constraint created by F().requires_all()."""

    CONSTRAINT_TYPE: ClassVar[ConstraintType] = ConstraintType.REQUIRES_ALL_OF

    source: str
    other_fields: tuple[str, ...]

    def validate(self, ctx: ValidationContext) -> str | None:
        if not _is_set(ctx.get_value(self.source)):
            return None

        missing = [f for f in self.other_fields if not _is_set(ctx.get_value(f))]
        if missing:
            source_display = ctx.format_field(self.source)
            missing_display = ctx.format_fields(missing)
            return f"{source_display} requires {missing_display} to be set"
        return None


# =============================================================================
# CONSTRAINTS MIXIN
# =============================================================================


class ConstraintsMixin:
    """
    Mixin that enforces constraint annotations on Pydantic models.

    Add this mixin to your model class to enable declarative cross-field validation.
    Constraints are collected at class definition time via __init_subclass__ and
    enforced at validation time via @model_validator(mode='after').

    Example:
        class MyConfig(BaseModel, ConstraintsMixin):
            option_a: Annotated[str | None, ConflictsWith("opts")] = None
            option_b: Annotated[str | None, ConflictsWith("opts")] = None

    CLI Parameter Integration:
        When fields are annotated with cyclopts.Parameter (or CLIParameter), the
        error messages will use the CLI flag names (e.g., --request-count) instead
        of field names (e.g., request_count) for better user experience.

    Available Constraints:
        Group-based (fields share a group identifier):
            - ConflictsWith(group): Only one field can be set (not None)
            - TruthyExclusiveGroup(group): Only one field can be truthy
            - AtLeastOne(group): At least one field must be set
            - AtLeastOneTruthy(group): At least one field must be truthy
            - AllOrNone(group): All fields set or none set
            - AtLeastOneUnless(group, unless_field, unless_value): At least one unless condition
            - GroupConflictsWith(group, conflicts_with): Two groups mutually exclusive

        Field-specific:
            - Requires(field): If this is set, other field required
            - RequiresIfTruthy(field): If this is truthy, other field required
            - RequiresAllOf([fields]): If this is set, all listed fields required

        Comparison:
            - CompareTo(field, op): General comparison
            - GreaterThan(field): This > other
            - GreaterThanOrEqual(field): This >= other
            - LessThan(field): This < other
            - LessThanOrEqual(field): This <= other

        Conditional:
            - RequiredIf(field, value): Required when field == value
            - RequiredIfIn(field, values): Required when field in values
            - ForbiddenWith(field, value): Forbidden when field == value
            - ForbiddenWithIn(field, values): Forbidden when field in values
    """

    model_fields_set: ClassVar[set[str]]

    _constraint_groups: ClassVar[dict[ConstraintType, list[ConstraintGroup]]] = {}
    _field_display_names: ClassVar[dict[str, str]] = {}
    _cli_to_field: ClassVar[dict[str, str]] = {}
    _group_fields: ClassVar[dict[Any, list[str]]] = {}
    _wired_constraints: ClassVar[list[BaseConstraint]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Collect constraints, display names, and CLI mappings at class definition time
        if issubclass(cls, BaseModel):
            cls._constraint_groups = collect_constraints(cls)
            cls._field_display_names = collect_field_display_names(cls)
            cls._cli_to_field = collect_cli_to_field_mapping(cls)
            cls._group_fields = collect_group_fields(cls)
            # Wired constraints are set by @wired_constraints decorator before __init_subclass__
            # Just ensure it's a list (may be inherited from parent)
            if not hasattr(cls, "_wired_constraints") or cls._wired_constraints is None:
                cls._wired_constraints = []

    def any_field_set_in_group(self, group: Any) -> bool:
        """Check if any field in the given cyclopts Group was explicitly set.

        Args:
            group: A cyclopts Group instance.

        Returns:
            True if at least one field in the group is in model_fields_set.
        """
        fields = self._group_fields.get(group, [])
        return bool(self.model_fields_set & set(fields))

    @model_validator(mode="after")
    def _validate_constraints(self) -> Self:
        """Validate all collected constraints (both annotated and wired)."""
        violations: list[ConstraintViolation] = []

        # Validate annotated constraints (from field annotations)
        for _constraint_type, groups in self._constraint_groups.items():
            for group in groups:
                ctx = ValidationContext(
                    model=self,
                    fields=group.fields,
                    display_names=self._field_display_names,
                    cli_to_field=self._cli_to_field,
                    all_groups=self._constraint_groups,
                )
                error = group.constraint.validate(ctx)

                if error:
                    violations.append(
                        ConstraintViolation(
                            constraint_class=type(group.constraint),
                            constraint_type=group.constraint.CONSTRAINT_TYPE,
                            message=error,
                            fields=group.fields,
                            group_name=group.group_name,
                        )
                    )

        # Validate wired constraints (from @wired_constraints decorator)
        for constraint in self._wired_constraints:
            fields = _get_constraint_fields(constraint)
            ctx = ValidationContext(
                model=self,
                fields=fields,
                display_names=self._field_display_names,
                cli_to_field=self._cli_to_field,
                all_groups=self._constraint_groups,
            )
            error = constraint.validate(ctx)

            if error:
                violations.append(
                    ConstraintViolation(
                        constraint_class=type(constraint),
                        constraint_type=constraint.CONSTRAINT_TYPE,
                        message=error,
                        fields=fields,
                        group_name="wired",
                    )
                )

        if violations:
            raise ConstraintsError(violations)

        return self  # type: ignore[return-value]
