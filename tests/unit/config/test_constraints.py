# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the declarative constraint annotations module."""

from typing import Annotated

import pytest
from pydantic import BaseModel, Field, ValidationError

from aiperf.config.constraints import (
    AllOrNone,
    AtLeastOne,
    AtLeastOneTruthy,
    AtLeastOneUnless,
    CompareTo,
    ComparisonOp,
    ConflictsWith,
    ConstraintGroup,
    Constraints,
    ConstraintsError,
    ConstraintsMixin,
    ConstraintType,
    ConstraintViolation,
    ExactlyOne,
    ExactlyOneTruthy,
    ExclusiveGroup,
    F,
    FieldCountRange,
    ForbiddenWith,
    ForbiddenWithIn,
    Forbids,
    ForbidsIfTruthy,
    GreaterThan,
    GreaterThanOrEqual,
    GroupConflictsWith,
    LessThan,
    LessThanOrEqual,
    Matches,
    MaxFieldsSet,
    MinFieldsSet,
    NotEqualTo,
    RequiredIf,
    RequiredIfIn,
    Requires,
    RequiresAllOf,
    RequiresAnyOf,
    RequiresIfTruthy,
    TruthyExclusiveGroup,
    _format_field_list,
    _format_field_name,
    _get_nested_value,
    collect_constraints,
    collect_field_display_names,
    collect_group_fields,
    wired_constraints,
)

# =============================================================================
# TEST MODELS
# =============================================================================


class ExclusiveGroupModel(BaseModel, ConstraintsMixin):
    option_a: Annotated[str | None, ExclusiveGroup("opts")] = None
    option_b: Annotated[str | None, ExclusiveGroup("opts")] = None
    option_c: Annotated[str | None, ExclusiveGroup("opts")] = None


class TruthyExclusiveGroupModel(BaseModel, ConstraintsMixin):
    use_ssl: Annotated[bool, TruthyExclusiveGroup("mode")] = False
    use_plaintext: Annotated[bool, TruthyExclusiveGroup("mode")] = False


class AtLeastOneModel(BaseModel, ConstraintsMixin):
    username: Annotated[str | None, AtLeastOne("auth")] = None
    token: Annotated[str | None, AtLeastOne("auth")] = None


class AtLeastOneTruthyModel(BaseModel, ConstraintsMixin):
    name: Annotated[str, AtLeastOneTruthy("identity")] = ""
    id: Annotated[int, AtLeastOneTruthy("identity")] = 0


class AllOrNoneModel(BaseModel, ConstraintsMixin):
    ssl_cert: Annotated[str | None, AllOrNone("ssl")] = None
    ssl_key: Annotated[str | None, AllOrNone("ssl")] = None
    ssl_ca: Annotated[str | None, AllOrNone("ssl")] = None


class RequiresModel(BaseModel, ConstraintsMixin):
    cert_path: Annotated[str | None, Requires("key_path")] = None
    key_path: str | None = None


class RequiresIfTruthyModel(BaseModel, ConstraintsMixin):
    enable_ssl: Annotated[bool, RequiresIfTruthy("cert_path")] = False
    cert_path: str | None = None


class RequiresAllOfModel(BaseModel, ConstraintsMixin):
    enable_auth: Annotated[str | None, RequiresAllOf(["username", "password"])] = None
    username: str | None = None
    password: str | None = None


class CompareToModel(BaseModel, ConstraintsMixin):
    min_val: int = 0
    max_val: Annotated[int, CompareTo("min_val", ComparisonOp.GE)] = 100


class GreaterThanModel(BaseModel, ConstraintsMixin):
    min_val: int = 0
    max_val: Annotated[int, GreaterThan("min_val")] = 100


class GreaterThanOrEqualModel(BaseModel, ConstraintsMixin):
    min_val: int = 0
    max_val: Annotated[int, GreaterThanOrEqual("min_val")] = 100


class LessThanModel(BaseModel, ConstraintsMixin):
    end: int = 100
    start: Annotated[int, LessThan("end")] = 0


class LessThanOrEqualModel(BaseModel, ConstraintsMixin):
    end_offset: int = 100
    start_offset: Annotated[int | None, LessThanOrEqual("end_offset")] = None


class RequiredIfModel(BaseModel, ConstraintsMixin):
    auth_type: str = "none"
    cert: Annotated[str | None, RequiredIf("auth_type", "ssl")] = None


class RequiredIfInModel(BaseModel, ConstraintsMixin):
    phase_type: str = "concurrency"
    rate: Annotated[float | None, RequiredIfIn("phase_type", ["poisson", "gamma"])] = (
        None
    )


class ForbiddenWithModel(BaseModel, ConstraintsMixin):
    phase_type: str = "concurrency"
    rate: Annotated[float | None, ForbiddenWith("phase_type", "concurrency")] = None


class ForbiddenWithInModel(BaseModel, ConstraintsMixin):
    phase_type: str = "poisson"
    fixed_schedule: Annotated[
        bool | None, ForbiddenWithIn("phase_type", ["concurrency", "fixed"])
    ] = None


class AtLeastOneUnlessModel(BaseModel, ConstraintsMixin):
    fixed_schedule: bool = False
    request_count: Annotated[
        int | None, AtLeastOneUnless("stop", "fixed_schedule", True)
    ] = None
    duration: Annotated[
        float | None, AtLeastOneUnless("stop", "fixed_schedule", True)
    ] = None
    num_sessions: Annotated[
        int | None, AtLeastOneUnless("stop", "fixed_schedule", True)
    ] = None


class GroupExclusiveGroupModel(BaseModel, ConstraintsMixin):
    prefix_length: Annotated[
        int | None, GroupConflictsWith("prefix", "user_context")
    ] = None
    num_prefixes: Annotated[
        int | None, GroupConflictsWith("prefix", "user_context")
    ] = None
    user_context_length: Annotated[
        int | None, GroupConflictsWith("user_context", "prefix")
    ] = None
    shared_system: Annotated[
        int | None, GroupConflictsWith("user_context", "prefix")
    ] = None


class MultipleConstraintsModel(BaseModel, ConstraintsMixin):
    mode: str = "basic"
    value: Annotated[
        int | None,
        RequiredIf("mode", "advanced"),
        GreaterThan("min_value"),
    ] = None
    min_value: int = 10


class MultipleGroupsModel(BaseModel, ConstraintsMixin):
    opt_a: Annotated[str | None, ExclusiveGroup("group1")] = None
    opt_b: Annotated[str | None, ExclusiveGroup("group1")] = None
    opt_x: Annotated[str | None, ExclusiveGroup("group2")] = None
    opt_y: Annotated[str | None, ExclusiveGroup("group2")] = None


class ParentModel(BaseModel, ConstraintsMixin):
    parent_a: Annotated[str | None, ExclusiveGroup("parent_opts")] = None
    parent_b: Annotated[str | None, ExclusiveGroup("parent_opts")] = None


class ChildModel(ParentModel):
    child_x: Annotated[str | None, AtLeastOne("child_auth")] = None
    child_y: Annotated[str | None, AtLeastOne("child_auth")] = None


class ExactlyOneModel(BaseModel, ConstraintsMixin):
    output_file: Annotated[str | None, ExactlyOne("output")] = None
    output_stdout: Annotated[bool | None, ExactlyOne("output")] = None
    output_null: Annotated[bool | None, ExactlyOne("output")] = None


class ExactlyOneTruthyModel(BaseModel, ConstraintsMixin):
    use_http: Annotated[bool, ExactlyOneTruthy("protocol")] = False
    use_grpc: Annotated[bool, ExactlyOneTruthy("protocol")] = False
    use_websocket: Annotated[bool, ExactlyOneTruthy("protocol")] = False


class MinFieldsSetModel(BaseModel, ConstraintsMixin):
    name: Annotated[str | None, MinFieldsSet("identity", 2)] = None
    email: Annotated[str | None, MinFieldsSet("identity", 2)] = None
    phone: Annotated[str | None, MinFieldsSet("identity", 2)] = None


class MaxFieldsSetModel(BaseModel, ConstraintsMixin):
    opt_a: Annotated[str | None, MaxFieldsSet("options", 2)] = None
    opt_b: Annotated[str | None, MaxFieldsSet("options", 2)] = None
    opt_c: Annotated[str | None, MaxFieldsSet("options", 2)] = None
    opt_d: Annotated[str | None, MaxFieldsSet("options", 2)] = None


class FieldCountRangeModel(BaseModel, ConstraintsMixin):
    feat_a: Annotated[str | None, FieldCountRange("features", 1, 3)] = None
    feat_b: Annotated[str | None, FieldCountRange("features", 1, 3)] = None
    feat_c: Annotated[str | None, FieldCountRange("features", 1, 3)] = None
    feat_d: Annotated[str | None, FieldCountRange("features", 1, 3)] = None


class RequiresAnyOfModel(BaseModel, ConstraintsMixin):
    enable_notifications: Annotated[
        bool | None, RequiresAnyOf(["email", "phone", "webhook"])
    ] = None
    email: str | None = None
    phone: str | None = None
    webhook: str | None = None


class ForbidsModel(BaseModel, ConstraintsMixin):
    use_cache: Annotated[bool | None, Forbids("cache_bypass")] = None
    cache_bypass: bool | None = None


class ForbidsIfTruthyModel(BaseModel, ConstraintsMixin):
    debug_mode: Annotated[bool, ForbidsIfTruthy("production_key")] = False
    production_key: str | None = None


class MatchesModel(BaseModel, ConstraintsMixin):
    password: str = ""
    password_confirm: Annotated[str, Matches("password")] = ""


class NotEqualToModel(BaseModel, ConstraintsMixin):
    primary_endpoint: str | None = None
    backup_endpoint: Annotated[str | None, NotEqualTo("primary_endpoint")] = None


# =============================================================================
# TESTS: ExclusiveGroup
# =============================================================================


class TestExclusiveGroup:
    def test_none_set_valid(self):
        model = ExclusiveGroupModel()
        assert model.option_a is None

    def test_one_set_valid(self):
        model = ExclusiveGroupModel(option_a="value")
        assert model.option_a == "value"

    def test_two_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ExclusiveGroupModel(option_a="val1", option_b="val2")
        assert "Only one of" in str(exc_info.value)

    def test_all_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ExclusiveGroupModel(option_a="val1", option_b="val2", option_c="val3")
        assert "Only one of" in str(exc_info.value)


# =============================================================================
# TESTS: TruthyExclusiveGroup
# =============================================================================


class TestTruthyExclusiveGroup:
    def test_both_false_valid(self):
        model = TruthyExclusiveGroupModel(use_ssl=False, use_plaintext=False)
        assert model.use_ssl is False

    def test_one_true_valid(self):
        model = TruthyExclusiveGroupModel(use_ssl=True, use_plaintext=False)
        assert model.use_ssl is True

    def test_both_true_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            TruthyExclusiveGroupModel(use_ssl=True, use_plaintext=True)
        assert "Only one of" in str(exc_info.value)
        assert "can be enabled" in str(exc_info.value)


# =============================================================================
# TESTS: AtLeastOne
# =============================================================================


class TestAtLeastOne:
    def test_none_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            AtLeastOneModel()
        assert "At least one of" in str(exc_info.value)

    def test_one_set_valid(self):
        model = AtLeastOneModel(username="user")
        assert model.username == "user"

    def test_other_set_valid(self):
        model = AtLeastOneModel(token="tok123")
        assert model.token == "tok123"

    def test_both_set_valid(self):
        model = AtLeastOneModel(username="user", token="tok123")
        assert model.username == "user"


# =============================================================================
# TESTS: AtLeastOneTruthy
# =============================================================================


class TestAtLeastOneTruthy:
    def test_all_falsy_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            AtLeastOneTruthyModel(name="", id=0)
        assert "must have a non-empty value" in str(exc_info.value)

    def test_one_truthy_valid(self):
        model = AtLeastOneTruthyModel(name="Alice", id=0)
        assert model.name == "Alice"

    def test_other_truthy_valid(self):
        model = AtLeastOneTruthyModel(name="", id=42)
        assert model.id == 42


# =============================================================================
# TESTS: AllOrNone
# =============================================================================


class TestAllOrNone:
    def test_none_set_valid(self):
        model = AllOrNoneModel()
        assert model.ssl_cert is None

    def test_all_set_valid(self):
        model = AllOrNoneModel(ssl_cert="cert", ssl_key="key", ssl_ca="ca")
        assert model.ssl_cert == "cert"

    def test_partial_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            AllOrNoneModel(ssl_cert="cert")
        assert "must all be specified together" in str(exc_info.value)

    def test_two_of_three_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            AllOrNoneModel(ssl_cert="cert", ssl_key="key")
        assert "missing:" in str(exc_info.value)


# =============================================================================
# TESTS: ExactlyOne
# =============================================================================


class TestExactlyOne:
    def test_none_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ExactlyOneModel()
        assert "Exactly one of" in str(exc_info.value)
        assert "none were set" in str(exc_info.value)

    def test_one_set_valid(self):
        model = ExactlyOneModel(output_file="/tmp/out.txt")
        assert model.output_file == "/tmp/out.txt"

    def test_another_one_set_valid(self):
        model = ExactlyOneModel(output_stdout=True)
        assert model.output_stdout is True

    def test_two_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ExactlyOneModel(output_file="/tmp/out.txt", output_stdout=True)
        assert "Exactly one of" in str(exc_info.value)
        assert "Got multiple" in str(exc_info.value)

    def test_all_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ExactlyOneModel(
                output_file="/tmp/out.txt", output_stdout=True, output_null=True
            )
        assert "Exactly one of" in str(exc_info.value)


# =============================================================================
# TESTS: ExactlyOneTruthy
# =============================================================================


class TestExactlyOneTruthy:
    def test_none_truthy_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ExactlyOneTruthyModel(use_http=False, use_grpc=False, use_websocket=False)
        assert "Exactly one of" in str(exc_info.value)
        assert "none were" in str(exc_info.value)

    def test_one_truthy_valid(self):
        model = ExactlyOneTruthyModel(use_http=True, use_grpc=False)
        assert model.use_http is True

    def test_another_one_truthy_valid(self):
        model = ExactlyOneTruthyModel(use_grpc=True)
        assert model.use_grpc is True

    def test_two_truthy_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ExactlyOneTruthyModel(use_http=True, use_grpc=True)
        assert "Exactly one of" in str(exc_info.value)
        assert "Got multiple" in str(exc_info.value)


# =============================================================================
# TESTS: MinFieldsSet
# =============================================================================


class TestMinFieldsSet:
    def test_none_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            MinFieldsSetModel()
        assert "At least 2 of" in str(exc_info.value)
        assert "Only 0 set" in str(exc_info.value)

    def test_one_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            MinFieldsSetModel(name="Alice")
        assert "At least 2 of" in str(exc_info.value)
        assert "Only 1 set" in str(exc_info.value)

    def test_exactly_min_set_valid(self):
        model = MinFieldsSetModel(name="Alice", email="alice@example.com")
        assert model.name == "Alice"
        assert model.email == "alice@example.com"

    def test_more_than_min_set_valid(self):
        model = MinFieldsSetModel(
            name="Alice", email="alice@example.com", phone="555-1234"
        )
        assert model.name == "Alice"


# =============================================================================
# TESTS: MaxFieldsSet
# =============================================================================


class TestMaxFieldsSet:
    def test_none_set_valid(self):
        model = MaxFieldsSetModel()
        assert model.opt_a is None

    def test_one_set_valid(self):
        model = MaxFieldsSetModel(opt_a="a")
        assert model.opt_a == "a"

    def test_exactly_max_set_valid(self):
        model = MaxFieldsSetModel(opt_a="a", opt_b="b")
        assert model.opt_a == "a"

    def test_one_over_max_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            MaxFieldsSetModel(opt_a="a", opt_b="b", opt_c="c")
        assert "At most 2 of" in str(exc_info.value)
        assert "Got 3" in str(exc_info.value)

    def test_all_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            MaxFieldsSetModel(opt_a="a", opt_b="b", opt_c="c", opt_d="d")
        assert "At most 2 of" in str(exc_info.value)


# =============================================================================
# TESTS: FieldCountRange
# =============================================================================


class TestFieldCountRange:
    def test_none_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            FieldCountRangeModel()
        assert "At least 1 of" in str(exc_info.value)

    def test_exactly_min_set_valid(self):
        model = FieldCountRangeModel(feat_a="a")
        assert model.feat_a == "a"

    def test_middle_count_valid(self):
        model = FieldCountRangeModel(feat_a="a", feat_b="b")
        assert model.feat_a == "a"

    def test_exactly_max_set_valid(self):
        model = FieldCountRangeModel(feat_a="a", feat_b="b", feat_c="c")
        assert model.feat_a == "a"

    def test_over_max_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            FieldCountRangeModel(feat_a="a", feat_b="b", feat_c="c", feat_d="d")
        assert "At most 3 of" in str(exc_info.value)
        assert "Got 4" in str(exc_info.value)


# =============================================================================
# TESTS: RequiresAnyOf
# =============================================================================


class TestRequiresAnyOf:
    def test_trigger_not_set_valid(self):
        model = RequiresAnyOfModel()
        assert model.enable_notifications is None

    def test_trigger_set_with_one_required_valid(self):
        model = RequiresAnyOfModel(enable_notifications=True, email="test@example.com")
        assert model.enable_notifications is True

    def test_trigger_set_with_another_required_valid(self):
        model = RequiresAnyOfModel(enable_notifications=True, webhook="http://hook.com")
        assert model.webhook == "http://hook.com"

    def test_trigger_set_with_multiple_required_valid(self):
        model = RequiresAnyOfModel(
            enable_notifications=True, email="test@example.com", phone="555-1234"
        )
        assert model.email == "test@example.com"

    def test_trigger_set_none_required_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            RequiresAnyOfModel(enable_notifications=True)
        assert "at least one of" in str(exc_info.value)


# =============================================================================
# TESTS: Forbids
# =============================================================================


class TestForbidsConstraint:
    def test_neither_set_valid(self):
        model = ForbidsModel()
        assert model.use_cache is None
        assert model.cache_bypass is None

    def test_source_set_forbidden_unset_valid(self):
        model = ForbidsModel(use_cache=True)
        assert model.use_cache is True

    def test_forbidden_set_source_unset_valid(self):
        model = ForbidsModel(cache_bypass=True)
        assert model.cache_bypass is True

    def test_both_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ForbidsModel(use_cache=True, cache_bypass=True)
        assert "cannot be used when" in str(exc_info.value)


# =============================================================================
# TESTS: ForbidsIfTruthy
# =============================================================================


class TestForbidsIfTruthyConstraint:
    def test_source_false_forbidden_set_valid(self):
        model = ForbidsIfTruthyModel(debug_mode=False, production_key="key123")
        assert model.production_key == "key123"

    def test_source_true_forbidden_unset_valid(self):
        model = ForbidsIfTruthyModel(debug_mode=True)
        assert model.debug_mode is True

    def test_source_true_forbidden_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ForbidsIfTruthyModel(debug_mode=True, production_key="key123")
        assert "cannot be used when" in str(exc_info.value)
        assert "True" in str(exc_info.value)


# =============================================================================
# TESTS: Matches
# =============================================================================


class TestMatches:
    def test_both_empty_valid(self):
        model = MatchesModel(password="", password_confirm="")
        assert model.password == ""

    def test_both_same_valid(self):
        model = MatchesModel(password="secret123", password_confirm="secret123")
        assert model.password == "secret123"

    def test_different_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            MatchesModel(password="secret123", password_confirm="different")
        assert "must match" in str(exc_info.value)


# =============================================================================
# TESTS: NotEqualTo
# =============================================================================


class TestNotEqualTo:
    def test_both_none_skipped(self):
        model = NotEqualToModel()
        assert model.primary_endpoint is None

    def test_one_set_skipped(self):
        model = NotEqualToModel(primary_endpoint="http://primary.com")
        assert model.primary_endpoint == "http://primary.com"

    def test_different_values_valid(self):
        model = NotEqualToModel(
            primary_endpoint="http://primary.com",
            backup_endpoint="http://backup.com",
        )
        assert model.primary_endpoint == "http://primary.com"
        assert model.backup_endpoint == "http://backup.com"

    def test_same_values_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            NotEqualToModel(
                primary_endpoint="http://same.com",
                backup_endpoint="http://same.com",
            )
        assert "must be different from" in str(exc_info.value)


# =============================================================================
# TESTS: Requires
# =============================================================================


class TestRequires:
    def test_neither_set_valid(self):
        model = RequiresModel()
        assert model.cert_path is None

    def test_both_set_valid(self):
        model = RequiresModel(cert_path="/cert", key_path="/key")
        assert model.cert_path == "/cert"

    def test_only_required_set_valid(self):
        model = RequiresModel(key_path="/key")
        assert model.key_path == "/key"

    def test_source_without_required_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            RequiresModel(cert_path="/cert")
        assert "'key_path' is required" in str(exc_info.value)


# =============================================================================
# TESTS: RequiresIfTruthy
# =============================================================================


class TestRequiresIfTruthy:
    def test_false_no_cert_valid(self):
        model = RequiresIfTruthyModel(enable_ssl=False)
        assert model.enable_ssl is False

    def test_true_with_cert_valid(self):
        model = RequiresIfTruthyModel(enable_ssl=True, cert_path="/cert")
        assert model.enable_ssl is True

    def test_true_without_cert_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            RequiresIfTruthyModel(enable_ssl=True)
        assert "'cert_path' is required" in str(exc_info.value)
        assert "True" in str(exc_info.value)


# =============================================================================
# TESTS: RequiresAllOf
# =============================================================================


class TestRequiresAllOf:
    def test_trigger_not_set_valid(self):
        model = RequiresAllOfModel()
        assert model.enable_auth is None

    def test_trigger_set_all_required_valid(self):
        model = RequiresAllOfModel(enable_auth="yes", username="user", password="pass")
        assert model.enable_auth == "yes"

    def test_trigger_set_missing_one_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            RequiresAllOfModel(enable_auth="yes", username="user")
        assert "Missing:" in str(exc_info.value)
        assert "password" in str(exc_info.value)

    def test_trigger_set_missing_all_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            RequiresAllOfModel(enable_auth="yes")
        assert "also required" in str(exc_info.value)


# =============================================================================
# TESTS: CompareTo
# =============================================================================


class TestCompareTo:
    def test_greater_than_valid(self):
        model = CompareToModel(min_val=10, max_val=20)
        assert model.max_val == 20

    def test_greater_than_equal_invalid(self):
        # GE means max_val >= min_val, so max_val=10, min_val=10 is valid
        model = CompareToModel(min_val=10, max_val=10)
        assert model.max_val == 10

    def test_greater_than_less_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            CompareToModel(min_val=10, max_val=5)
        assert "must be greater than or equal to" in str(exc_info.value)


# =============================================================================
# TESTS: GreaterThanOrEqual
# =============================================================================


class TestGreaterThanOrEqual:
    def test_greater_valid(self):
        model = GreaterThanOrEqualModel(min_val=10, max_val=20)
        assert model.max_val == 20

    def test_equal_valid(self):
        model = GreaterThanOrEqualModel(min_val=10, max_val=10)
        assert model.max_val == 10

    def test_less_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            GreaterThanOrEqualModel(min_val=10, max_val=5)
        assert "must be greater than or equal to" in str(exc_info.value)


# =============================================================================
# TESTS: LessThan
# =============================================================================


class TestLessThan:
    def test_less_than_valid(self):
        model = LessThanModel(start=10, end=100)
        assert model.start == 10

    def test_less_than_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            LessThanModel(start=100, end=50)
        assert "must be less than" in str(exc_info.value)


# =============================================================================
# TESTS: LessThanOrEqual
# =============================================================================


class TestLessThanOrEqual:
    def test_less_valid(self):
        model = LessThanOrEqualModel(start_offset=10, end_offset=100)
        assert model.start_offset == 10

    def test_equal_valid(self):
        model = LessThanOrEqualModel(start_offset=100, end_offset=100)
        assert model.start_offset == 100

    def test_greater_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            LessThanOrEqualModel(start_offset=150, end_offset=100)
        assert "must be less than or equal to" in str(exc_info.value)

    def test_none_skipped(self):
        model = LessThanOrEqualModel(start_offset=None, end_offset=100)
        assert model.start_offset is None


# =============================================================================
# TESTS: CompareToFull
# =============================================================================


class TestCompareToFull:
    def test_compare_to_ge_valid(self):
        model = CompareToModel(min_val=5, max_val=10)
        assert model.max_val == 10

    def test_compare_to_ge_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            CompareToModel(min_val=20, max_val=10)
        assert "must be greater than or equal to" in str(exc_info.value)


# =============================================================================
# TESTS: RequiredIf
# =============================================================================


class TestRequiredIf:
    def test_condition_not_met_valid(self):
        model = RequiredIfModel(auth_type="none")
        assert model.cert is None

    def test_condition_met_with_value_valid(self):
        model = RequiredIfModel(auth_type="ssl", cert="/path/to/cert")
        assert model.cert == "/path/to/cert"

    def test_condition_met_without_value_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            RequiredIfModel(auth_type="ssl")
        assert "'cert' is required" in str(exc_info.value)
        assert "'ssl'" in str(exc_info.value)


# =============================================================================
# TESTS: RequiredIfIn
# =============================================================================


class TestRequiredIfIn:
    def test_trigger_not_in_values_valid(self):
        model = RequiredIfInModel(phase_type="concurrency")
        assert model.rate is None

    def test_trigger_in_values_with_field_valid(self):
        model = RequiredIfInModel(phase_type="poisson", rate=1.5)
        assert model.rate == 1.5

    def test_trigger_in_values_without_field_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            RequiredIfInModel(phase_type="gamma")
        assert "'rate' is required" in str(exc_info.value)


# =============================================================================
# TESTS: ForbiddenWith
# =============================================================================


class TestForbiddenWith:
    def test_trigger_not_met_with_field_valid(self):
        model = ForbiddenWithModel(phase_type="poisson", rate=1.5)
        assert model.rate == 1.5

    def test_trigger_met_without_field_valid(self):
        model = ForbiddenWithModel(phase_type="concurrency")
        assert model.rate is None

    def test_trigger_met_with_field_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ForbiddenWithModel(phase_type="concurrency", rate=1.5)
        assert "'rate' cannot be used" in str(exc_info.value)


# =============================================================================
# TESTS: ForbiddenWithIn
# =============================================================================


class TestForbiddenWithIn:
    def test_trigger_not_in_values_with_field_valid(self):
        model = ForbiddenWithInModel(phase_type="poisson", fixed_schedule=True)
        assert model.fixed_schedule is True

    def test_trigger_in_values_without_field_valid(self):
        model = ForbiddenWithInModel(phase_type="concurrency")
        assert model.fixed_schedule is None

    def test_trigger_in_values_with_field_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ForbiddenWithInModel(phase_type="fixed", fixed_schedule=True)
        assert "'fixed_schedule' cannot be used" in str(exc_info.value)


# =============================================================================
# TESTS: AtLeastOneUnless
# =============================================================================


class TestAtLeastOneUnless:
    def test_unless_condition_met_none_set_valid(self):
        model = AtLeastOneUnlessModel(fixed_schedule=True)
        assert model.request_count is None

    def test_unless_condition_not_met_one_set_valid(self):
        model = AtLeastOneUnlessModel(fixed_schedule=False, request_count=100)
        assert model.request_count == 100

    def test_unless_condition_not_met_none_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            AtLeastOneUnlessModel(fixed_schedule=False)
        assert "At least one of" in str(exc_info.value)
        assert "is not True" in str(exc_info.value)

    def test_unless_condition_not_met_multiple_set_valid(self):
        model = AtLeastOneUnlessModel(
            fixed_schedule=False, request_count=100, duration=60.0
        )
        assert model.request_count == 100


# =============================================================================
# TESTS: GroupConflictsWith
# =============================================================================


class TestGroupConflictsWith:
    def test_neither_group_set_valid(self):
        model = GroupExclusiveGroupModel()
        assert model.prefix_length is None

    def test_only_first_group_set_valid(self):
        model = GroupExclusiveGroupModel(prefix_length=100, num_prefixes=5)
        assert model.prefix_length == 100

    def test_only_second_group_set_valid(self):
        model = GroupExclusiveGroupModel(user_context_length=200, shared_system=50)
        assert model.user_context_length == 200

    def test_both_groups_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            GroupExclusiveGroupModel(prefix_length=100, user_context_length=200)
        assert "Cannot use" in str(exc_info.value)
        assert "mutually exclusive" in str(exc_info.value)

    def test_multiple_from_each_group_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            GroupExclusiveGroupModel(
                prefix_length=100, num_prefixes=5, user_context_length=200
            )
        assert "Cannot use" in str(exc_info.value)


# =============================================================================
# TESTS: Multiple Constraints
# =============================================================================


class TestMultipleConstraints:
    def test_basic_mode_no_value_valid(self):
        model = MultipleConstraintsModel(mode="basic")
        assert model.value is None

    def test_advanced_mode_with_valid_value(self):
        model = MultipleConstraintsModel(mode="advanced", value=50, min_value=10)
        assert model.value == 50

    def test_advanced_mode_without_value_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            MultipleConstraintsModel(mode="advanced")
        assert "'value' is required" in str(exc_info.value)

    def test_advanced_mode_with_low_value_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            MultipleConstraintsModel(mode="advanced", value=5, min_value=10)
        assert "must be greater than" in str(exc_info.value)


# =============================================================================
# TESTS: Multiple Groups
# =============================================================================


class TestMultipleGroups:
    def test_one_from_each_group_valid(self):
        model = MultipleGroupsModel(opt_a="a", opt_x="x")
        assert model.opt_a == "a"

    def test_two_from_same_group_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            MultipleGroupsModel(opt_a="a", opt_b="b")
        assert "Only one of" in str(exc_info.value)


# =============================================================================
# TESTS: Inheritance
# =============================================================================


class TestInheritance:
    def test_child_inherits_parent_constraints(self):
        with pytest.raises(ValidationError) as exc_info:
            ChildModel(parent_a="a", parent_b="b", child_x="x")
        assert "Only one of" in str(exc_info.value)

    def test_child_has_own_constraints(self):
        with pytest.raises(ValidationError) as exc_info:
            ChildModel(parent_a="a")
        assert "At least one of" in str(exc_info.value)

    def test_child_valid_with_both_satisfied(self):
        model = ChildModel(parent_a="a", child_x="x")
        assert model.parent_a == "a"


# =============================================================================
# TESTS: collect_constraints
# =============================================================================


class TestCollectConstraints:
    def test_collects_mutually_exclusive(self):
        constraints = collect_constraints(ExclusiveGroupModel)
        assert ConstraintType.MUTUALLY_EXCLUSIVE in constraints
        groups = constraints[ConstraintType.MUTUALLY_EXCLUSIVE]
        assert len(groups) == 1
        assert set(groups[0].fields) == {"option_a", "option_b", "option_c"}

    def test_collects_at_least_one(self):
        constraints = collect_constraints(AtLeastOneModel)
        assert ConstraintType.AT_LEAST_ONE in constraints

    def test_collects_requires_field(self):
        constraints = collect_constraints(RequiresModel)
        assert ConstraintType.REQUIRES_FIELD in constraints
        groups = constraints[ConstraintType.REQUIRES_FIELD]
        assert len(groups) == 1
        assert groups[0].fields == ("cert_path",)

    def test_collects_from_parent(self):
        constraints = collect_constraints(ChildModel)
        assert ConstraintType.MUTUALLY_EXCLUSIVE in constraints
        assert ConstraintType.AT_LEAST_ONE in constraints


# =============================================================================
# TESTS: ConstraintGroup
# =============================================================================


class TestConstraintGroup:
    def test_creation(self):
        constraint = Requires("key_path")
        group = ConstraintGroup(
            constraint_type=ConstraintType.REQUIRES_FIELD,
            group_name="cert_path",
            fields=("cert_path",),
            constraint=constraint,
        )
        assert group.constraint_type == ConstraintType.REQUIRES_FIELD
        assert group.fields == ("cert_path",)
        assert group.constraint is constraint

    def test_frozen(self):
        group = ConstraintGroup(
            constraint_type=ConstraintType.MUTUALLY_EXCLUSIVE,
            group_name="opts",
            fields=("a", "b"),
            constraint=ExclusiveGroup("opts"),
        )
        with pytest.raises(AttributeError):
            group.fields = ("c",)  # type: ignore


# =============================================================================
# TESTS: Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_comparison_with_none_skipped(self):
        model = LessThanOrEqualModel(start_offset=None)
        assert model.start_offset is None

    def test_empty_string_is_set(self):
        # Empty string is "set" (not None) but not truthy
        class EmptyStringModel(BaseModel, ConstraintsMixin):
            val: Annotated[str | None, ExclusiveGroup("g")] = None
            other: Annotated[str | None, ExclusiveGroup("g")] = None

        # Both set to empty strings should fail ExclusiveGroup
        with pytest.raises(ValidationError):
            EmptyStringModel(val="", other="")

    def test_requires_all_of_with_tuple(self):
        class TupleModel(BaseModel, ConstraintsMixin):
            trigger: Annotated[str | None, RequiresAllOf(("a", "b"))] = None
            a: str | None = None
            b: str | None = None

        model = TupleModel(trigger="yes", a="x", b="y")
        assert model.trigger == "yes"

    def test_conditional_requirement_in_with_tuple(self):
        class TupleModel(BaseModel, ConstraintsMixin):
            mode: str = "a"
            val: Annotated[str | None, RequiredIfIn("mode", ("x", "y"))] = None

        model = TupleModel(mode="a")
        assert model.val is None


# =============================================================================
# TESTS: ConstraintType Enum
# =============================================================================


class TestConstraintTypeEnum:
    def test_all_constraint_classes_have_validate_method(self):
        """Verify all constraint classes implement the validate method."""
        from aiperf.config.constraints import BaseConstraint

        constraint_classes = [
            ConflictsWith,
            ExclusiveGroup,
            TruthyExclusiveGroup,
            AtLeastOne,
            AtLeastOneTruthy,
            AllOrNone,
            ExactlyOne,
            ExactlyOneTruthy,
            MinFieldsSet,
            MaxFieldsSet,
            FieldCountRange,
            AtLeastOneUnless,
            GroupConflictsWith,
            Requires,
            RequiresIfTruthy,
            RequiresAllOf,
            RequiresAnyOf,
            Forbids,
            ForbidsIfTruthy,
            CompareTo,
            GreaterThan,
            GreaterThanOrEqual,
            LessThan,
            LessThanOrEqual,
            Matches,
            NotEqualTo,
            RequiredIf,
            RequiredIfIn,
            ForbiddenWith,
            ForbiddenWithIn,
        ]

        for cls in constraint_classes:
            assert issubclass(cls, BaseConstraint), (
                f"{cls.__name__} should inherit from BaseConstraint"
            )
            assert hasattr(cls, "validate"), (
                f"{cls.__name__} must have a validate method"
            )

    def test_case_insensitive(self):
        assert ConstraintType("MUTUALLY_EXCLUSIVE") == ConstraintType.MUTUALLY_EXCLUSIVE
        assert ConstraintType("mutually_exclusive") == ConstraintType.MUTUALLY_EXCLUSIVE


# =============================================================================
# TESTS: Constraints Shorthand
# =============================================================================


class ShorthandComparisonModel(BaseModel, ConstraintsMixin):
    min_val: int = 0
    max_val: Annotated[int, Constraints(gt="min_val")] = 100
    mid_val: Annotated[int, Constraints(gte="min_val", lte="max_val")] = 50


class ShorthandConflictsModel(BaseModel, ConstraintsMixin):
    opt_a: Annotated[str | None, Constraints(exclusive_group="opts")] = None
    opt_b: Annotated[str | None, Constraints(exclusive_group="opts")] = None


class ShorthandRequiresModel(BaseModel, ConstraintsMixin):
    cert: Annotated[str | None, Constraints(requires="key")] = None
    key: str | None = None


class ShorthandForbidsModel(BaseModel, ConstraintsMixin):
    use_cache: Annotated[bool | None, Constraints(forbids="bypass")] = None
    bypass: bool | None = None


class ShorthandConditionalModel(BaseModel, ConstraintsMixin):
    mode: str = "basic"
    advanced_option: Annotated[
        str | None, Constraints(required_if=("mode", "advanced"))
    ] = None


class ShorthandMultipleModel(BaseModel, ConstraintsMixin):
    min_val: int = 0
    mode: str = "basic"
    value: Annotated[
        int | None,
        Constraints(gt="min_val", required_if=("mode", "advanced")),
    ] = None


class ShorthandCountModel(BaseModel, ConstraintsMixin):
    a: Annotated[str | None, Constraints(min_set=("opts", 2))] = None
    b: Annotated[str | None, Constraints(min_set=("opts", 2))] = None
    c: Annotated[str | None, Constraints(min_set=("opts", 2))] = None


class TestConstraintsShorthandComparison:
    def test_gt_valid(self):
        model = ShorthandComparisonModel(min_val=10, max_val=20, mid_val=15)
        assert model.max_val == 20

    def test_gt_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ShorthandComparisonModel(min_val=20, max_val=10, mid_val=15)
        assert "must be greater than" in str(exc_info.value)

    def test_multiple_comparisons_valid(self):
        model = ShorthandComparisonModel(min_val=0, mid_val=50, max_val=100)
        assert model.mid_val == 50

    def test_multiple_comparisons_invalid_lower(self):
        with pytest.raises(ValidationError) as exc_info:
            ShorthandComparisonModel(min_val=60, mid_val=50, max_val=100)
        assert "must be greater than or equal to" in str(exc_info.value)

    def test_multiple_comparisons_invalid_upper(self):
        with pytest.raises(ValidationError) as exc_info:
            ShorthandComparisonModel(min_val=0, mid_val=150, max_val=100)
        assert "must be less than or equal to" in str(exc_info.value)


class TestConstraintsShorthandConflicts:
    def test_none_set_valid(self):
        model = ShorthandConflictsModel()
        assert model.opt_a is None

    def test_one_set_valid(self):
        model = ShorthandConflictsModel(opt_a="value")
        assert model.opt_a == "value"

    def test_both_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ShorthandConflictsModel(opt_a="a", opt_b="b")
        assert "Only one of" in str(exc_info.value)


class TestConstraintsShorthandRequires:
    def test_neither_set_valid(self):
        model = ShorthandRequiresModel()
        assert model.cert is None

    def test_both_set_valid(self):
        model = ShorthandRequiresModel(cert="/path/cert", key="/path/key")
        assert model.cert == "/path/cert"

    def test_source_without_required_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ShorthandRequiresModel(cert="/path/cert")
        assert "'key' is required" in str(exc_info.value)


class TestConstraintsShorthandForbids:
    def test_neither_set_valid(self):
        model = ShorthandForbidsModel()
        assert model.use_cache is None

    def test_source_only_valid(self):
        model = ShorthandForbidsModel(use_cache=True)
        assert model.use_cache is True

    def test_both_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ShorthandForbidsModel(use_cache=True, bypass=True)
        assert "cannot be used when" in str(exc_info.value)


class TestConstraintsShorthandConditional:
    def test_condition_not_met_valid(self):
        model = ShorthandConditionalModel(mode="basic")
        assert model.advanced_option is None

    def test_condition_met_with_value_valid(self):
        model = ShorthandConditionalModel(mode="advanced", advanced_option="enabled")
        assert model.advanced_option == "enabled"

    def test_condition_met_without_value_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ShorthandConditionalModel(mode="advanced")
        assert "'advanced_option' is required" in str(exc_info.value)


class TestConstraintsShorthandMultiple:
    def test_basic_mode_no_value_valid(self):
        model = ShorthandMultipleModel(mode="basic")
        assert model.value is None

    def test_advanced_with_valid_value(self):
        model = ShorthandMultipleModel(mode="advanced", value=50, min_val=10)
        assert model.value == 50

    def test_advanced_without_value_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ShorthandMultipleModel(mode="advanced")
        assert "'value' is required" in str(exc_info.value)

    def test_advanced_with_low_value_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ShorthandMultipleModel(mode="advanced", value=5, min_val=10)
        assert "must be greater than" in str(exc_info.value)


class TestConstraintsShorthandCount:
    def test_none_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ShorthandCountModel()
        assert "At least 2 of" in str(exc_info.value)

    def test_one_set_invalid(self):
        with pytest.raises(ValidationError) as exc_info:
            ShorthandCountModel(a="value")
        assert "At least 2 of" in str(exc_info.value)

    def test_two_set_valid(self):
        model = ShorthandCountModel(a="a", b="b")
        assert model.a == "a"

    def test_all_set_valid(self):
        model = ShorthandCountModel(a="a", b="b", c="c")
        assert model.a == "a"


class TestConstraintsDataclass:
    def test_constraints_generates_correct_objects(self):
        c = Constraints(gt="min", exclusive_group="group")
        assert len(c._constraints) == 2
        assert isinstance(c._constraints[0], GreaterThan)
        assert isinstance(c._constraints[1], ExclusiveGroup)

    def test_constraints_is_frozen(self):
        c = Constraints(gt="min")
        with pytest.raises(AttributeError):
            c.gt = "other"  # type: ignore

    def test_empty_constraints_has_no_objects(self):
        c = Constraints()
        assert len(c._constraints) == 0

    def test_all_comparison_shorthands(self):
        c = Constraints(gt="a", gte="b", lt="c", lte="d", eq="e", ne="f")
        assert len(c._constraints) == 6
        assert isinstance(c._constraints[0], GreaterThan)
        assert isinstance(c._constraints[1], GreaterThanOrEqual)
        assert isinstance(c._constraints[2], LessThan)
        assert isinstance(c._constraints[3], LessThanOrEqual)
        assert isinstance(c._constraints[4], Matches)
        assert isinstance(c._constraints[5], NotEqualTo)

    def test_all_dependency_shorthands(self):
        c = Constraints(
            requires="a",
            requires_truthy="b",
            requires_all=("c", "d"),
            requires_any=("e", "f"),
            forbids="g",
            forbids_truthy="h",
        )
        assert len(c._constraints) == 6

    def test_all_group_shorthands(self):
        c = Constraints(
            exclusive_group="g1",
            exclusive_group_nonempty="g2",
            one_of="g3",
            one_of_truthy="g4",
            exactly_one="g5",
            exactly_one_truthy="g6",
            all_or_none="g7",
        )
        assert len(c._constraints) == 7

    def test_count_shorthands(self):
        c = Constraints(
            min_set=("group1", 2),
            max_set=("group2", 3),
            count_range=("group3", 1, 4),
        )
        assert len(c._constraints) == 3

    def test_conflicting_groups_shorthand(self):
        c = Constraints(conflicting_groups=("prefix", "context"))
        assert len(c._constraints) == 1
        assert isinstance(c._constraints[0], GroupConflictsWith)

    def test_conditional_shorthands(self):
        c = Constraints(
            one_of_unless=("stop", "fixed_schedule", True),
            required_if=("mode", "advanced"),
            required_if_in=("phase", ("poisson", "gamma")),
            forbidden_if=("mode", "disabled"),
            forbidden_if_in=("phase", ("fixed", "concurrent")),
        )
        assert len(c._constraints) == 5


# =============================================================================
# TESTS: Additional Shorthand Usage
# =============================================================================


class ShorthandAllGroupModel(BaseModel, ConstraintsMixin):
    a: Annotated[str | None, Constraints(one_of="grp")] = None
    b: Annotated[str | None, Constraints(one_of="grp")] = None


class ShorthandOneTruthyModel(BaseModel, ConstraintsMixin):
    a: Annotated[bool, Constraints(one_of_truthy="flags")] = False
    b: Annotated[bool, Constraints(one_of_truthy="flags")] = False


class ShorthandExactlyOneModel(BaseModel, ConstraintsMixin):
    a: Annotated[str | None, Constraints(exactly_one="pick")] = None
    b: Annotated[str | None, Constraints(exactly_one="pick")] = None


class ShorthandAllOrNoneModel(BaseModel, ConstraintsMixin):
    cert: Annotated[str | None, Constraints(all_or_none="ssl")] = None
    key: Annotated[str | None, Constraints(all_or_none="ssl")] = None


class ShorthandMaxSetModel(BaseModel, ConstraintsMixin):
    a: Annotated[str | None, Constraints(max_set=("opts", 1))] = None
    b: Annotated[str | None, Constraints(max_set=("opts", 1))] = None


class ShorthandCountRangeModel(BaseModel, ConstraintsMixin):
    a: Annotated[str | None, Constraints(count_range=("opts", 1, 2))] = None
    b: Annotated[str | None, Constraints(count_range=("opts", 1, 2))] = None
    c: Annotated[str | None, Constraints(count_range=("opts", 1, 2))] = None


class ShorthandConflictingGroupsModel(BaseModel, ConstraintsMixin):
    prefix_a: Annotated[
        str | None, Constraints(conflicting_groups=("prefix", "ctx"))
    ] = None
    prefix_b: Annotated[
        str | None, Constraints(conflicting_groups=("prefix", "ctx"))
    ] = None
    ctx_a: Annotated[str | None, Constraints(conflicting_groups=("ctx", "prefix"))] = (
        None
    )


class ShorthandOneOfUnlessModel(BaseModel, ConstraintsMixin):
    fixed: bool = False
    count: Annotated[int | None, Constraints(one_of_unless=("stop", "fixed", True))] = (
        None
    )
    duration: Annotated[
        float | None, Constraints(one_of_unless=("stop", "fixed", True))
    ] = None


class ShorthandForbiddenIfModel(BaseModel, ConstraintsMixin):
    mode: str = "enabled"
    option: Annotated[str | None, Constraints(forbidden_if=("mode", "disabled"))] = None


class ShorthandRequiredIfInModel(BaseModel, ConstraintsMixin):
    phase: str = "fixed"
    rate: Annotated[
        float | None, Constraints(required_if_in=("phase", ("poisson", "gamma")))
    ] = None


class ShorthandForbiddenIfInModel(BaseModel, ConstraintsMixin):
    phase: str = "poisson"
    fixed_count: Annotated[
        int | None, Constraints(forbidden_if_in=("phase", ("poisson", "gamma")))
    ] = None


class ShorthandMatchesModel(BaseModel, ConstraintsMixin):
    password: str | None = None
    confirm: Annotated[str | None, Constraints(eq="password")] = None


class TestShorthandOneOf:
    def test_none_set_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandAllGroupModel()

    def test_one_set_valid(self):
        model = ShorthandAllGroupModel(a="val")
        assert model.a == "val"


class TestShorthandOneTruthy:
    def test_none_truthy_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandOneTruthyModel()

    def test_one_truthy_valid(self):
        model = ShorthandOneTruthyModel(a=True)
        assert model.a is True


class TestShorthandExactlyOne:
    def test_none_set_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandExactlyOneModel()

    def test_one_set_valid(self):
        model = ShorthandExactlyOneModel(a="val")
        assert model.a == "val"

    def test_both_set_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandExactlyOneModel(a="a", b="b")


class TestShorthandAllOrNone:
    def test_none_set_valid(self):
        model = ShorthandAllOrNoneModel()
        assert model.cert is None

    def test_all_set_valid(self):
        model = ShorthandAllOrNoneModel(cert="cert", key="key")
        assert model.cert == "cert"

    def test_partial_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandAllOrNoneModel(cert="cert")


class TestShorthandMaxSet:
    def test_none_set_valid(self):
        model = ShorthandMaxSetModel()
        assert model.a is None

    def test_one_set_valid(self):
        model = ShorthandMaxSetModel(a="val")
        assert model.a == "val"

    def test_both_set_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandMaxSetModel(a="a", b="b")


class TestShorthandCountRange:
    def test_none_set_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandCountRangeModel()

    def test_one_set_valid(self):
        model = ShorthandCountRangeModel(a="val")
        assert model.a == "val"

    def test_two_set_valid(self):
        model = ShorthandCountRangeModel(a="a", b="b")
        assert model.a == "a"

    def test_three_set_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandCountRangeModel(a="a", b="b", c="c")


class TestShorthandConflictingGroups:
    def test_one_group_set_valid(self):
        model = ShorthandConflictingGroupsModel(prefix_a="val")
        assert model.prefix_a == "val"

    def test_both_groups_set_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandConflictingGroupsModel(prefix_a="p", ctx_a="c")


class TestShorthandOneOfUnless:
    def test_unless_met_none_set_valid(self):
        model = ShorthandOneOfUnlessModel(fixed=True)
        assert model.count is None

    def test_unless_not_met_one_set_valid(self):
        model = ShorthandOneOfUnlessModel(fixed=False, count=100)
        assert model.count == 100

    def test_unless_not_met_none_set_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandOneOfUnlessModel(fixed=False)


class TestShorthandForbiddenIf:
    def test_condition_not_met_with_value_valid(self):
        model = ShorthandForbiddenIfModel(mode="enabled", option="val")
        assert model.option == "val"

    def test_condition_met_with_value_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandForbiddenIfModel(mode="disabled", option="val")


class TestShorthandRequiredIfIn:
    def test_not_in_values_no_field_valid(self):
        model = ShorthandRequiredIfInModel(phase="fixed")
        assert model.rate is None

    def test_in_values_with_field_valid(self):
        model = ShorthandRequiredIfInModel(phase="poisson", rate=1.5)
        assert model.rate == 1.5

    def test_in_values_no_field_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandRequiredIfInModel(phase="gamma")


class TestShorthandForbiddenIfIn:
    def test_not_in_values_with_field_valid(self):
        model = ShorthandForbiddenIfInModel(phase="fixed", fixed_count=100)
        assert model.fixed_count == 100

    def test_in_values_with_field_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandForbiddenIfInModel(phase="poisson", fixed_count=100)


class TestShorthandMatches:
    def test_both_none_valid(self):
        model = ShorthandMatchesModel()
        assert model.password is None

    def test_one_none_skipped(self):
        model = ShorthandMatchesModel(password="secret")
        assert model.password == "secret"

    def test_both_same_valid(self):
        model = ShorthandMatchesModel(password="secret", confirm="secret")
        assert model.confirm == "secret"

    def test_different_invalid(self):
        with pytest.raises(ValidationError):
            ShorthandMatchesModel(password="secret", confirm="wrong")


# =============================================================================
# TESTS: CLI Parameter Display Names
# =============================================================================


# Mock CLI Parameter class (simulates cyclopts.Parameter)
class MockCLIParameter:
    """Mock CLI parameter class for testing display name extraction."""

    def __init__(self, *names: str):
        self.name = names


# Test models with CLI parameters
class CLIParameterModel(BaseModel, ConstraintsMixin):
    """Model with CLI parameter annotations for display name testing."""

    request_count: Annotated[
        int | None,
        MockCLIParameter("--request-count", "-n"),
        Constraints(exclusive_group="stop_mode"),
    ] = None
    num_sessions: Annotated[
        int | None,
        MockCLIParameter("--conversation-num"),
        Constraints(exclusive_group="stop_mode"),
    ] = None
    # Field without CLI parameter
    internal_value: Annotated[str | None, Constraints(exclusive_group="stop_mode")] = (
        None
    )


class CLIComparisonModel(BaseModel, ConstraintsMixin):
    """Model for testing comparison constraint error messages with CLI names."""

    min_val: Annotated[int, MockCLIParameter("--min-value")] = 0
    max_val: Annotated[
        int, MockCLIParameter("--max-value"), Constraints(gte="min_val")
    ] = 100


class CLIRequiresModel(BaseModel, ConstraintsMixin):
    """Model for testing requires constraint error messages with CLI names."""

    enable_feature: Annotated[
        bool,
        MockCLIParameter("--enable-feature"),
        Constraints(requires_truthy="config_path"),
    ] = False
    config_path: Annotated[str | None, MockCLIParameter("--config-path")] = None


class CLIAtLeastOneModel(BaseModel, ConstraintsMixin):
    """Model for testing at-least-one constraint error messages with CLI names."""

    username: Annotated[
        str | None, MockCLIParameter("--username"), Constraints(one_of="auth")
    ] = None
    token: Annotated[
        str | None, MockCLIParameter("--token"), Constraints(one_of="auth")
    ] = None


class CLIAllOrNoneModel(BaseModel, ConstraintsMixin):
    """Model for testing all-or-none constraint error messages with CLI names."""

    ssl_cert: Annotated[
        str | None, MockCLIParameter("--ssl-cert"), Constraints(all_or_none="ssl")
    ] = None
    ssl_key: Annotated[
        str | None, MockCLIParameter("--ssl-key"), Constraints(all_or_none="ssl")
    ] = None


class MixedParameterModel(BaseModel, ConstraintsMixin):
    """Model with mix of CLI and non-CLI fields."""

    cli_field: Annotated[
        str | None,
        MockCLIParameter("--cli-field"),
        Constraints(exclusive_group="test_group"),
    ] = None
    plain_field: Annotated[str | None, Constraints(exclusive_group="test_group")] = None


class TestCollectFieldDisplayNames:
    """Tests for collect_field_display_names function."""

    def test_extracts_cli_names(self):
        """Test that CLI parameter names are extracted correctly."""
        display_names = collect_field_display_names(CLIParameterModel)
        assert display_names["request_count"] == "--request-count"
        assert display_names["num_sessions"] == "--conversation-num"

    def test_skips_fields_without_cli_param(self):
        """Test that fields without CLI parameters are not included."""
        display_names = collect_field_display_names(CLIParameterModel)
        assert "internal_value" not in display_names

    def test_extracts_from_comparison_model(self):
        """Test extraction from comparison model."""
        display_names = collect_field_display_names(CLIComparisonModel)
        assert display_names["min_val"] == "--min-value"
        assert display_names["max_val"] == "--max-value"

    def test_extracts_from_requires_model(self):
        """Test extraction from requires model."""
        display_names = collect_field_display_names(CLIRequiresModel)
        assert display_names["enable_feature"] == "--enable-feature"
        assert display_names["config_path"] == "--config-path"

    def test_mixed_model_partial_names(self):
        """Test model with mix of CLI and non-CLI fields."""
        display_names = collect_field_display_names(MixedParameterModel)
        assert display_names["cli_field"] == "--cli-field"
        assert "plain_field" not in display_names


class TestFormatFieldName:
    """Tests for _format_field_name helper function."""

    def test_uses_display_name_when_available(self):
        """Test that CLI name is used when available."""
        display_names = {"request_count": "--request-count"}
        result = _format_field_name("request_count", display_names)
        assert result == "--request-count"

    def test_uses_quoted_field_name_when_no_display_name(self):
        """Test that field name is quoted when no CLI name available."""
        display_names = {}
        result = _format_field_name("some_field", display_names)
        assert result == "'some_field'"

    def test_empty_display_names(self):
        """Test with empty display names dict."""
        result = _format_field_name("field", {})
        assert result == "'field'"


class TestFormatFieldList:
    """Tests for _format_field_list helper function."""

    def test_formats_list_with_cli_names(self):
        """Test formatting a list where all fields have CLI names."""
        display_names = {
            "request_count": "--request-count",
            "num_sessions": "--conversation-num",
        }
        result = _format_field_list(["request_count", "num_sessions"], display_names)
        assert result == "[--request-count, --conversation-num]"

    def test_formats_list_with_mixed_names(self):
        """Test formatting a list with mix of CLI and plain fields."""
        display_names = {"request_count": "--request-count"}
        result = _format_field_list(["request_count", "plain_field"], display_names)
        assert result == "[--request-count, 'plain_field']"

    def test_formats_list_with_no_cli_names(self):
        """Test formatting a list where no fields have CLI names."""
        result = _format_field_list(["field_a", "field_b"], {})
        assert result == "['field_a', 'field_b']"

    def test_formats_tuple(self):
        """Test that tuples work as well as lists."""
        display_names = {"a": "--opt-a", "b": "--opt-b"}
        result = _format_field_list(("a", "b"), display_names)
        assert result == "[--opt-a, --opt-b]"


class TestCLIDisplayNamesInErrorMessages:
    """Tests that error messages use CLI parameter names."""

    def test_conflicts_error_uses_cli_names(self):
        """Test that conflicts error message uses CLI names."""
        with pytest.raises(ValidationError) as exc_info:
            CLIParameterModel(request_count=100, num_sessions=50)
        error_msg = str(exc_info.value)
        assert "--request-count" in error_msg
        assert "--conversation-num" in error_msg
        # Check that field names don't appear in the main error text (they may appear in input_value)
        assert (
            "Only one of [--request-count" in error_msg
        )  # CLI names used, not field names

    def test_conflicts_error_mixed_cli_and_plain(self):
        """Test conflicts error with mix of CLI and plain fields."""
        with pytest.raises(ValidationError) as exc_info:
            MixedParameterModel(cli_field="a", plain_field="b")
        error_msg = str(exc_info.value)
        assert "--cli-field" in error_msg
        assert "'plain_field'" in error_msg  # Plain field should be quoted

    def test_comparison_error_uses_cli_names(self):
        """Test that comparison error message uses CLI names."""
        with pytest.raises(ValidationError) as exc_info:
            CLIComparisonModel(min_val=100, max_val=50)
        error_msg = str(exc_info.value)
        assert "--max-value" in error_msg
        assert "--min-value" in error_msg
        assert "must be greater than or equal to" in error_msg

    def test_requires_truthy_error_uses_cli_names(self):
        """Test that requires_truthy error message uses CLI names."""
        with pytest.raises(ValidationError) as exc_info:
            CLIRequiresModel(enable_feature=True)  # config_path not set
        error_msg = str(exc_info.value)
        assert "--config-path" in error_msg
        assert "--enable-feature" in error_msg
        assert "is required when" in error_msg

    def test_at_least_one_error_uses_cli_names(self):
        """Test that at_least_one error message uses CLI names."""
        with pytest.raises(ValidationError) as exc_info:
            CLIAtLeastOneModel()  # Neither username nor token set
        error_msg = str(exc_info.value)
        assert "--username" in error_msg
        assert "--token" in error_msg
        assert "At least one of" in error_msg

    def test_all_or_none_error_uses_cli_names(self):
        """Test that all_or_none error message uses CLI names."""
        with pytest.raises(ValidationError) as exc_info:
            CLIAllOrNoneModel(ssl_cert="cert.pem")  # ssl_key not set
        error_msg = str(exc_info.value)
        assert "--ssl-cert" in error_msg
        assert "--ssl-key" in error_msg
        assert "must all be specified together" in error_msg


class TestCLIDisplayNamesStored:
    """Tests that _field_display_names is populated on the class."""

    def test_display_names_stored_on_class(self):
        """Test that display names are stored as class variable."""
        assert hasattr(CLIParameterModel, "_field_display_names")
        assert (
            CLIParameterModel._field_display_names["request_count"] == "--request-count"
        )

    def test_display_names_inherited(self):
        """Test that child classes have their own display names."""

        class ChildModel(CLIParameterModel):
            extra_field: Annotated[
                str | None,
                MockCLIParameter("--extra"),
                Constraints(exclusive_group="extra_group"),
            ] = None

        # Child should have parent's display names plus its own
        assert ChildModel._field_display_names["request_count"] == "--request-count"
        assert ChildModel._field_display_names["extra_field"] == "--extra"

    def test_model_without_cli_params_has_empty_display_names(self):
        """Test that models without CLI params have empty display names."""

        class PlainModel(BaseModel, ConstraintsMixin):
            field_a: Annotated[str | None, Constraints(exclusive_group="grp")] = None
            field_b: Annotated[str | None, Constraints(exclusive_group="grp")] = None

        assert PlainModel._field_display_names == {}


class TestConstraintsError:
    """Tests for the ConstraintsError exception class."""

    def test_constraints_error_extends_value_error(self):
        """Test that ConstraintsError is a subclass of ValueError."""
        assert issubclass(ConstraintsError, ValueError)

    def test_constraint_violation_is_frozen(self):
        """Test that ConstraintViolation is immutable."""
        violation = ConstraintViolation(
            constraint_class=ExclusiveGroup,
            constraint_type=ConstraintType.MUTUALLY_EXCLUSIVE,
            message="test message",
            fields=("a", "b"),
            group_name="test_group",
        )

        with pytest.raises(AttributeError):
            violation.message = "changed"  # type: ignore[misc]

    def test_constraints_error_str_single_violation(self):
        """Test __str__ with a single violation."""
        violation = ConstraintViolation(
            constraint_class=ExclusiveGroup,
            constraint_type=ConstraintType.MUTUALLY_EXCLUSIVE,
            message="Only one of ['a', 'b'] can be set",
            fields=("a", "b"),
            group_name="opts",
        )
        error = ConstraintsError([violation])

        assert str(error) == "Only one of ['a', 'b'] can be set"

    def test_constraints_error_str_multiple_violations(self):
        """Test __str__ with multiple violations returns newline-separated."""
        v1 = ConstraintViolation(
            constraint_class=ExclusiveGroup,
            constraint_type=ConstraintType.MUTUALLY_EXCLUSIVE,
            message="First error",
            fields=("a", "b"),
            group_name="opts",
        )
        v2 = ConstraintViolation(
            constraint_class=AtLeastOne,
            constraint_type=ConstraintType.AT_LEAST_ONE,
            message="Second error",
            fields=("c", "d"),
            group_name="required",
        )
        error = ConstraintsError([v1, v2])

        assert str(error) == "First error\nSecond error"

    def test_constraints_error_repr(self):
        """Test __repr__ includes all violation details."""
        violation = ConstraintViolation(
            constraint_class=ExclusiveGroup,
            constraint_type=ConstraintType.MUTUALLY_EXCLUSIVE,
            message="test message",
            fields=("a", "b"),
            group_name="opts",
        )
        error = ConstraintsError([violation])

        repr_str = repr(error)
        assert "ConstraintsError" in repr_str
        assert "ConstraintViolation" in repr_str
        assert "ExclusiveGroup" in repr_str
        assert "violations=" in repr_str
        assert "('a', 'b')" in repr_str

    def test_constraints_error_violations_accessible(self):
        """Test that violations list is accessible."""
        v1 = ConstraintViolation(
            constraint_class=ExclusiveGroup,
            constraint_type=ConstraintType.MUTUALLY_EXCLUSIVE,
            message="First",
            fields=("a",),
            group_name="g1",
        )
        v2 = ConstraintViolation(
            constraint_class=GreaterThan,
            constraint_type=ConstraintType.COMPARE_TO,
            message="Second",
            fields=("b",),
            group_name="b",
        )
        error = ConstraintsError([v1, v2])

        assert len(error.violations) == 2
        assert error.violations[0].constraint_class is ExclusiveGroup
        assert error.violations[1].constraint_class is GreaterThan

    def test_pydantic_wraps_constraints_error_in_validation_error(self):
        """Test that Pydantic wraps ConstraintsError in ValidationError."""

        class ExclusiveModel(BaseModel, ConstraintsMixin):
            opt_a: Annotated[str | None, ExclusiveGroup("opts")] = None
            opt_b: Annotated[str | None, ExclusiveGroup("opts")] = None

        # Pydantic catches ValueError (and subclasses) and wraps in ValidationError
        with pytest.raises(ValidationError) as exc_info:
            ExclusiveModel(opt_a="x", opt_b="y")

        # The error message from ConstraintsError should be in the ValidationError
        error_str = str(exc_info.value)
        assert "Only one of" in error_str
        assert "opt_a" in error_str or "'opt_a'" in error_str

    def test_multiple_violations_in_validation_error(self):
        """Test that multiple constraint violations appear in ValidationError message."""

        class MultiErrorModel(BaseModel, ConstraintsMixin):
            opt_a: Annotated[str | None, ExclusiveGroup("opts")] = None
            opt_b: Annotated[str | None, ExclusiveGroup("opts")] = None
            req_a: Annotated[str | None, AtLeastOne("required")] = None
            req_b: Annotated[str | None, AtLeastOne("required")] = None

        with pytest.raises(ValidationError) as exc_info:
            MultiErrorModel(opt_a="x", opt_b="y")

        error_str = str(exc_info.value)
        # Both error messages should be present
        assert "Only one of" in error_str
        assert "At least one of" in error_str


# =============================================================================
# WIRED CONSTRAINTS TESTS
# =============================================================================


class TestWiredConstraintsDecorator:
    """Tests for the @wired_constraints decorator."""

    def test_decorator_stores_constraints_on_class(self):
        """Test that decorator stores constraints on the class."""

        @wired_constraints(
            ConflictsWith("a", "b"),
            GreaterThan("x", source="y"),
        )
        class Config(BaseModel, ConstraintsMixin):
            pass

        assert hasattr(Config, "_wired_constraints")
        assert len(Config._wired_constraints) == 2

    def test_decorator_preserves_existing_constraints(self):
        """Test that decorator preserves existing wired constraints."""

        @wired_constraints(ConflictsWith("a", "b"))
        class Parent(BaseModel, ConstraintsMixin):
            pass

        @wired_constraints(GreaterThan("x", source="y"))
        class Child(Parent):
            pass

        # Child should have both constraints
        assert len(Child._wired_constraints) == 2


class TestWiredConstraintsDotNotation:
    """Tests for dot-notation path resolution in wired constraints."""

    def test_simple_dot_path(self):
        """Test simple dot-notation path resolution."""

        class Inner(BaseModel):
            value: int = 10

        @wired_constraints(
            GreaterThan("inner.value", source="outer_value"),
        )
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)
            outer_value: int = 20

        # Valid: outer_value (20) > inner.value (10)
        config = Config()
        assert config.outer_value == 20
        assert config.inner.value == 10

    def test_dot_path_validation_fails(self):
        """Test that dot-path constraint validation fails correctly."""

        class Inner(BaseModel):
            value: int = 10

        @wired_constraints(
            GreaterThan("inner.value", source="outer_value"),
        )
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)
            outer_value: int = 20

        # Invalid: outer_value (5) < inner.value (10)
        with pytest.raises(ValidationError) as exc_info:
            Config(outer_value=5)

        assert "outer_value" in str(exc_info.value)
        assert "inner.value" in str(exc_info.value)

    def test_deep_nested_path(self):
        """Test deeply nested dot-notation paths."""

        class Level3(BaseModel):
            value: int = 5

        class Level2(BaseModel):
            level3: Level3 = Field(default_factory=Level3)

        class Level1(BaseModel):
            level2: Level2 = Field(default_factory=Level2)

        @wired_constraints(
            GreaterThan("level1.level2.level3.value", source="top_value"),
        )
        class Config(BaseModel, ConstraintsMixin):
            level1: Level1 = Field(default_factory=Level1)
            top_value: int = 10

        # Valid: top_value (10) > level1.level2.level3.value (5)
        config = Config()
        assert config.top_value == 10

        # Invalid: top_value (3) < level1.level2.level3.value (5)
        with pytest.raises(ValidationError):
            Config(top_value=3)


class TestWiredConstraintsArrayIndexing:
    """Tests for array indexing in wired constraints."""

    def test_array_index_path(self):
        """Test array indexing in paths."""

        class Item(BaseModel):
            value: int = 10

        @wired_constraints(
            GreaterThan("items[0].value", source="max_value"),
        )
        class Config(BaseModel, ConstraintsMixin):
            items: list[Item] = Field(default_factory=lambda: [Item()])
            max_value: int = 20

        # Valid: max_value (20) > items[0].value (10)
        config = Config()
        assert config.max_value == 20

    def test_array_index_validation_fails(self):
        """Test that array index constraint validation fails correctly."""

        class Item(BaseModel):
            value: int = 10

        @wired_constraints(
            GreaterThan("items[0].value", source="max_value"),
        )
        class Config(BaseModel, ConstraintsMixin):
            items: list[Item] = Field(default_factory=lambda: [Item()])
            max_value: int = 20

        # Invalid: max_value (5) < items[0].value (10)
        with pytest.raises(ValidationError) as exc_info:
            Config(max_value=5)

        assert "max_value" in str(exc_info.value)
        assert "items[0].value" in str(exc_info.value)

    def test_multiple_array_indices(self):
        """Test comparing values from different array indices."""

        class Item(BaseModel):
            value: int

        @wired_constraints(
            GreaterThan("items[0].value", source="items[1].value"),
        )
        class Config(BaseModel, ConstraintsMixin):
            items: list[Item]

        # Valid: items[1].value (20) > items[0].value (10)
        config = Config(items=[Item(value=10), Item(value=20)])
        assert config.items[1].value == 20

        # Invalid: items[1].value (5) < items[0].value (10)
        with pytest.raises(ValidationError):
            Config(items=[Item(value=10), Item(value=5)])


class TestWiredConstraintsConflictsWith:
    """Tests for ConflictsWith with wired constraints."""

    def test_conflicts_with_truthy_both_false_valid(self):
        """Test ConflictsWith with truthy=True when both are False."""

        class Load(BaseModel):
            enabled: bool = False

        class Benchmark(BaseModel):
            seamless: bool = False

        @wired_constraints(
            ConflictsWith("load.enabled", "benchmark.seamless", truthy=True),
        )
        class Config(BaseModel, ConstraintsMixin):
            load: Load = Field(default_factory=Load)
            benchmark: Benchmark = Field(default_factory=Benchmark)

        # Both False is valid
        config = Config()
        assert config.load.enabled is False
        assert config.benchmark.seamless is False

    def test_conflicts_with_truthy_one_true_valid(self):
        """Test ConflictsWith with truthy=True when only one is True."""

        class Load(BaseModel):
            enabled: bool = False

        class Benchmark(BaseModel):
            seamless: bool = False

        @wired_constraints(
            ConflictsWith("load.enabled", "benchmark.seamless", truthy=True),
        )
        class Config(BaseModel, ConstraintsMixin):
            load: Load = Field(default_factory=Load)
            benchmark: Benchmark = Field(default_factory=Benchmark)

        # Only one True is valid
        config = Config(load=Load(enabled=True))
        assert config.load.enabled is True
        assert config.benchmark.seamless is False

    def test_conflicts_with_truthy_both_true_invalid(self):
        """Test ConflictsWith with truthy=True when both are True."""

        class Load(BaseModel):
            enabled: bool = False

        class Benchmark(BaseModel):
            seamless: bool = False

        @wired_constraints(
            ConflictsWith("load.enabled", "benchmark.seamless", truthy=True),
        )
        class Config(BaseModel, ConstraintsMixin):
            load: Load = Field(default_factory=Load)
            benchmark: Benchmark = Field(default_factory=Benchmark)

        # Both True is invalid
        with pytest.raises(ValidationError) as exc_info:
            Config(load=Load(enabled=True), benchmark=Benchmark(seamless=True))

        assert "load.enabled" in str(exc_info.value)
        assert "benchmark.seamless" in str(exc_info.value)

    def test_conflicts_with_default_is_set_check(self):
        """Test ConflictsWith default behavior checks is-set (not None)."""

        class Options(BaseModel):
            opt_a: str | None = None
            opt_b: str | None = None

        @wired_constraints(
            ConflictsWith("options.opt_a", "options.opt_b"),
        )
        class Config(BaseModel, ConstraintsMixin):
            options: Options = Field(default_factory=Options)

        # Both None is valid
        config = Config()
        assert config.options.opt_a is None

        # One set is valid
        config = Config(options=Options(opt_a="value"))
        assert config.options.opt_a == "value"

        # Both set is invalid
        with pytest.raises(ValidationError):
            Config(options=Options(opt_a="a", opt_b="b"))


class TestWiredConstraintsComparisonWithSource:
    """Tests for comparison constraints with explicit source."""

    def test_greater_than_with_source(self):
        """Test GreaterThan with explicit source parameter."""

        class Settings(BaseModel):
            min_val: int = 10
            max_val: int = 100

        @wired_constraints(
            GreaterThan("settings.min_val", source="settings.max_val"),
        )
        class Config(BaseModel, ConstraintsMixin):
            settings: Settings = Field(default_factory=Settings)

        # Valid: max_val (100) > min_val (10)
        config = Config()
        assert config.settings.max_val == 100

        # Invalid: max_val (5) < min_val (10)
        with pytest.raises(ValidationError) as exc_info:
            Config(settings=Settings(max_val=5))

        error_str = str(exc_info.value)
        assert "settings.max_val" in error_str
        assert "settings.min_val" in error_str
        assert "greater than" in error_str

    def test_less_than_with_source(self):
        """Test LessThan with explicit source parameter."""

        class Range(BaseModel):
            start: int = 0
            end: int = 100

        @wired_constraints(
            LessThan("range.end", source="range.start"),
        )
        class Config(BaseModel, ConstraintsMixin):
            range: Range = Field(default_factory=Range)

        # Valid: start (0) < end (100)
        config = Config()
        assert config.range.start == 0

        # Invalid: start (150) > end (100)
        with pytest.raises(ValidationError):
            Config(range=Range(start=150))

    def test_comparison_skips_none_values(self):
        """Test that comparison constraints skip None values."""

        class Settings(BaseModel):
            min_val: int | None = None
            max_val: int | None = None

        @wired_constraints(
            GreaterThan("settings.min_val", source="settings.max_val"),
        )
        class Config(BaseModel, ConstraintsMixin):
            settings: Settings = Field(default_factory=Settings)

        # Both None - should pass (skipped)
        config = Config()
        assert config.settings.min_val is None

        # One None - should pass (skipped)
        config = Config(settings=Settings(max_val=100))
        assert config.settings.max_val == 100


class TestWiredConstraintsMultiple:
    """Tests for multiple wired constraints."""

    def test_multiple_constraints_all_pass(self):
        """Test multiple wired constraints that all pass."""

        class Settings(BaseModel):
            min_val: int = 10
            max_val: int = 100
            enabled: bool = False

        class Options(BaseModel):
            debug: bool = False

        @wired_constraints(
            GreaterThan("settings.min_val", source="settings.max_val"),
            ConflictsWith("settings.enabled", "options.debug", truthy=True),
        )
        class Config(BaseModel, ConstraintsMixin):
            settings: Settings = Field(default_factory=Settings)
            options: Options = Field(default_factory=Options)

        # All constraints pass
        config = Config()
        assert config.settings.max_val == 100

    def test_multiple_constraints_one_fails(self):
        """Test multiple wired constraints where one fails."""

        class Settings(BaseModel):
            min_val: int = 10
            max_val: int = 100
            enabled: bool = False

        class Options(BaseModel):
            debug: bool = False

        @wired_constraints(
            GreaterThan("settings.min_val", source="settings.max_val"),
            ConflictsWith("settings.enabled", "options.debug", truthy=True),
        )
        class Config(BaseModel, ConstraintsMixin):
            settings: Settings = Field(default_factory=Settings)
            options: Options = Field(default_factory=Options)

        # Comparison constraint fails
        with pytest.raises(ValidationError) as exc_info:
            Config(settings=Settings(max_val=5))

        assert "settings.max_val" in str(exc_info.value)

    def test_multiple_constraints_multiple_fail(self):
        """Test multiple wired constraints where multiple fail."""

        class Settings(BaseModel):
            min_val: int = 10
            max_val: int = 100
            enabled: bool = False

        class Options(BaseModel):
            debug: bool = False

        @wired_constraints(
            GreaterThan("settings.min_val", source="settings.max_val"),
            ConflictsWith("settings.enabled", "options.debug", truthy=True),
        )
        class Config(BaseModel, ConstraintsMixin):
            settings: Settings = Field(default_factory=Settings)
            options: Options = Field(default_factory=Options)

        # Both constraints fail
        with pytest.raises(ValidationError) as exc_info:
            Config(
                settings=Settings(max_val=5, enabled=True),
                options=Options(debug=True),
            )

        error_str = str(exc_info.value)
        # Both errors should be present
        assert "settings.max_val" in error_str
        assert "settings.enabled" in error_str


class TestWiredConstraintsCombinedWithAnnotated:
    """Tests for wired constraints combined with annotated constraints."""

    def test_wired_and_annotated_both_validated(self):
        """Test that both wired and annotated constraints are validated."""

        class Inner(BaseModel):
            value: int = 10

        @wired_constraints(
            GreaterThan("inner.value", source="max_value"),
        )
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)
            max_value: int = 20
            # Annotated constraint
            opt_a: Annotated[str | None, ExclusiveGroup("opts")] = None
            opt_b: Annotated[str | None, ExclusiveGroup("opts")] = None

        # Both pass
        config = Config(opt_a="value")
        assert config.opt_a == "value"

        # Annotated constraint fails
        with pytest.raises(ValidationError) as exc_info:
            Config(opt_a="a", opt_b="b")

        assert "Only one of" in str(exc_info.value)

        # Wired constraint fails
        with pytest.raises(ValidationError) as exc_info:
            Config(max_value=5)

        assert "max_value" in str(exc_info.value)


class TestGetNestedValue:
    """Tests for the _get_nested_value helper function."""

    def test_simple_attribute(self):
        """Test simple attribute access without dots."""

        class Obj:
            value = 42

        assert _get_nested_value(Obj(), "value") == 42

    def test_dot_path(self):
        """Test dot-notation path."""

        class Inner:
            value = 42

        class Outer:
            inner = Inner()

        assert _get_nested_value(Outer(), "inner.value") == 42

    def test_array_index(self):
        """Test array indexing."""

        class Obj:
            items = [1, 2, 3]

        assert _get_nested_value(Obj(), "items[0]") == 1
        assert _get_nested_value(Obj(), "items[1]") == 2
        assert _get_nested_value(Obj(), "items[2]") == 3

    def test_mixed_path(self):
        """Test mixed dot and array notation."""

        class Item:
            value = 42

        class Obj:
            items = [Item()]

        assert _get_nested_value(Obj(), "items[0].value") == 42

    def test_invalid_attribute_raises(self):
        """Test that invalid attribute raises AttributeError."""

        class Obj:
            pass

        with pytest.raises(AttributeError):
            _get_nested_value(Obj(), "nonexistent")

    def test_invalid_index_raises(self):
        """Test that invalid index raises IndexError."""

        class Obj:
            items = [1, 2]

        with pytest.raises(IndexError):
            _get_nested_value(Obj(), "items[10]")


# =============================================================================
# FLUENT F() API TESTS
# =============================================================================


class TestFRepr:
    """Tests for F() representation."""

    def test_repr(self):
        """Test F() repr."""
        f = F("load.enabled")
        assert repr(f) == "F('load.enabled')"


class TestFComparisonOperators:
    """Tests for F() comparison operators."""

    def test_greater_than_valid(self):
        """Test F() > F() when valid."""

        class Inner(BaseModel):
            max_val: int = 100
            min_val: int = 10

        @wired_constraints(F("inner.max_val") > F("inner.min_val"))
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)

        config = Config()
        assert config.inner.max_val == 100

    def test_greater_than_invalid(self):
        """Test F() > F() when invalid."""

        class Inner(BaseModel):
            max_val: int = 100
            min_val: int = 10

        @wired_constraints(F("inner.max_val") > F("inner.min_val"))
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)

        with pytest.raises(ValidationError) as exc_info:
            Config(inner=Inner(max_val=5))

        assert "inner.max_val" in str(exc_info.value)
        assert "greater than" in str(exc_info.value)

    def test_less_than_valid(self):
        """Test F() < F() when valid."""

        class Inner(BaseModel):
            start: int = 0
            end: int = 100

        @wired_constraints(F("inner.start") < F("inner.end"))
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)

        config = Config()
        assert config.inner.start == 0

    def test_less_than_invalid(self):
        """Test F() < F() when invalid."""

        class Inner(BaseModel):
            start: int = 0
            end: int = 100

        @wired_constraints(F("inner.start") < F("inner.end"))
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)

        with pytest.raises(ValidationError):
            Config(inner=Inner(start=150))

    def test_greater_than_or_equal_valid(self):
        """Test F() >= F() when valid (equal)."""

        class Inner(BaseModel):
            a: int = 10
            b: int = 10

        @wired_constraints(F("inner.a") >= F("inner.b"))
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)

        config = Config()
        assert config.inner.a == 10

    def test_less_than_or_equal_valid(self):
        """Test F() <= F() when valid (equal)."""

        class Inner(BaseModel):
            a: int = 10
            b: int = 10

        @wired_constraints(F("inner.a") <= F("inner.b"))
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)

        config = Config()
        assert config.inner.a == 10

    def test_equal_valid(self):
        """Test F() == F() when valid."""

        class Inner(BaseModel):
            password: str = "secret"
            confirm: str = "secret"

        @wired_constraints(F("inner.password") == F("inner.confirm"))
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)

        config = Config()
        assert config.inner.password == "secret"

    def test_equal_invalid(self):
        """Test F() == F() when invalid."""

        class Inner(BaseModel):
            password: str = "secret"
            confirm: str = "secret"

        @wired_constraints(F("inner.password") == F("inner.confirm"))
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)

        with pytest.raises(ValidationError):
            Config(inner=Inner(confirm="different"))

    def test_not_equal_valid(self):
        """Test F() != F() when valid."""

        class Inner(BaseModel):
            a: int = 10
            b: int = 20

        @wired_constraints(F("inner.a") != F("inner.b"))
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)

        config = Config()
        assert config.inner.a != config.inner.b

    def test_not_equal_invalid(self):
        """Test F() != F() when invalid."""

        class Inner(BaseModel):
            a: int = 10
            b: int = 20

        @wired_constraints(F("inner.a") != F("inner.b"))
        class Config(BaseModel, ConstraintsMixin):
            inner: Inner = Field(default_factory=Inner)

        with pytest.raises(ValidationError):
            Config(inner=Inner(a=20, b=20))


class TestFConflictsWith:
    """Tests for F().conflicts_with()."""

    def test_conflicts_with_none_truthy_valid(self):
        """Test conflicts_with with truthy=True when none are truthy."""

        class Load(BaseModel):
            enabled: bool = False

        class Debug(BaseModel):
            enabled: bool = False

        @wired_constraints(
            F("load.enabled").conflicts_with(F("debug.enabled"), truthy=True)
        )
        class Config(BaseModel, ConstraintsMixin):
            load: Load = Field(default_factory=Load)
            debug: Debug = Field(default_factory=Debug)

        config = Config()
        assert config.load.enabled is False

    def test_conflicts_with_one_truthy_valid(self):
        """Test conflicts_with with truthy=True when one is truthy."""

        class Load(BaseModel):
            enabled: bool = False

        class Debug(BaseModel):
            enabled: bool = False

        @wired_constraints(
            F("load.enabled").conflicts_with(F("debug.enabled"), truthy=True)
        )
        class Config(BaseModel, ConstraintsMixin):
            load: Load = Field(default_factory=Load)
            debug: Debug = Field(default_factory=Debug)

        config = Config(load=Load(enabled=True))
        assert config.load.enabled is True

    def test_conflicts_with_both_truthy_invalid(self):
        """Test conflicts_with with truthy=True when both are truthy."""

        class Load(BaseModel):
            enabled: bool = False

        class Debug(BaseModel):
            enabled: bool = False

        @wired_constraints(
            F("load.enabled").conflicts_with(F("debug.enabled"), truthy=True)
        )
        class Config(BaseModel, ConstraintsMixin):
            load: Load = Field(default_factory=Load)
            debug: Debug = Field(default_factory=Debug)

        with pytest.raises(ValidationError) as exc_info:
            Config(load=Load(enabled=True), debug=Debug(enabled=True))

        error_str = str(exc_info.value)
        assert "load.enabled" in error_str
        assert "debug.enabled" in error_str

    def test_conflicts_with_multiple_fields(self):
        """Test conflicts_with with multiple conflicting fields."""

        class Options(BaseModel):
            a: str | None = None
            b: str | None = None
            c: str | None = None

        @wired_constraints(F("opts.a").conflicts_with(F("opts.b"), F("opts.c")))
        class Config(BaseModel, ConstraintsMixin):
            opts: Options = Field(default_factory=Options)

        # One set is valid
        config = Config(opts=Options(a="value"))
        assert config.opts.a == "value"

        # Two set is invalid
        with pytest.raises(ValidationError):
            Config(opts=Options(a="a", b="b"))


class TestFRequires:
    """Tests for F().requires()."""

    def test_requires_both_set_valid(self):
        """Test requires when both are set."""

        class SSL(BaseModel):
            cert: str | None = None
            key: str | None = None

        @wired_constraints(F("ssl.cert").requires(F("ssl.key")))
        class Config(BaseModel, ConstraintsMixin):
            ssl: SSL = Field(default_factory=SSL)

        config = Config(ssl=SSL(cert="/cert", key="/key"))
        assert config.ssl.cert == "/cert"

    def test_requires_neither_set_valid(self):
        """Test requires when neither is set."""

        class SSL(BaseModel):
            cert: str | None = None
            key: str | None = None

        @wired_constraints(F("ssl.cert").requires(F("ssl.key")))
        class Config(BaseModel, ConstraintsMixin):
            ssl: SSL = Field(default_factory=SSL)

        config = Config()
        assert config.ssl.cert is None

    def test_requires_source_set_required_missing_invalid(self):
        """Test requires when source is set but required is missing."""

        class SSL(BaseModel):
            cert: str | None = None
            key: str | None = None

        @wired_constraints(F("ssl.cert").requires(F("ssl.key")))
        class Config(BaseModel, ConstraintsMixin):
            ssl: SSL = Field(default_factory=SSL)

        with pytest.raises(ValidationError) as exc_info:
            Config(ssl=SSL(cert="/cert"))

        error_str = str(exc_info.value)
        assert "ssl.cert" in error_str
        assert "ssl.key" in error_str
        assert "requires" in error_str


class TestFRequiresAll:
    """Tests for F().requires_all()."""

    def test_requires_all_all_set_valid(self):
        """Test requires_all when all are set."""

        class DB(BaseModel):
            host: str | None = None
            port: int | None = None
            user: str | None = None

        @wired_constraints(F("db.host").requires_all(F("db.port"), F("db.user")))
        class Config(BaseModel, ConstraintsMixin):
            db: DB = Field(default_factory=DB)

        config = Config(db=DB(host="localhost", port=5432, user="admin"))
        assert config.db.host == "localhost"

    def test_requires_all_source_not_set_valid(self):
        """Test requires_all when source is not set."""

        class DB(BaseModel):
            host: str | None = None
            port: int | None = None
            user: str | None = None

        @wired_constraints(F("db.host").requires_all(F("db.port"), F("db.user")))
        class Config(BaseModel, ConstraintsMixin):
            db: DB = Field(default_factory=DB)

        config = Config()
        assert config.db.host is None

    def test_requires_all_missing_one_invalid(self):
        """Test requires_all when one required is missing."""

        class DB(BaseModel):
            host: str | None = None
            port: int | None = None
            user: str | None = None

        @wired_constraints(F("db.host").requires_all(F("db.port"), F("db.user")))
        class Config(BaseModel, ConstraintsMixin):
            db: DB = Field(default_factory=DB)

        with pytest.raises(ValidationError) as exc_info:
            Config(db=DB(host="localhost", port=5432))

        error_str = str(exc_info.value)
        assert "db.host" in error_str
        assert "db.user" in error_str
        assert "requires" in error_str


# =============================================================================
# TESTS: Group Fields Lookup
# =============================================================================


class MockGroup:
    """Mock cyclopts Group for testing group field extraction."""

    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return f"MockGroup({self._name!r})"


class MockCLIGroupParameter:
    """Mock CLI parameter with both name and group attributes."""

    def __init__(self, *names: str, group: MockGroup):
        self.name = names
        self.group = (group,)


GROUP_A = MockGroup("GroupA")
GROUP_B = MockGroup("GroupB")
GROUP_C = MockGroup("GroupC")


class GroupFieldModel(BaseModel, ConstraintsMixin):
    """Model with fields assigned to mock groups."""

    x: Annotated[
        int, Field(description="x"), MockCLIGroupParameter("--x", group=GROUP_A)
    ] = 1
    y: Annotated[
        int, Field(description="y"), MockCLIGroupParameter("--y", group=GROUP_A)
    ] = 2
    z: Annotated[
        int, Field(description="z"), MockCLIGroupParameter("--z", group=GROUP_B)
    ] = 3
    plain: Annotated[str | None, Field(description="plain")] = None


class TestCollectGroupFields:
    """Tests for collect_group_fields function."""

    def test_groups_fields_correctly(self):
        result = collect_group_fields(GroupFieldModel)
        assert set(result[GROUP_A]) == {"x", "y"}
        assert result[GROUP_B] == ["z"]

    def test_excludes_fields_without_group(self):
        result = collect_group_fields(GroupFieldModel)
        for fields in result.values():
            assert "plain" not in fields

    def test_unknown_group_not_in_result(self):
        result = collect_group_fields(GroupFieldModel)
        assert GROUP_C not in result

    def test_empty_model(self):
        class EmptyModel(BaseModel, ConstraintsMixin):
            val: Annotated[int, Field(description="val")] = 0

        assert collect_group_fields(EmptyModel) == {}

    def test_inherited_fields(self):
        class ChildModel(GroupFieldModel):
            w: Annotated[
                int, Field(description="w"), MockCLIGroupParameter("--w", group=GROUP_B)
            ] = 4

        result = collect_group_fields(ChildModel)
        assert set(result[GROUP_A]) == {"x", "y"}
        assert set(result[GROUP_B]) == {"z", "w"}

    def test_stored_on_class(self):
        assert hasattr(GroupFieldModel, "_group_fields")
        assert set(GroupFieldModel._group_fields[GROUP_A]) == {"x", "y"}
        assert GroupFieldModel._group_fields[GROUP_B] == ["z"]


class TestAnyFieldSetInGroup:
    """Tests for any_field_set_in_group method."""

    def test_false_when_all_defaults(self):
        m = GroupFieldModel()
        assert m.any_field_set_in_group(GROUP_A) is False
        assert m.any_field_set_in_group(GROUP_B) is False

    def test_true_when_field_explicitly_set(self):
        m = GroupFieldModel(x=10)
        assert m.any_field_set_in_group(GROUP_A) is True
        assert m.any_field_set_in_group(GROUP_B) is False

    def test_true_when_set_to_default_value(self):
        """Explicitly passing the default value still counts as 'set'."""
        m = GroupFieldModel(z=3)
        assert m.any_field_set_in_group(GROUP_B) is True

    def test_multiple_fields_in_group(self):
        m = GroupFieldModel(x=10, y=20)
        assert m.any_field_set_in_group(GROUP_A) is True

    def test_different_groups_independent(self):
        m = GroupFieldModel(z=99)
        assert m.any_field_set_in_group(GROUP_A) is False
        assert m.any_field_set_in_group(GROUP_B) is True

    def test_unknown_group_returns_false(self):
        m = GroupFieldModel()
        assert m.any_field_set_in_group(GROUP_C) is False

    def test_field_without_group_ignored(self):
        m = GroupFieldModel(plain="hello")
        assert m.any_field_set_in_group(GROUP_A) is False
        assert m.any_field_set_in_group(GROUP_B) is False
