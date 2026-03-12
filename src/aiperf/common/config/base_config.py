# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
from enum import Enum
from pathlib import Path
from typing import Any, Self

import orjson
from pydantic import BaseModel
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

ADD_TO_TEMPLATE = "add_to_template"


class BaseConfig(BaseModel):
    """
    Base configuration class for all configurations.
    """

    def to_json(
        self,
        indent: bool = False,
        exclude_unset: bool = False,
        exclude_none: bool = False,
    ) -> bytes:
        """Serialize the config to JSON bytes.

        Args:
            indent: Whether to pretty-print with indentation.
            exclude_unset: Whether to exclude fields that weren't explicitly set.
                Use this for Kubernetes ConfigMaps to avoid validation issues
                when loading the config back.
            exclude_none: Whether to exclude fields with None values.

        Returns:
            JSON bytes representation of the config.
        """
        opts = orjson.OPT_SERIALIZE_NUMPY
        if indent:
            opts |= orjson.OPT_INDENT_2
        return orjson.dumps(
            self.model_dump(
                mode="json", exclude_unset=exclude_unset, exclude_none=exclude_none
            ),
            option=opts,
        )

    def to_json_str(
        self,
        indent: bool = False,
        exclude_unset: bool = False,
        exclude_none: bool = False,
    ) -> str:
        """Serialize the config to a JSON string.

        Args:
            indent: Whether to pretty-print with indentation.
            exclude_unset: Whether to exclude fields that weren't explicitly set.
                Use this for Kubernetes ConfigMaps to avoid validation issues
                when loading the config back.
            exclude_none: Whether to exclude fields with None values.

        Returns:
            JSON string representation of the config.
        """
        return self.to_json(
            indent=indent, exclude_unset=exclude_unset, exclude_none=exclude_none
        ).decode("utf-8")

    @classmethod
    def from_json(cls, data: bytes | str) -> Self:
        """Deserialize a config from JSON.

        Args:
            data: JSON bytes or string.

        Returns:
            Config instance.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        return cls.model_validate(orjson.loads(data))

    @classmethod
    def from_json_file(cls, path: Path) -> Self:
        """Load a config from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Config instance.
        """
        with open(path, "rb") as f:
            return cls.from_json(f.read())

    def serialize_to_yaml(self, verbose: bool = False, indent: int = 4) -> str:
        """
        Serialize a Pydantic model to a YAML string.

        Args:
            verbose: Whether to include verbose comments in the YAML output.
            indent: The per-level indentation to use.
        """
        # Dump model to dict with context (flags propagate recursively)
        context = {
            "verbose": verbose,
        }

        data = self.model_dump(context=context)

        # Attach comments recursively
        commented_data = self._attach_comments(
            data=data,
            model=self,
            context=context,
            indent=indent,
        )

        # Dump to YAML
        yaml = YAML(pure=True)
        yaml.indent(mapping=indent, sequence=indent, offset=indent)

        stream = io.StringIO()
        yaml.dump(commented_data, stream)
        return stream.getvalue()

    @staticmethod
    def _attach_comments(
        data: Any,
        model: BaseModel,
        context: dict,
        indent: int,
        indent_level: int = 0,
    ) -> Any:
        """
        Recursively convert dicts to ruamel.yaml CommentedMap and attach comments from
        Pydantic field descriptions, or based on context (e.g., verbose flag).

        Args:
            data: The raw data to convert to a CommentedMap.
            model: The Pydantic model that contains the field descriptions.
            context: The Pydantic serializer context which contains the serializer flags.
            indent: The per-level indentation to use for the comments.
            indent_level: The current level of indentation. The actual indentation is
                `indent * indent_level`.

        Returns:
            The data with comments attached.
        """
        if isinstance(data, dict):
            # Create a CommentedMap to store the commented data. This is a special type of
            # dict provided by the ruamel.yaml library that preserves the order of the keys and
            # allows for comments to be attached to the keys.
            commented_map = CommentedMap()

            for field_name, value in data.items():
                field = model.__class__.model_fields.get(field_name)

                if not BaseConfig._should_add_field_to_template(field):
                    continue

                if BaseConfig._is_a_nested_config(field, value):
                    # Recursively process nested models
                    commented_map[field_name] = BaseConfig._attach_comments(
                        value,
                        getattr(model, field_name),
                        context=context,
                        indent=indent,
                        indent_level=indent_level + 1,
                    )

                    commented_map.yaml_set_comment_before_after_key(
                        field_name,
                        before="\n",
                        indent=indent * (indent_level + 1),
                    )
                else:
                    # Attach the value to the commented map
                    commented_map[field_name] = BaseConfig._preprocess_value(value)

                # Attach comment if verbose and description exists
                if context.get("verbose") and field and field.description:
                    # Set the comment before the key, with the specified indentation
                    commented_map.yaml_set_comment_before_after_key(
                        field_name,
                        before="\n" + field.description,
                        indent=indent * indent_level,
                    )

            return commented_map

        return data

    @staticmethod
    def _should_add_field_to_template(field: Any) -> bool:
        # Check if the field should be added to the template based on json_schema_extra
        # and the add_to_template flag.
        # If add_to_template is False, we skip adding the field to the template.
        # If add_to_template is True or not present, we include the field in the template.
        if field and field.json_schema_extra:
            return field.json_schema_extra.get(ADD_TO_TEMPLATE, True)
        else:
            return True

    @staticmethod
    def _is_a_nested_config(field: Any, value: Any) -> bool:
        if not isinstance(value, dict) or not field:
            return False
        try:
            return issubclass(field.annotation, BaseModel)
        except TypeError:
            return False

    @staticmethod
    def _preprocess_value(value: Any) -> Any:
        """
        Preprocess the value before serialization.
        """

        if isinstance(value, Enum):
            return str(value).lower()
        elif isinstance(value, Path):
            return str(value)
        else:
            return value
