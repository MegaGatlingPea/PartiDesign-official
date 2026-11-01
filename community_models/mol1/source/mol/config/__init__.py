# Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration: the parser and value types, plus the default settings."""

from .base import basic_configs, configs
from .data import data_configs
from .inference import inference_configs
from .model_type import model_configs
from .parser import (
    ArgumentNotSet,
    ConfigManager,
    load_config,
    parse_configs,
    parse_sys_args,
    save_config,
)
from .types import (
    DefaultNoneWithType,
    GlobalConfigValue,
    ListValue,
    RequiredValue,
    ValueMaybeNone,
    get_bool_value,
)

__all__ = [
    # parser and value types
    "ArgumentNotSet",
    "ConfigManager",
    "DefaultNoneWithType",
    "GlobalConfigValue",
    "ListValue",
    "RequiredValue",
    "ValueMaybeNone",
    "get_bool_value",
    "load_config",
    "parse_configs",
    "parse_sys_args",
    "save_config",
    # default settings
    "basic_configs",
    "configs",
    "data_configs",
    "inference_configs",
    "model_configs",
]
