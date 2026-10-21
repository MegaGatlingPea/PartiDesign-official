# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Load external resources, such as external tools or data resources."""

from collections.abc import Iterator
import os
import pathlib
import typing
from typing import BinaryIO, Final, Literal, TextIO

from importlib import resources
import alphafold3.common


_DATA_ROOT:  Final[pathlib.Path] = (
    resources.files(alphafold3.common).joinpath('..').resolve()
)
ROOT = _DATA_ROOT


_AF3_CCD_ENV = 'PARTIDESIGN_AF3_CCD_DIR'
_AF3_GENERATED_CCD = frozenset({
    'constants/converters/ccd.pickle',
    'constants/converters/chemical_component_sets.pickle',
})


def _resource_path(name: str | os.PathLike[str]) -> pathlib.Path:
  path = pathlib.Path(name)
  try:
    relative = path.resolve(strict=False).relative_to(_DATA_ROOT)
  except ValueError:
    relative = path
  key = relative.as_posix().lstrip('./')
  override = os.environ.get(_AF3_CCD_ENV)
  if override and key in _AF3_GENERATED_CCD:
    return pathlib.Path(override).expanduser() / relative.name
  return _DATA_ROOT / path


def filename(name: str | os.PathLike[str]) -> str:
  """Returns the absolute path to an external resource.

  Note that this calls resources.GetResourceFilename under the hood and hence
  causes par file unpacking, which might be unfriendly on diskless machines.


  Args:
    name: the name of the resource corresponding to its path relative to the
      root of the repository.
  """
  return _resource_path(name).as_posix()


@typing.overload
def open_resource(
    name: str | os.PathLike[str], mode: Literal['r', 'rt'] = 'rt'
) -> TextIO:
  ...


@typing.overload
def open_resource(
    name: str | os.PathLike[str], mode: Literal['rb']
) -> BinaryIO:
  ...


def open_resource(
    name: str | os.PathLike[str], mode: str = 'rb'
) -> TextIO | BinaryIO:
  """Returns an open file object for the named resource.

  Args:
    name: the name of the resource corresponding to its path relative to the
      root of the repository.
    mode: the mode to use when opening the file.
  """
  return _resource_path(name).open(mode)  # pyrefly: ignore[bad-return]


def get_resource_dir(path: str | os.PathLike[str]) -> os.PathLike[str]:
  return _resource_path(path)


def walk(path: str) -> Iterator[tuple[str, list[str], list[str]]]:
  """Walks the directory tree of resources similar to os.walk."""
  return os.walk((_DATA_ROOT / path).as_posix())
