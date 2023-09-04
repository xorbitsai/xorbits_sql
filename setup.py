# Copyright 2022-2023 XProbe Inc.
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

import os
import sys

from setuptools import setup


repo_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(repo_root)

# Resolve path issue of versioneer
sys.path.append(repo_root)
versioneer = __import__("versioneer")


# build long description
def build_long_description():
    readme_path = os.path.join(repo_root, "README.md")

    with open(readme_path, encoding="utf-8") as f:
        return f.read()


setup_options = dict(
    version=versioneer.get_version(),
    long_description=build_long_description(),
    long_description_content_type="text/markdown",
)
setup(**setup_options)
