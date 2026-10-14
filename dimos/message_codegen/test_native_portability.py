# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Target selection/cache checks; real macOS builds run in the hosted CI matrix."""

from pathlib import Path
import subprocess

import pytest

from dimos.message_codegen import native_build


@pytest.fixture
def darwin_tools(monkeypatch):
    monkeypatch.setattr(native_build.sys, "platform", "darwin")
    for name in [
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
        "CMAKE_OSX_SYSROOT",
        "CMAKE_OSX_DEPLOYMENT_TARGET",
        "CMAKE_OSX_ARCHITECTURES",
        "CMAKE_TOOLCHAIN_FILE",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(native_build.shutil, "which", lambda name: "/tools/" + name)
    monkeypatch.setattr(native_build, "find_spec", lambda name: True)
    monkeypatch.setattr(native_build.metadata, "version", lambda name: "1.0")

    def output(command, **kwargs):
        return "/SDKs/MacOSX.sdk\n" if command[0] == "xcrun" else "test compiler 1"

    monkeypatch.setattr(native_build.subprocess, "check_output", output)


def test_macos_sdk_and_target_are_forwarded_and_invalidate_cache(darwin_tools, monkeypatch):
    first = native_build._toolchain_key()
    assert native_build._cmake_toolchain_args() == ["-DCMAKE_OSX_SYSROOT=/SDKs/MacOSX.sdk"]
    monkeypatch.setenv("MACOSX_DEPLOYMENT_TARGET", "13.0")
    second = native_build._toolchain_key()
    assert second != first
    monkeypatch.setenv("CMAKE_OSX_ARCHITECTURES", "arm64;x86_64")
    third = native_build._toolchain_key()
    assert third != second
    monkeypatch.setenv("SDKROOT", "/SDKs/Other.sdk")
    assert native_build._toolchain_key() != third
    assert native_build._cmake_toolchain_args() == [
        "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64",
        "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0",
        "-DCMAKE_OSX_SYSROOT=/SDKs/Other.sdk",
    ]


def test_missing_macos_sdk_tools_fail_without_installing(darwin_tools, monkeypatch):
    monkeypatch.setattr(
        native_build.shutil, "which", lambda name: None if name == "xcrun" else "/tools/" + name
    )
    with pytest.raises(RuntimeError, match="Command Line Tools"):
        native_build._cmake_toolchain_args()


@pytest.mark.parametrize("apple,origin", [(True, "@loader_path"), (False, "$ORIGIN")])
def test_nested_cmake_projects_receive_target_and_platform_rpath(tmp_path, apple, origin):
    template = Path(native_build.__file__).parent / "templates/native-toolchain.cmake"
    script = tmp_path / "probe.cmake"
    script.write_text(
        f"set(APPLE {'TRUE' if apple else 'FALSE'})\n"
        'set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")\n'
        'set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")\n'
        f'include("{template}")\n'
        'message("origin=${_dimos_origin}")\n'
        'message("args=${_dimos_toolchain_args}")\n'
    )
    result = subprocess.run(
        ["cmake", "-P", str(script)], capture_output=True, text=True, check=True
    )
    assert f"origin={origin}" in result.stderr
    assert "-DCMAKE_OSX_ARCHITECTURES=arm64|x86_64" in result.stderr
    assert "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0" in result.stderr


def test_export_uses_saved_target_and_original_toolchain(tmp_path, monkeypatch):
    prefix = tmp_path / "install"
    prefix.mkdir()
    (prefix / "native-prefixes.json").write_text("[]")
    original = tmp_path / "original.cmake"
    original.write_text('set(CMAKE_CXX_FLAGS_INIT "-DREQUIRED_TARGET=42")\n')
    native_build.save_target_settings(prefix, [f"-DCMAKE_TOOLCHAIN_FILE={original}"])
    monkeypatch.setenv("CMAKE_TOOLCHAIN_FILE", "/wrong/current/toolchain")
    exported = native_build.write_cmake_toolchain(prefix, tmp_path / "exported.cmake")
    source = tmp_path / "consumer"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(consumer LANGUAGES CXX)\n"
        "add_executable(consumer main.cpp)\n"
    )
    (source / "main.cpp").write_text("static_assert(REQUIRED_TARGET == 42); int main() {}\n")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(source / "build"),
            f"-DCMAKE_TOOLCHAIN_FILE={exported}",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(["cmake", "--build", str(source / "build")], check=True, capture_output=True)
    subprocess.run([str(source / "build/consumer")], check=True)
    with pytest.raises(ValueError, match="must not replace"):
        native_build.write_cmake_toolchain(prefix, original)
    original.write_text("# changed\n")
    with pytest.raises(ValueError, match="changed or is missing"):
        native_build.write_cmake_toolchain(prefix, exported)


def test_export_preserves_saved_macos_target_without_current_environment(tmp_path, monkeypatch):
    (tmp_path / "native-prefixes.json").write_text("[]")
    native_build.save_target_settings(
        tmp_path,
        [
            "-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0",
            "-DCMAKE_OSX_SYSROOT=/SDKs/Original.sdk",
        ],
    )
    monkeypatch.setenv("CMAKE_OSX_ARCHITECTURES", "wrong")
    exported = native_build.write_cmake_toolchain(tmp_path, tmp_path / "exported.cmake")
    probe = tmp_path / "probe.cmake"
    probe.write_text(
        f'include("{exported}")\n'
        + 'message("target=${CMAKE_OSX_ARCHITECTURES}|${CMAKE_OSX_DEPLOYMENT_TARGET}|${CMAKE_OSX_SYSROOT}")\n'
    )
    result = subprocess.run(["cmake", "-P", str(probe)], check=True, capture_output=True, text=True)
    assert "target=arm64;x86_64|13.0|/SDKs/Original.sdk" in result.stderr


def test_custom_toolchain_content_invalidates_cache(darwin_tools, monkeypatch, tmp_path):
    custom = tmp_path / "custom.cmake"
    custom.write_text("set(TARGET_CHOICE one)\n")
    monkeypatch.setenv("CMAKE_TOOLCHAIN_FILE", str(custom))
    first = native_build._toolchain_key()
    custom.write_text("set(TARGET_CHOICE two)\n")
    assert native_build._toolchain_key() != first
