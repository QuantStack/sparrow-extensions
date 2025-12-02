// Copyright 2024 Man Group Operations Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <doctest/doctest.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <string>

#include <sparrow_extensions/config/sparrow_extensions_version.hpp>

#ifdef HAS_BETTER_JUNIT_REPORTER
#include "better_junit_reporter.hpp"
#endif

TEST_CASE("versions are readable")
{
    // TODO: once available on OSX, use `<format>` facility instead.
    // We only try to make sure the version values are printable, whatever their type.
    // AKA this is not written to be fancy but to force conversion to string.
    using namespace sparrow_extensions;
    [[maybe_unused]] const std::string printable_version = std::string("sparrow version : ")
                                                           + std::to_string(SPARROW_EXTENSIONS_VERSION_MAJOR) + "."
                                                           + std::to_string(SPARROW_EXTENSIONS_VERSION_MINOR) + "."
                                                           + std::to_string(SPARROW_EXTENSIONS_VERSION_PATCH);

    [[maybe_unused]] const std::string printable_binary_version = std::string("sparrow binary version: ")
                                                                  + std::to_string(SPARROW_EXTENSIONS_BINARY_CURRENT) + "."
                                                                  + std::to_string(SPARROW_EXTENSIONS_BINARY_REVISION)
                                                                  + "." + std::to_string(SPARROW_EXTENSIONS_BINARY_AGE);
}
