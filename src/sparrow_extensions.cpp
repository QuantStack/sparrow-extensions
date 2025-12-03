// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Minimal translation unit for Windows DLL linking.
// Header-only libraries require at least one exported symbol
// when built as a shared library on Windows to generate the import library.

#include "sparrow_extensions/config/config.hpp"
#include "sparrow_extensions/config/sparrow_extensions_version.hpp"

namespace sparrow_extensions
{
    SPARROW_EXTENSIONS_API int version_major() noexcept
    {
        return SPARROW_EXTENSIONS_VERSION_MAJOR;
    }

    SPARROW_EXTENSIONS_API int version_minor() noexcept
    {
        return SPARROW_EXTENSIONS_VERSION_MINOR;
    }

    SPARROW_EXTENSIONS_API int version_patch() noexcept
    {
        return SPARROW_EXTENSIONS_VERSION_PATCH;
    }
}
