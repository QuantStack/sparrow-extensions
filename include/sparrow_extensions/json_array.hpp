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

#pragma once

#include "sparrow/layout/array_registry.hpp"
#include "sparrow/utils/extension.hpp"
#include "sparrow/variable_size_binary_array.hpp"
#include "sparrow/variable_size_binary_view_array.hpp"
#include "sparrow/utils/extension.hpp"

namespace sparrow
{

    using json_extension = simple_extension<"arrow.json">;
    /**
     * @brief JSON array with 32-bit offsets.
     *
     * A variable-size string array for storing JSON-encoded data where the cumulative
     * length of all strings does not exceed 2^31-1 bytes. This is the standard choice
     * for most JSON datasets.
     *
     * The JSON extension type is defined as:
     * - Extension name: "arrow.json"
     * - Storage type: String (Utf8)
     * - Extension metadata: none
     *
     * Related Apache Arrow specification:
     * https://arrow.apache.org/docs/format/CanonicalExtensions.html#json
     *
     * @see big_json_array for larger datasets requiring 64-bit offsets
     * @see json_view_array for view-based storage
     */
    using json_array = variable_size_binary_array_impl<
        arrow_traits<std::string>::value_type,
        arrow_traits<std::string>::const_reference,
        std::int32_t,
        json_extension>;

    /**
     * @brief JSON array with 64-bit offsets.
     *
     * A variable-size string array for storing JSON-encoded data where the cumulative
     * length of all strings may exceed 2^31-1 bytes. Use this for very large JSON datasets.
     *
     * The JSON extension type is defined as:
     * - Extension name: "arrow.json"
     * - Storage type: LargeString (LargeUtf8)
     * - Extension metadata: none
     *
     * Related Apache Arrow specification:
     * https://arrow.apache.org/docs/format/CanonicalExtensions.html#json
     *
     * @see json_array for smaller datasets with 32-bit offsets
     * @see json_view_array for view-based storage
     */
    using big_json_array = variable_size_binary_array_impl<
        arrow_traits<std::string>::value_type,
        arrow_traits<std::string>::const_reference,
        std::int64_t,
        json_extension>;

    /**
     * @brief JSON array with view-based storage.
     *
     * A variable-size string view array for storing JSON-encoded data using the
     * Binary View layout, which is optimized for performance by storing short values
     * inline and using references to external buffers for longer values.
     *
     * The JSON extension type is defined as:
     * - Extension name: "arrow.json"
     * - Storage type: StringView (Utf8View)
     * - Extension metadata: none
     *
     * Related Apache Arrow specification:
     * https://arrow.apache.org/docs/format/CanonicalExtensions.html#json
     *
     * @see json_array for offset-based storage with 32-bit offsets
     * @see big_json_array for offset-based storage with 64-bit offsets
     */
    using json_view_array = variable_size_binary_view_array_impl<
        arrow_traits<std::string>::value_type,
        arrow_traits<std::string>::const_reference,
        json_extension>;

    namespace detail
    {
        template <>
        struct get_data_type_from_array<sparrow::json_array>
        {
            [[nodiscard]] static constexpr sparrow::data_type get()
            {
                return sparrow::data_type::STRING;
            }
        };

        template <>
        struct get_data_type_from_array<sparrow::big_json_array>
        {
            [[nodiscard]] static constexpr sparrow::data_type get()
            {
                return sparrow::data_type::LARGE_STRING;
            }
        };

        template <>
        struct get_data_type_from_array<sparrow::json_view_array>
        {
            [[nodiscard]] static constexpr sparrow::data_type get()
            {
                return sparrow::data_type::STRING_VIEW;
            }
        };
    }

    namespace detail
    {
        inline const bool json_arrays_registered = []()
        {
            auto& registry = array_registry::instance();

            constexpr std::string_view extension_name = "arrow.json";

            // Register json_array (STRING base type)
            registry.register_extension(
                data_type::STRING,
                extension_name,
                [](arrow_proxy proxy)
                {
                    return cloning_ptr<array_wrapper>{
                        new array_wrapper_impl<json_array>(json_array(std::move(proxy)))
                    };
                }
            );

            // Register big_json_array (LARGE_STRING base type)
            registry.register_extension(
                data_type::LARGE_STRING,
                extension_name,
                [](arrow_proxy proxy)
                {
                    return cloning_ptr<array_wrapper>{
                        new array_wrapper_impl<big_json_array>(big_json_array(std::move(proxy)))
                    };
                }
            );

            // Register json_view_array (STRING_VIEW base type)
            registry.register_extension(
                data_type::STRING_VIEW,
                extension_name,
                [](arrow_proxy proxy)
                {
                    return cloning_ptr<array_wrapper>{
                        new array_wrapper_impl<json_view_array>(json_view_array(std::move(proxy)))
                    };
                }
            );

            return true;
        }();
    }  // namespace detail
}