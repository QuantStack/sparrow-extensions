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

#include <cstdint>
#include <type_traits>
#include <vector>

#include <doctest/doctest.h>

#include "sparrow/array.hpp"
#include "sparrow/list_array.hpp"
#include "sparrow/primitive_array.hpp"

#include "sparrow_extensions/variable_shape_tensor.hpp"

namespace sparrow_extensions
{
    TEST_SUITE("variable_shape_tensor")
    {
        using metadata = variable_shape_tensor_extension::metadata;

        TEST_CASE("metadata::is_valid")
        {
            SUBCASE("empty metadata")
            {
                metadata meta{std::nullopt, std::nullopt, std::nullopt};
                CHECK(meta.is_valid());
            }

            SUBCASE("valid with dim_names only")
            {
                metadata meta{std::vector<std::string>{"C", "H", "W"}, std::nullopt, std::nullopt};
                CHECK(meta.is_valid());
            }

            SUBCASE("valid with permutation only")
            {
                metadata meta{std::nullopt, std::vector<std::int64_t>{2, 0, 1}, std::nullopt};
                CHECK(meta.is_valid());
            }

            SUBCASE("valid with uniform_shape only")
            {
                metadata meta{
                    std::nullopt,
                    std::nullopt,
                    std::vector<std::optional<std::int32_t>>{400, std::nullopt, 3}
                };
                CHECK(meta.is_valid());
            }

            SUBCASE("valid with all fields")
            {
                metadata meta{
                    std::vector<std::string>{"H", "W", "C"},
                    std::vector<std::int64_t>{0, 1, 2},
                    std::vector<std::optional<std::int32_t>>{400, std::nullopt, 3}
                };
                CHECK(meta.is_valid());
            }

            SUBCASE("invalid - mismatched dim_names and permutation sizes")
            {
                metadata meta{
                    std::vector<std::string>{"C", "H"},
                    std::vector<std::int64_t>{2, 0, 1},
                    std::nullopt
                };
                CHECK_FALSE(meta.is_valid());
            }

            SUBCASE("invalid - mismatched dim_names and uniform_shape sizes")
            {
                metadata meta{
                    std::vector<std::string>{"H", "W", "C"},
                    std::nullopt,
                    std::vector<std::optional<std::int32_t>>{400, std::nullopt}
                };
                CHECK_FALSE(meta.is_valid());
            }

            SUBCASE("invalid - empty permutation")
            {
                metadata meta{std::nullopt, std::vector<std::int64_t>{}, std::nullopt};
                CHECK_FALSE(meta.is_valid());
            }

            SUBCASE("invalid - permutation with duplicate values")
            {
                metadata meta{std::nullopt, std::vector<std::int64_t>{0, 0, 1}, std::nullopt};
                CHECK_FALSE(meta.is_valid());
            }

            SUBCASE("invalid - permutation out of range")
            {
                metadata meta{std::nullopt, std::vector<std::int64_t>{0, 1, 3}, std::nullopt};
                CHECK_FALSE(meta.is_valid());
            }

            SUBCASE("invalid - negative value in permutation")
            {
                metadata meta{std::nullopt, std::vector<std::int64_t>{-1, 0, 1}, std::nullopt};
                CHECK_FALSE(meta.is_valid());
            }

            SUBCASE("invalid - negative dimension in uniform_shape")
            {
                metadata meta{
                    std::nullopt,
                    std::nullopt,
                    std::vector<std::optional<std::int32_t>>{400, std::nullopt, -3}
                };
                CHECK_FALSE(meta.is_valid());
            }

            SUBCASE("invalid - zero dimension in uniform_shape")
            {
                metadata meta{
                    std::nullopt,
                    std::nullopt,
                    std::vector<std::optional<std::int32_t>>{0, std::nullopt, 3}
                };
                CHECK_FALSE(meta.is_valid());
            }
        }

        TEST_CASE("metadata::get_ndim")
        {
            SUBCASE("from dim_names")
            {
                metadata meta{std::vector<std::string>{"C", "H", "W"}, std::nullopt, std::nullopt};
                auto ndim = meta.get_ndim();
                REQUIRE(ndim.has_value());
                CHECK_EQ(*ndim, 3);
            }

            SUBCASE("from permutation")
            {
                metadata meta{std::nullopt, std::vector<std::int64_t>{2, 0, 1, 3}, std::nullopt};
                auto ndim = meta.get_ndim();
                REQUIRE(ndim.has_value());
                CHECK_EQ(*ndim, 4);
            }

            SUBCASE("from uniform_shape")
            {
                metadata meta{
                    std::nullopt,
                    std::nullopt,
                    std::vector<std::optional<std::int32_t>>{400, std::nullopt}
                };
                auto ndim = meta.get_ndim();
                REQUIRE(ndim.has_value());
                CHECK_EQ(*ndim, 2);
            }

            SUBCASE("no ndim available")
            {
                metadata meta{std::nullopt, std::nullopt, std::nullopt};
                auto ndim = meta.get_ndim();
                CHECK_FALSE(ndim.has_value());
            }
        }

        TEST_CASE("metadata::to_json")
        {
            SUBCASE("empty metadata")
            {
                metadata meta{std::nullopt, std::nullopt, std::nullopt};
                const std::string json = meta.to_json();
                CHECK_EQ(json, "{}");
            }

            SUBCASE("with dim_names only")
            {
                metadata meta{std::vector<std::string>{"C", "H", "W"}, std::nullopt, std::nullopt};
                const std::string json = meta.to_json();
                CHECK_EQ(json, R"({"dim_names":["C","H","W"]})");
            }

            SUBCASE("with permutation only")
            {
                metadata meta{std::nullopt, std::vector<std::int64_t>{2, 0, 1}, std::nullopt};
                const std::string json = meta.to_json();
                CHECK_EQ(json, R"({"permutation":[2,0,1]})");
            }

            SUBCASE("with uniform_shape only")
            {
                metadata meta{
                    std::nullopt,
                    std::nullopt,
                    std::vector<std::optional<std::int32_t>>{400, std::nullopt, 3}
                };
                const std::string json = meta.to_json();
                CHECK_EQ(json, R"({"uniform_shape":[400,null,3]})");
            }

            SUBCASE("with dim_names and uniform_shape")
            {
                metadata meta{
                    std::vector<std::string>{"H", "W", "C"},
                    std::nullopt,
                    std::vector<std::optional<std::int32_t>>{400, std::nullopt, 3}
                };
                const std::string json = meta.to_json();
                CHECK_EQ(json, R"({"dim_names":["H","W","C"],"uniform_shape":[400,null,3]})");
            }

            SUBCASE("with all fields")
            {
                metadata meta{
                    std::vector<std::string>{"X", "Y", "Z"},
                    std::vector<std::int64_t>{2, 0, 1},
                    std::vector<std::optional<std::int32_t>>{std::nullopt, 10, std::nullopt}
                };
                const std::string json = meta.to_json();
                CHECK_EQ(
                    json,
                    R"({"dim_names":["X","Y","Z"],"permutation":[2,0,1],"uniform_shape":[null,10,null]})"
                );
            }
        }

        TEST_CASE("metadata::from_json")
        {
            SUBCASE("empty JSON")
            {
                const std::string json = "{}";
                const metadata meta = metadata::from_json(json);
                CHECK(meta.is_valid());
                CHECK_FALSE(meta.dim_names.has_value());
                CHECK_FALSE(meta.permutation.has_value());
                CHECK_FALSE(meta.uniform_shape.has_value());
            }

            SUBCASE("with dim_names")
            {
                const std::string json = R"({"dim_names":["C","H","W"]})";
                const metadata meta = metadata::from_json(json);
                CHECK(meta.is_valid());
                REQUIRE(meta.dim_names.has_value());
                REQUIRE_EQ(meta.dim_names->size(), 3);
                CHECK_EQ((*meta.dim_names)[0], "C");
                CHECK_EQ((*meta.dim_names)[1], "H");
                CHECK_EQ((*meta.dim_names)[2], "W");
                CHECK_FALSE(meta.permutation.has_value());
                CHECK_FALSE(meta.uniform_shape.has_value());
            }

            SUBCASE("with permutation")
            {
                const std::string json = R"({"permutation":[2,0,1]})";
                const metadata meta = metadata::from_json(json);
                CHECK(meta.is_valid());
                CHECK_FALSE(meta.dim_names.has_value());
                REQUIRE(meta.permutation.has_value());
                REQUIRE_EQ(meta.permutation->size(), 3);
                CHECK_EQ((*meta.permutation)[0], 2);
                CHECK_EQ((*meta.permutation)[1], 0);
                CHECK_EQ((*meta.permutation)[2], 1);
                CHECK_FALSE(meta.uniform_shape.has_value());
            }

            SUBCASE("with uniform_shape")
            {
                const std::string json = R"({"uniform_shape":[400,null,3]})";
                const metadata meta = metadata::from_json(json);
                CHECK(meta.is_valid());
                CHECK_FALSE(meta.dim_names.has_value());
                CHECK_FALSE(meta.permutation.has_value());
                REQUIRE(meta.uniform_shape.has_value());
                REQUIRE_EQ(meta.uniform_shape->size(), 3);
                REQUIRE((*meta.uniform_shape)[0].has_value());
                CHECK_EQ(*(*meta.uniform_shape)[0], 400);
                CHECK_FALSE((*meta.uniform_shape)[1].has_value());
                REQUIRE((*meta.uniform_shape)[2].has_value());
                CHECK_EQ(*(*meta.uniform_shape)[2], 3);
            }

            SUBCASE("with all fields")
            {
                const std::string json =
                    R"({"dim_names":["H","W","C"],"permutation":[0,1,2],"uniform_shape":[400,null,3]})";
                const metadata meta = metadata::from_json(json);
                CHECK(meta.is_valid());
                REQUIRE(meta.dim_names.has_value());
                CHECK_EQ(meta.dim_names->size(), 3);
                REQUIRE(meta.permutation.has_value());
                CHECK_EQ(meta.permutation->size(), 3);
                REQUIRE(meta.uniform_shape.has_value());
                CHECK_EQ(meta.uniform_shape->size(), 3);
            }

            SUBCASE("with whitespace")
            {
                const std::string json = R"(  {  "dim_names"  : [ "X" , "Y" ]  }  )";
                const metadata meta = metadata::from_json(json);
                CHECK(meta.is_valid());
                REQUIRE(meta.dim_names.has_value());
                REQUIRE_EQ(meta.dim_names->size(), 2);
            }

            SUBCASE("invalid - malformed JSON")
            {
                const std::string json = R"({"dim_names":["C","H","W")";
                CHECK_THROWS_AS(metadata::from_json(json), std::runtime_error);
            }
        }

        TEST_CASE("metadata::round-trip serialization")
        {
            SUBCASE("empty metadata")
            {
                metadata original{std::nullopt, std::nullopt, std::nullopt};
                const std::string json = original.to_json();
                const metadata parsed = metadata::from_json(json);
                CHECK(parsed.dim_names == original.dim_names);
                CHECK(parsed.permutation == original.permutation);
                CHECK(parsed.uniform_shape == original.uniform_shape);
            }

            SUBCASE("with all fields")
            {
                metadata original{
                    std::vector<std::string>{"H", "W", "C"},
                    std::vector<std::int64_t>{2, 0, 1},
                    std::vector<std::optional<std::int32_t>>{400, std::nullopt, 3}
                };
                const std::string json = original.to_json();
                const metadata parsed = metadata::from_json(json);
                CHECK(parsed.dim_names == original.dim_names);
                CHECK(parsed.permutation == original.permutation);
                CHECK(parsed.uniform_shape == original.uniform_shape);
            }
        }

        TEST_CASE("variable_shape_tensor_array::child_accessors")
        {
            // Create simple 2D tensors with shape [2, 3] and [1, 4]
            const std::uint64_t ndim = 2;
            
            // Create data arrays - List<Float32>
            sparrow::primitive_array<float> flat_data_all({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
            std::vector<std::size_t> offsets = {0, 6, 10};
            
            sparrow::primitive_array<float> values_copy(flat_data_all);
            sparrow::list_array tensor_data(sparrow::array(std::move(values_copy)), std::move(offsets));

            // Create shapes - FixedSizedList<Int32>[2]
            sparrow::primitive_array<std::int32_t> flat_shapes({2, 3, 1, 4});
            sparrow::fixed_sized_list_array tensor_shapes(
                2,  // list_size = ndim
                sparrow::array(std::move(flat_shapes))
            );

            metadata meta{std::nullopt, std::nullopt, std::nullopt};
            
            variable_shape_tensor_array tensor_array(
                ndim,
                sparrow::array(std::move(tensor_data)),
                sparrow::array(std::move(tensor_shapes)),
                meta
            );

            SUBCASE("data_child const access")
            {
                const auto& const_array = tensor_array;
                const auto* data_wrapper = const_array.data_child();
                REQUIRE(data_wrapper != nullptr);
            }

            SUBCASE("data_child mutable access")
            {
                auto* data_wrapper = tensor_array.data_child();
                REQUIRE(data_wrapper != nullptr);
            }

            SUBCASE("shape_child const access")
            {
                const auto& const_array = tensor_array;
                const auto* shape_wrapper = const_array.shape_child();
                REQUIRE(shape_wrapper != nullptr);
            }

            SUBCASE("shape_child mutable access")
            {
                auto* shape_wrapper = tensor_array.shape_child();
                REQUIRE(shape_wrapper != nullptr);
            }
        }

        TEST_CASE("variable_shape_tensor_array::basic_operations")
        {
            // Create simple 1D tensors
            const std::uint64_t ndim = 1;
            
            // Create data arrays - List<Int32> with 2 tensors
            sparrow::primitive_array<std::int32_t> flat_data({1, 2, 3, 4, 5});
            std::vector<std::size_t> offsets = {0, 3, 5};
            sparrow::list_array tensor_data(sparrow::array(std::move(flat_data)), std::move(offsets));

            // Create shapes - FixedSizedList<Int32>[1]
            sparrow::primitive_array<std::int32_t> flat_shapes({3, 2});
            sparrow::fixed_sized_list_array tensor_shapes(
                1,  // list_size = ndim
                sparrow::array(std::move(flat_shapes))
            );

            metadata meta{std::nullopt, std::nullopt, std::nullopt};
            
            variable_shape_tensor_array tensor_array(
                ndim,
                sparrow::array(std::move(tensor_data)),
                sparrow::array(std::move(tensor_shapes)),
                meta
            );

            SUBCASE("size")
            {
                CHECK_EQ(tensor_array.size(), 2);
            }

            SUBCASE("ndim")
            {
                auto ndim_result = tensor_array.ndim();
                CHECK_FALSE(ndim_result.has_value());  // metadata doesn't specify ndim
            }

            SUBCASE("get_metadata")
            {
                const auto& retrieved_meta = tensor_array.get_metadata();
                CHECK_FALSE(retrieved_meta.dim_names.has_value());
                CHECK_FALSE(retrieved_meta.permutation.has_value());
                CHECK_FALSE(retrieved_meta.uniform_shape.has_value());
            }

            SUBCASE("storage access")
            {
                const auto& storage = tensor_array.storage();
                CHECK_EQ(storage.size(), 2);
                
                auto& mutable_storage = tensor_array.storage();
                CHECK_EQ(mutable_storage.size(), 2);
            }

            SUBCASE("get_arrow_proxy")
            {
                const auto& const_proxy = tensor_array.get_arrow_proxy();
                CHECK_EQ(const_proxy.length(), 2);
                
                auto& mutable_proxy = tensor_array.get_arrow_proxy();
                CHECK_EQ(mutable_proxy.length(), 2);
            }
        }

        TEST_CASE("variable_shape_tensor_array::with_metadata")
        {
            const std::uint64_t ndim = 3;
            
            // Create data arrays
            sparrow::primitive_array<float> flat_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
            std::vector<std::size_t> offsets = {0, 6};
            sparrow::list_array tensor_data(sparrow::array(std::move(flat_data)), std::move(offsets));

            // Create shapes
            sparrow::primitive_array<std::int32_t> flat_shapes({2, 1, 3});
            sparrow::fixed_sized_list_array tensor_shapes(
                3,
                sparrow::array(std::move(flat_shapes))
            );

            metadata meta{
                std::vector<std::string>{"H", "W", "C"},
                std::nullopt,
                std::vector<std::optional<std::int32_t>>{std::nullopt, std::nullopt, 3}
            };
            
            variable_shape_tensor_array tensor_array(
                ndim,
                sparrow::array(std::move(tensor_data)),
                sparrow::array(std::move(tensor_shapes)),
                meta
            );

            SUBCASE("metadata preserved")
            {
                const auto& retrieved_meta = tensor_array.get_metadata();
                REQUIRE(retrieved_meta.dim_names.has_value());
                CHECK_EQ(retrieved_meta.dim_names->size(), 3);
                CHECK_EQ((*retrieved_meta.dim_names)[0], "H");
                CHECK_EQ((*retrieved_meta.dim_names)[1], "W");
                CHECK_EQ((*retrieved_meta.dim_names)[2], "C");
            }

            SUBCASE("ndim from metadata")
            {
                auto ndim_result = tensor_array.ndim();
                REQUIRE(ndim_result.has_value());
                CHECK_EQ(*ndim_result, 3);
            }
        }

        TEST_CASE("variable_shape_tensor_array::with_validity_bitmap")
        {
            const std::uint64_t ndim = 1;
            
            // Create data arrays with 3 tensors
            sparrow::primitive_array<std::int32_t> flat_data({1, 2, 3, 4, 5, 6});
            std::vector<std::size_t> offsets = {0, 2, 4, 6};
            sparrow::list_array tensor_data(sparrow::array(std::move(flat_data)), std::move(offsets));

            // Create shapes
            sparrow::primitive_array<std::int32_t> flat_shapes({2, 2, 2});
            sparrow::fixed_sized_list_array tensor_shapes(
                1,
                sparrow::array(std::move(flat_shapes))
            );

            metadata meta{std::nullopt, std::nullopt, std::nullopt};
            
            // Create with validity bitmap - mark middle tensor as null
            std::vector<bool> validity{true, false, true};
            
            variable_shape_tensor_array tensor_array(
                ndim,
                sparrow::array(std::move(tensor_data)),
                sparrow::array(std::move(tensor_shapes)),
                meta,
                std::move(validity)
            );

            SUBCASE("size with validity")
            {
                CHECK_EQ(tensor_array.size(), 3);
            }
        }

        TEST_CASE("variable_shape_tensor_array::with_name_and_arrow_metadata")
        {
            const std::uint64_t ndim = 2;
            
            // Create minimal tensor array
            sparrow::primitive_array<std::int32_t> flat_data({1, 2});
            std::vector<std::size_t> offsets = {0, 2};
            sparrow::list_array tensor_data(sparrow::array(std::move(flat_data)), std::move(offsets));

            sparrow::primitive_array<std::int32_t> flat_shapes({1, 2});
            sparrow::fixed_sized_list_array tensor_shapes(
                2,
                sparrow::array(std::move(flat_shapes))
            );

            metadata meta{std::nullopt, std::nullopt, std::nullopt};
            
            std::vector<sparrow::metadata_pair> arrow_meta = {
                {"custom_key", "custom_value"}
            };
            
            variable_shape_tensor_array tensor_array(
                ndim,
                sparrow::array(std::move(tensor_data)),
                sparrow::array(std::move(tensor_shapes)),
                meta,
                "my_tensor_array",
                arrow_meta
            );

            SUBCASE("name preserved")
            {
                const auto& proxy = tensor_array.get_arrow_proxy();
                CHECK_EQ(proxy.name(), "my_tensor_array");
            }

            SUBCASE("arrow metadata preserved")
            {
                const auto& proxy = tensor_array.get_arrow_proxy();
                const auto metadata_opt = proxy.metadata();
                REQUIRE(metadata_opt.has_value());
                
                // Should have extension metadata plus custom metadata
                bool found_custom = false;
                for (const auto& [key, value] : *metadata_opt)
                {
                    if (key == "custom_key" && value == "custom_value")
                    {
                        found_custom = true;
                        break;
                    }
                }
                CHECK(found_custom);
            }
        }

        TEST_CASE("variable_shape_tensor_array::inner_typedefs")
        {
            // Verify that the inner typedefs are properly defined
            using inner_val = variable_shape_tensor_array::inner_value_type;
            using inner_ref = variable_shape_tensor_array::inner_reference;
            using inner_const_ref = variable_shape_tensor_array::inner_const_reference;
            
            // These should all be sparrow::struct_value
            CHECK(std::is_same_v<inner_val, sparrow::struct_value>);
            CHECK(std::is_same_v<inner_ref, sparrow::struct_value>);
            CHECK(std::is_same_v<inner_const_ref, sparrow::struct_value>);
        }

        TEST_CASE("variable_shape_tensor_array::empty")
        {
            SUBCASE("empty array")
            {
                // Create empty array
                sparrow::primitive_array<std::int32_t> flat_data({});
                std::vector<std::size_t> offsets = {0};
                sparrow::list_array tensor_data(sparrow::array(std::move(flat_data)), std::move(offsets));

                sparrow::primitive_array<std::int32_t> flat_shapes({});
                sparrow::fixed_sized_list_array tensor_shapes(1, sparrow::array(std::move(flat_shapes)));

                metadata meta{std::nullopt, std::nullopt, std::nullopt};
                
                variable_shape_tensor_array tensor_array(
                    1,
                    sparrow::array(std::move(tensor_data)),
                    sparrow::array(std::move(tensor_shapes)),
                    meta
                );

                CHECK(tensor_array.empty());
                CHECK_EQ(tensor_array.size(), 0);
            }

            SUBCASE("non-empty array")
            {
                sparrow::primitive_array<std::int32_t> flat_data({1, 2});
                std::vector<std::size_t> offsets = {0, 2};
                sparrow::list_array tensor_data(sparrow::array(std::move(flat_data)), std::move(offsets));

                sparrow::primitive_array<std::int32_t> flat_shapes({2});
                sparrow::fixed_sized_list_array tensor_shapes(1, sparrow::array(std::move(flat_shapes)));

                metadata meta{std::nullopt, std::nullopt, std::nullopt};
                
                variable_shape_tensor_array tensor_array(
                    1,
                    sparrow::array(std::move(tensor_data)),
                    sparrow::array(std::move(tensor_shapes)),
                    meta
                );

                CHECK_FALSE(tensor_array.empty());
                CHECK_EQ(tensor_array.size(), 1);
            }
        }

        TEST_CASE("variable_shape_tensor_array::at")
        {
            sparrow::primitive_array<std::int32_t> flat_data({1, 2, 3, 4, 5, 6});
            std::vector<std::size_t> offsets = {0, 2, 4, 6};
            sparrow::list_array tensor_data(sparrow::array(std::move(flat_data)), std::move(offsets));

            sparrow::primitive_array<std::int32_t> flat_shapes({2, 2, 2});
            sparrow::fixed_sized_list_array tensor_shapes(1, sparrow::array(std::move(flat_shapes)));

            metadata meta{std::nullopt, std::nullopt, std::nullopt};
            
            variable_shape_tensor_array tensor_array(
                1,
                sparrow::array(std::move(tensor_data)),
                sparrow::array(std::move(tensor_shapes)),
                meta
            );

            SUBCASE("valid access")
            {
                auto elem0 = tensor_array.at(0);
                CHECK(elem0.has_value());
                
                auto elem1 = tensor_array.at(1);
                CHECK(elem1.has_value());
                
                auto elem2 = tensor_array.at(2);
                CHECK(elem2.has_value());
            }

            SUBCASE("out of range")
            {
                CHECK_THROWS_AS(tensor_array.at(3), std::out_of_range);
                CHECK_THROWS_AS(tensor_array.at(10), std::out_of_range);
            }
        }

        TEST_CASE("variable_shape_tensor_array::is_valid")
        {
            SUBCASE("valid array")
            {
                sparrow::primitive_array<std::int32_t> flat_data({1, 2});
                std::vector<std::size_t> offsets = {0, 2};
                sparrow::list_array tensor_data(sparrow::array(std::move(flat_data)), std::move(offsets));

                sparrow::primitive_array<std::int32_t> flat_shapes({2});
                sparrow::fixed_sized_list_array tensor_shapes(1, sparrow::array(std::move(flat_shapes)));

                metadata meta{std::nullopt, std::nullopt, std::nullopt};
                
                variable_shape_tensor_array tensor_array(
                    1,
                    sparrow::array(std::move(tensor_data)),
                    sparrow::array(std::move(tensor_shapes)),
                    meta
                );

                CHECK(tensor_array.is_valid());
            }

            SUBCASE("valid array with metadata")
            {
                sparrow::primitive_array<float> flat_data({1.0f, 2.0f, 3.0f});
                std::vector<std::size_t> offsets = {0, 3};
                sparrow::list_array tensor_data(sparrow::array(std::move(flat_data)), std::move(offsets));

                sparrow::primitive_array<std::int32_t> flat_shapes({1, 3});
                sparrow::fixed_sized_list_array tensor_shapes(2, sparrow::array(std::move(flat_shapes)));

                metadata meta{
                    std::vector<std::string>{"H", "W"},
                    std::nullopt,
                    std::nullopt
                };
                
                variable_shape_tensor_array tensor_array(
                    2,
                    sparrow::array(std::move(tensor_data)),
                    sparrow::array(std::move(tensor_shapes)),
                    meta
                );

                CHECK(tensor_array.is_valid());
            }
        }

        TEST_CASE("variable_shape_tensor_array::field_names")
        {
            CHECK_EQ(variable_shape_tensor_array::data_field_name(), "data");
            CHECK_EQ(variable_shape_tensor_array::shape_field_name(), "shape");
        }

        TEST_CASE("variable_shape_tensor_array::iterators")
        {
            sparrow::primitive_array<std::int32_t> flat_data({1, 2, 3, 4, 5, 6});
            std::vector<std::size_t> offsets = {0, 2, 4, 6};
            sparrow::list_array tensor_data(sparrow::array(std::move(flat_data)), std::move(offsets));

            sparrow::primitive_array<std::int32_t> flat_shapes({2, 2, 2});
            sparrow::fixed_sized_list_array tensor_shapes(1, sparrow::array(std::move(flat_shapes)));

            metadata meta{std::nullopt, std::nullopt, std::nullopt};
            
            variable_shape_tensor_array tensor_array(
                1,
                sparrow::array(std::move(tensor_data)),
                sparrow::array(std::move(tensor_shapes)),
                meta
            );

            SUBCASE("begin and end")
            {
                auto it_begin = tensor_array.begin();
                auto it_end = tensor_array.end();
                CHECK(it_begin != it_end);
            }

            SUBCASE("cbegin and cend")
            {
                auto it_begin = tensor_array.cbegin();
                auto it_end = tensor_array.cend();
                CHECK(it_begin != it_end);
            }

            SUBCASE("range-based for loop")
            {
                size_t count = 0;
                for (const auto& tensor : tensor_array)
                {
                    CHECK(tensor.has_value());
                    ++count;
                }
                CHECK_EQ(count, 3);
            }

            SUBCASE("iterator distance")
            {
                auto distance = std::distance(tensor_array.begin(), tensor_array.end());
                CHECK_EQ(static_cast<size_t>(distance), tensor_array.size());
            }
        }
    }
}  // namespace sparrow_extensions
