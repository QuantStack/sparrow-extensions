Variable Shape Tensor Array               {#variable_shape_tensor_array}
==============================

Introduction
------------

The Variable Shape Tensor Array is an Arrow-compatible array for storing multi-dimensional tensors with variable shapes according to the [Apache Arrow canonical extension specification for VariableShapeTensor](https://arrow.apache.org/docs/format/CanonicalExtensions.html#variable-shape-tensor).

This extension enables efficient storage and transfer of tensors (multi-dimensional arrays) where each tensor can have a different shape. Each element in the array represents a complete tensor, and the shapes are stored alongside the tensor data. The underlying storage uses Arrow's `Struct` type with two fields:
- `data`: A `List` holding the flattened tensor elements
- `shape`: A `FixedSizeList<int32>` storing the dimensions of each tensor

The VariableShapeTensor extension type is defined as:
- Extension name: `arrow.variable_shape_tensor`
- Storage type: `Struct<data: List<T>, shape: FixedSizeList<int32>[ndim]>` where `T` is the value type
- Extension metadata: JSON object containing optional dimension names, permutation, and uniform shape
- Number of dimensions (`ndim`): Fixed for all tensors in the array, but individual dimension sizes can vary

Metadata Structure
------------------

The extension metadata is a JSON object with the following optional fields:

```json
{
  "dim_names": ["name0", "name1", ..., "nameN"],              // optional
  "permutation": [idx0, idx1, ..., idxN],                     // optional
  "uniform_shape": [size0_or_null, size1_or_null, ..., sizeN_or_null]  // optional
}
```

### Fields

All fields are optional:

- **dim_names** (optional): Array of strings naming each dimension. The length must equal `ndim`.
- **permutation** (optional): Array defining the physical-to-logical dimension mapping. Must be a valid permutation of [0, 1, ..., N-1] where N is `ndim`.
- **uniform_shape** (optional): Array specifying which dimensions are uniform (have the same size across all tensors). Uniform dimensions are represented by `int32` values, while non-uniform dimensions are represented by `null`. If not provided, all dimensions are assumed to be non-uniform.

**Note**: With the exception of `permutation`, the parameters and storage of VariableShapeTensor relate to the **physical storage** of the tensor. For example, if a tensor has:
- `shape = [10, 20, 30]`
- `dim_names = [x, y, z]`
- `permutations = [2, 0, 1]`

This means the logical tensor has names `[z, x, y]` and shape `[30, 10, 20]`.

**Note**: Values inside each data tensor element are stored in **row-major/C-contiguous order** according to the corresponding shape.

Usage
-----

### Basic Usage

```cpp
#include "sparrow_extensions/variable_shape_tensor.hpp"
using namespace sparrow_extensions;
using namespace sparrow;

// Create 3 tensors with different shapes, all 2D:
// Tensor 0: shape [2, 3] -> 6 elements
// Tensor 1: shape [3, 2] -> 6 elements
// Tensor 2: shape [1, 4] -> 4 elements

// Create data lists (one list per tensor)
list_array data_list(
    primitive_array<std::int64_t>(std::vector<std::int64_t>{0, 6, 12, 16}),  // offsets
    primitive_array<float>(std::vector<float>{
        // Tensor 0 data
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        // Tensor 1 data
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
        // Tensor 2 data
        12.0f, 13.0f, 14.0f, 15.0f
    }),
    std::vector<bool>{}
);

// Create shape lists (fixed size list of int32, ndim=2)
fixed_sized_list_array shape_list(
    2,  // ndim
    primitive_array<std::int32_t>(std::vector<std::int32_t>{
        2, 3,  // Tensor 0 shape
        3, 2,  // Tensor 1 shape
        1, 4   // Tensor 2 shape
    }),
    std::vector<bool>{}
);

variable_shape_tensor_extension::metadata tensor_meta{
    std::nullopt,  // no dimension names
    std::nullopt,  // no permutation
    std::nullopt   // no uniform_shape
};

variable_shape_tensor_array tensor_array(
    2,  // ndim
    array(std::move(data_list)),
    array(std::move(shape_list)),
    tensor_meta
);

// Access properties
std::cout << "Number of tensors: " << tensor_array.size() << "\n";  // 3

auto ndim = tensor_array.ndim();
if (ndim.has_value())
{
    std::cout << "Number of dimensions: " << *ndim << "\n";
}

// Access individual tensors
auto first_tensor = tensor_array[0];
if (first_tensor.has_value())
{
    // Process first tensor... (shape [2, 3])
}
```

### With Dimension Names (NCHW Example)

According to the specification, dimension names for NCHW ordered data where the first logical dimension N is mapped to the data List array (each element in the List is a CHW tensor):

```cpp
// Single CHW tensor per row (N is implicit in the list)
list_array data_list(
    primitive_array<std::int64_t>(std::vector<std::int64_t>{0, 24}),
    primitive_array<float>(std::vector<float>(24, 0.0f)),  // 3*4*2 elements
    std::vector<bool>{}
);

fixed_sized_list_array shape_list(
    3,  // ndim
    primitive_array<std::int32_t>(std::vector<std::int32_t>{3, 4, 2}),  // C, H, W
    std::vector<bool>{}
);

variable_shape_tensor_extension::metadata tensor_meta{
    std::vector<std::string>{"C", "H", "W"},
    std::nullopt,
    std::nullopt
};

variable_shape_tensor_array tensor_array(
    3,
    array(std::move(data_list)),
    array(std::move(shape_list)),
    tensor_meta
);

const auto& meta = tensor_array.get_metadata();
if (meta.dim_names.has_value())
{
    for (const auto& name : *meta.dim_names)
    {
        std::cout << name << " ";  // C H W
    }
}
```

### With Uniform Shape (Color Images)

Example with color images with fixed height (400), variable width, and three color channels:

```cpp
// Create 2 color images with fixed height and channels, variable width
list_array data_list(
    primitive_array<std::int64_t>(std::vector<std::int64_t>{0, 4800, 9600}),
    primitive_array<std::uint8_t>(std::vector<std::uint8_t>(9600, 0)),
    std::vector<bool>{}
);

fixed_sized_list_array shape_list(
    3,  // ndim
    primitive_array<std::int32_t>(std::vector<std::int32_t>{
        400, 4, 3,   // Image 0: 400x4x3 = 4800 pixels
        400, 8, 3    // Image 1: 400x8x3 = 9600 pixels
    }),
    std::vector<bool>{}
);

variable_shape_tensor_extension::metadata tensor_meta{
    std::vector<std::string>{"H", "W", "C"},
    std::nullopt,
    std::vector<std::optional<std::int32_t>>{400, std::nullopt, 3}  // H=400, W=variable, C=3
};

variable_shape_tensor_array tensor_array(
    3,
    array(std::move(data_list)),
    array(std::move(shape_list)),
    tensor_meta
);

const auto& meta = tensor_array.get_metadata();
if (meta.uniform_shape.has_value())
{
    std::cout << "Uniform shape: ";
    for (const auto& dim : *meta.uniform_shape)
    {
        if (dim.has_value())
        {
            std::cout << *dim << " ";
        }
        else
        {
            std::cout << "null ";
        }
    }
    std::cout << "\n";  // "400 null 3"
}
```

### With Permutation

Physical shape is [1, 2, 3] but logical layout is [3, 1, 2]:

```cpp
list_array data_list(
    primitive_array<std::int64_t>(std::vector<std::int64_t>{0, 6}),
    primitive_array<double>(std::vector<double>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0}),
    std::vector<bool>{}
);

fixed_sized_list_array shape_list(
    3,
    primitive_array<std::int32_t>(std::vector<std::int32_t>{1, 2, 3}),  // physical shape
    std::vector<bool>{}
);

variable_shape_tensor_extension::metadata tensor_meta{
    std::nullopt,
    std::vector<std::int64_t>{2, 0, 1},  // permutation: logical[i] = physical[perm[i]]
    std::nullopt
};

variable_shape_tensor_array tensor_array(
    3,
    array(std::move(data_list)),
    array(std::move(shape_list)),
    tensor_meta
);

// Physical shape: [1, 2, 3]
// Logical shape: [3, 1, 2]
const auto& meta = tensor_array.get_metadata();
if (meta.permutation.has_value())
{
    std::cout << "Permutation: ";
    for (auto idx : *meta.permutation)
    {
        std::cout << idx << " ";  // 2 0 1
    }
}
```

### With Array Name and Metadata

```cpp
// Create variable shape tensors with custom name and Arrow metadata
list_array data_list(
    primitive_array<std::int64_t>(std::vector<std::int64_t>{0, 4}),
    primitive_array<float>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}),
    std::vector<bool>{}
);

fixed_sized_list_array shape_list(
    2,
    primitive_array<std::int32_t>(std::vector<std::int32_t>{2, 2}),
    std::vector<bool>{}
);

variable_shape_tensor_extension::metadata tensor_meta{
    std::vector<std::string>{"rows", "cols"},
    std::nullopt,
    std::nullopt
};

std::vector<metadata_pair> arrow_metadata{
    {"author", "data_science_team"},
    {"version", "1.5"},
    {"dataset", "experiment_xyz"}
};

variable_shape_tensor_array tensor_array(
    2,
    array(std::move(data_list)),
    array(std::move(shape_list)),
    tensor_meta,
    "variable_tensor_data",  // array name
    arrow_metadata           // additional metadata
);

// Access the Arrow proxy to read metadata
const auto& proxy = tensor_array.get_arrow_proxy();
std::cout << "Array name: " << proxy.name() << "\n";

if (auto meta_opt = proxy.metadata())
{
    for (const auto& [key, value] : *meta_opt)
    {
        std::cout << key << ": " << value << "\n";
    }
}
```

### With Validity Bitmap

```cpp
// Create 3 tensors, mark the second as null
list_array data_list(
    primitive_array<std::int64_t>(std::vector<std::int64_t>{0, 2, 4, 6}),
    primitive_array<int32_t>(std::vector<int32_t>{1, 2, 3, 4, 5, 6}),
    std::vector<bool>{}
);

fixed_sized_list_array shape_list(
    1,
    primitive_array<std::int32_t>(std::vector<std::int32_t>{2, 2, 2}),
    std::vector<bool>{}
);

variable_shape_tensor_extension::metadata tensor_meta{
    std::nullopt,
    std::nullopt,
    std::nullopt
};

std::vector<bool> validity{true, false, true};  // Second tensor is null

variable_shape_tensor_array tensor_array(
    1,
    array(std::move(data_list)),
    array(std::move(shape_list)),
    tensor_meta,
    validity
);

// Check validity
const auto& storage = tensor_array.storage();
CHECK(storage[0].has_value());   // valid
CHECK(!storage[1].has_value());  // null
CHECK(storage[2].has_value());   // valid
```

API Reference
-------------

### `variable_shape_tensor_extension`

The extension class provides static methods for working with the Arrow extension metadata.

#### Methods

- `static void init(arrow_proxy& proxy, const metadata& tensor_metadata)`: Initializes extension metadata on an arrow proxy
- `static metadata extract_metadata(const arrow_proxy& proxy)`: Extracts metadata from an arrow proxy

### `variable_shape_tensor_extension::metadata`

Stores the optional metadata for variable shape tensors.

#### Fields

- `std::optional<std::vector<std::string>> dim_names`: Optional dimension names
- `std::optional<std::vector<std::int64_t>> permutation`: Optional dimension permutation
- `std::optional<std::vector<std::optional<std::int32_t>>> uniform_shape`: Optional uniform shape specification

#### Methods

- `bool is_valid() const`: Validates the metadata structure
- `std::optional<std::size_t> get_ndim() const`: Returns the number of dimensions if determinable
- `std::string to_json() const`: Serializes metadata to JSON
- `static metadata from_json(std::string_view json)`: Deserializes metadata from JSON

### `variable_shape_tensor_array`

The main array class for working with variable shape tensors.

#### Constructors

- `variable_shape_tensor_array(arrow_proxy proxy)`: Constructs from an arrow proxy
- `variable_shape_tensor_array(uint64_t ndim, array&& tensor_data, array&& tensor_shapes, const metadata_type& tensor_metadata)`: Constructs from data and shapes
- Additional overloads with name, metadata, and validity bitmap support

#### Methods

- `size_type size() const`: Returns the number of tensors
- `const metadata_type& get_metadata() const`: Returns the metadata
- `std::optional<std::size_t> ndim() const`: Returns the number of dimensions if determinable
- `const struct_array& storage() const`: Returns the underlying struct array
- `auto operator[](size_type i) const`: Access tensor at index i
- `const arrow_proxy& get_arrow_proxy() const`: Returns the underlying arrow proxy

Best Practices
--------------

1. **Consistent Dimensionality**: All tensors in an array must have the same number of dimensions (`ndim`), even if individual dimension sizes vary.

2. **Uniform Shape Optimization**: Use `uniform_shape` metadata when you know certain dimensions will remain constant across all tensors. This can enable optimizations in downstream processing.

3. **Row-Major Order**: Always provide tensor data in row-major (C-contiguous) order to ensure compatibility with the Arrow specification.

4. **Permutation for Performance**: Use the `permutation` field when the logical view of the data differs from the physical storage layout, rather than copying and rearranging the data.

5. **Dimension Names**: Provide `dim_names` to make the data self-documenting and easier to work with in data analysis tools.

6. **Memory Efficiency**: Variable shape tensors can be more memory-efficient than fixed shape tensors when the variation in sizes is significant, as they only store the data that's actually needed.

Examples
--------

### Time Series with Variable Length

```cpp
// Store time series data where each series has a different length
// All 1D tensors (ndim=1) but with varying lengths

list_array data_list(
    primitive_array<std::int64_t>(std::vector<std::int64_t>{0, 100, 250, 500}),  // offsets
    primitive_array<double>(std::vector<double>(500)),  // Fill with time series data
    std::vector<bool>{}
);

fixed_sized_list_array shape_list(
    1,  // 1D tensors
    primitive_array<std::int32_t>(std::vector<std::int32_t>{
        100,  // Series 0: 100 points
        150,  // Series 1: 150 points
        250   // Series 2: 250 points
    }),
    std::vector<bool>{}
);

variable_shape_tensor_extension::metadata tensor_meta{
    std::vector<std::string>{"time"},
    std::nullopt,
    std::nullopt
};

variable_shape_tensor_array tensor_array(
    1,
    array(std::move(data_list)),
    array(std::move(shape_list)),
    tensor_meta
);
```

### Variable Width Images with Fixed Channels

```cpp
// Store RGB images with variable dimensions but always 3 channels
list_array data_list(
    primitive_array<std::int64_t>(std::vector<std::int64_t>{0, 6000, 15000}),
    primitive_array<std::uint8_t>(std::vector<std::uint8_t>(15000)),
    std::vector<bool>{}
);

fixed_sized_list_array shape_list(
    3,
    primitive_array<std::int32_t>(std::vector<std::int32_t>{
        100, 20, 3,  // Image 0: 100x20x3 = 6000
        150, 20, 3   // Image 1: 150x20x3 = 9000
    }),
    std::vector<bool>{}
);

variable_shape_tensor_extension::metadata tensor_meta{
    std::vector<std::string>{"H", "W", "C"},
    std::nullopt,
    std::vector<std::optional<std::int32_t>>{std::nullopt, std::nullopt, 3}  // Only channels uniform
};

variable_shape_tensor_array tensor_array(
    3,
    array(std::move(data_list)),
    array(std::move(shape_list)),
    tensor_meta
);
```

See Also
--------

- [Fixed Shape Tensor Array](@ref fixed_shape_tensor_array) - For tensors with uniform shape
- [Apache Arrow VariableShapeTensor Specification](https://arrow.apache.org/docs/format/CanonicalExtensions.html#variable-shape-tensor)
