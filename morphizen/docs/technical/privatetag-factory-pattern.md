<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# PrivateTag Pattern: Factory Methods with std::make_unique

## The Problem

When implementing a **factory pattern** in C++, you want to:
1. ✅ Force users to call factory methods (e.g., `create()`)
2. ✅ Prevent direct construction (e.g., `new TarFile(...)`)
3. ✅ Use `std::make_unique` inside factory methods (efficient, exception-safe)

But there's a conflict:
- ❌ If constructor is **private** → `std::make_unique` won't work
- ❌ If constructor is **public** → Users can bypass factory methods

**Why `std::make_unique` can't access private constructors:**

`std::make_unique` is a **template function in the standard library** - it's not a member of your class, and it's not a friend. When you call `std::make_unique<TarFile>(args...)`, it's just like any other external code trying to call your constructor. If the constructor is private, it fails.

```cpp
class TarFile {
private:
  TarFile(std::unique_ptr<std::iostream>&& stream);  // Private
};

// In factory method:
return std::make_unique<TarFile>(std::move(stream));  // ❌ Compile error!
// std::make_unique is external code, can't access private constructor
```

You **could** make `std::make_unique` a friend, but that's verbose and couples your class to the standard library implementation.

## The PrivateTag Solution

**Pattern:**
```cpp
class TarFile {
public:
  // Factory methods
  static std::unique_ptr<TarFile> create(std::unique_ptr<std::iostream>&& stream) {
    return std::make_unique<TarFile>(PrivateTag{}, std::move(stream));
  }

private:
  struct PrivateTag {};  // Private tag type

public:
  // Constructor is public BUT requires PrivateTag
  TarFile(PrivateTag, std::unique_ptr<std::iostream>&& stream)
      : stream_(std::move(stream)) {}

private:
  std::unique_ptr<std::iostream> stream_;
};
```

**How it works:**

The key insight: **Use access control on the TYPE (PrivateTag), not the constructor itself.**

1. Constructor is **public** (so `std::make_unique` can call it)
2. BUT requires a `PrivateTag` parameter
3. `PrivateTag` type is **private** (only `TarFile` can access it)
4. External callers **cannot construct PrivateTag** → cannot call constructor
5. Factory methods **can construct PrivateTag{}** → can use `std::make_unique`

The constructor is visible to everyone, but only TarFile can create the "key" (PrivateTag instance) needed to call it.

**Result:**
```cpp
// ✅ Works - factory method
auto tar = TarFile::create(std::move(stream));

// ❌ Compile error - PrivateTag is private
auto tar2 = std::make_unique<TarFile>(???, std::move(stream));
//                                    ^^^ can't create PrivateTag
```

## Alternative Approaches

### Alternative 1: Friend Declaration

```cpp
class TarFile {
public:
  static std::unique_ptr<TarFile> create(std::unique_ptr<std::iostream>&& stream);

private:
  TarFile(std::unique_ptr<std::iostream>&& stream);

  // Make std::make_unique a friend
  template<typename T, typename... Args>
  friend std::unique_ptr<T> std::make_unique(Args&&...);
};
```

**Pros:**
- ✅ Constructor is truly private
- ✅ Cleaner API (no PrivateTag parameter)
- ✅ Works in C++11+

**Cons:**
- ❌ Verbose friend declaration
- ❌ Exposes implementation detail (couples to std::make_unique)
- ❌ Fragile - breaks if standard library implementation changes

### Alternative 2: Passkey Idiom

```cpp
class TarFile {
public:
  class Key {
  private:
    Key() {}
    friend class TarFile;
  };

  static std::unique_ptr<TarFile> create(std::unique_ptr<std::iostream>&& stream) {
    return std::make_unique<TarFile>(Key{}, std::move(stream));
  }

  // Public constructor with Key parameter
  TarFile(Key, std::unique_ptr<std::iostream>&& stream);
};
```

**Pros:**
- ✅ More explicit intent (Key conveys "access token")
- ✅ Can grant access to specific external classes

**Cons:**
- ❌ More boilerplate (separate Key class)
- ❌ Same as PrivateTag but more complex

### Alternative 3: New + Private Constructor

```cpp
class TarFile {
public:
  static std::unique_ptr<TarFile> create(std::unique_ptr<std::iostream>&& stream) {
    return std::unique_ptr<TarFile>(new TarFile(std::move(stream)));
  }

private:
  TarFile(std::unique_ptr<std::iostream>&& stream);
};
```

**Pros:**
- ✅ Constructor is truly private
- ✅ No extra types needed

**Cons:**
- ❌ Uses `new` instead of `make_unique` (less safe, slightly less efficient)
- ❌ Doesn't get make_unique's exception safety benefits

## Comparison Summary

| Approach | Constructor | make_unique | Boilerplate | C++ Standard |
|----------|-------------|-------------|-------------|--------------|
| **PrivateTag** | Public w/ tag | ✅ Yes | Low (empty struct) | C++11 |
| Friend declaration | Private | ✅ Yes | Medium (template friend) | C++11 |
| Passkey idiom | Public w/ key | ✅ Yes | High (Key class) | C++11 |
| New + private ctor | Private | ❌ No | None | C++98 |

## When to Use PrivateTag

**Use PrivateTag when:**
- ✅ You want factory pattern enforcement
- ✅ You want to use `std::make_unique` (exception safety, efficiency)
- ✅ You need C++11 compatibility
- ✅ You want minimal boilerplate

**Consider alternatives when:**
- Passkey: Need to grant construction access to external classes
- Friend: Want cleaner API, willing to couple to std::make_unique implementation
- New: Don't care about make_unique benefits (exception safety, efficiency)

## Example in MorphiZen Codebase

**File:** `morphizen-core/src/tar_file.hpp`

```cpp
class TarFile {
public:
  // 6 factory methods
  static std::unique_ptr<TarFile> create(std::unique_ptr<std::iostream>&& stream);
  static std::unique_ptr<TarFile> create_from_path(const std::filesystem::path&, bool);
  static std::unique_ptr<TarFile> create();  // tmpfile
  static std::unique_ptr<TarFile> create(std::vector<char>&& buffer);
  static std::unique_ptr<TarFile> create(std::string&& buffer);
  static std::unique_ptr<TarFile> create(const char* data, size_t size);

private:
  struct PrivateTag {};  // See docs/technical/privatetag-factory-pattern.md

public:
  TarFile(PrivateTag, std::unique_ptr<std::iostream>&& stream);

private:
  std::unique_ptr<std::iostream> stream_;
};
```

**Why PrivateTag here:**
- Multiple factory methods with different data sources
- Need efficient unique_ptr creation
- C++11 codebase (can't use C++20 friend declaration)
- Don't need external class construction access (Passkey not needed)

## References

- [std::make_unique - cppreference](https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique)
- [Factory Method Pattern](https://refactoring.guru/design-patterns/factory-method)
