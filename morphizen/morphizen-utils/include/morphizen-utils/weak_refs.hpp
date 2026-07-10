/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include <cassert>
#include <memory>
#include <type_traits>
#include <unordered_map>

namespace morphizen::utils {

/**
 * @brief Weak singleton pattern implementation
 *
 * This template provides a singleton that doesn't prevent the object from
 * being destroyed when no other references exist. Useful for avoiding
 * static destruction order issues and memory leaks.
 *
 * @tparam T Type of the singleton object
 */
template <typename T> struct WeakSingleton {
  /// Weak reference to the singleton instance
  static std::weak_ptr<T> the_instance_;

  /**
   * @brief Create or retrieve the singleton instance
   * @param args Arguments to pass to T's constructor
   * @return Shared pointer to the singleton instance
   */
  template <typename... Args> static std::shared_ptr<T> create(Args &&...args) {
    std::shared_ptr<T> ret;
    if (the_instance_.expired()) {
      ret = std::make_shared<T>(std::forward<Args>(args)...);
      the_instance_ = ret;
    } else {
      ret = the_instance_.lock();
    }
    assert(ret != nullptr);
    return ret;
  }
};

// Static member definition
template <typename T> std::weak_ptr<T> WeakSingleton<T>::the_instance_;

// SFINAE utilities (C++17 void_t replacement)
template <class...> using void_t = void;

/**
 * @brief SFINAE helper to detect if a type has an initialize() method
 */
template <typename T, class = void>
struct has_initialize_method : std::false_type {};

template <typename T>
struct has_initialize_method<T,
                             void_t<decltype(std::declval<T>().initialize())>>
    : std::true_type {};

/**
 * @brief Forward declaration for WithInjection detection
 */
template <typename T> struct WithInjection;

/**
 * @brief SFINAE helper to detect if a type derives from WithInjection
 */
template <typename T>
using is_derived_from_with_injection =
    std::enable_if_t<std::is_base_of_v<WithInjection<T>, T>>;

template <typename T>
using is_not_derived_from_with_injection =
    std::enable_if_t<!std::is_base_of_v<WithInjection<T>, T>>;

/**
 * @brief Helper to conditionally call initialize() method
 *
 * This template provides conditional initialization logic:
 * - If T derives from WithInjection<T>, initialization is handled elsewhere
 * - If T has an initialize() method, call it
 * - Otherwise, do nothing
 */
template <typename T, class = void> struct invoke_initialize_if_possible {
  static void initialize(T * /*t*/) {
    // Default: do nothing
  }
};

// Specialization for types derived from WithInjection
template <typename T>
struct invoke_initialize_if_possible<T, is_derived_from_with_injection<T>> {
  static void initialize(T * /*t*/) {
    // WithInjection<T>::create(...) invokes initialize() already,
    // don't invoke it twice
  }
};

// Specialization for types with initialize() method (but not WithInjection)
template <typename T>
struct invoke_initialize_if_possible<
    T, std::enable_if_t<has_initialize_method<T>::value &&
                        !std::is_base_of_v<WithInjection<T>, T>>> {
  static void initialize(T *t) { t->initialize(); }
};

/**
 * @brief Weak storage for key-value pairs of objects
 *
 * This template provides a weak reference store that maps keys to objects.
 * Objects are automatically removed when no other references exist.
 *
 * @tparam K Key type
 * @tparam T Value type (must be constructible or have a create() method)
 */
template <typename K, typename T> struct WeakStore {
  /// Map of keys to weak references
  static std::unordered_map<K, std::weak_ptr<T>> the_store_;

  /**
   * @brief Create or retrieve an object by key
   * @param key Key to identify the object
   * @param args Arguments for object construction
   * @return Shared pointer to the object
   */
  template <typename... Args>
  static std::shared_ptr<T> create(const K &key, Args &&...args) {
    std::shared_ptr<T> ret;
    auto &weak_ref = the_store_[key];

    if (weak_ref.expired()) {
      ret = create_impl(std::forward<Args>(args)...);
      invoke_initialize_if_possible<T>::initialize(ret.get());
      weak_ref = ret;
    } else {
      ret = weak_ref.lock();
    }

    assert(ret != nullptr);
    return ret;
  }

  /**
   * @brief Get object by key without creating
   * @param key Key to look up
   * @return Shared pointer to object if it exists, nullptr otherwise
   */
  static std::shared_ptr<T> get(const K &key) {
    auto it = the_store_.find(key);
    if (it != the_store_.end() && !it->second.expired()) {
      return it->second.lock();
    }
    return nullptr;
  }

  /**
   * @brief Remove expired entries from the store
   */
  static void cleanup() {
    for (auto it = the_store_.begin(); it != the_store_.end();) {
      if (it->second.expired()) {
        it = the_store_.erase(it);
      } else {
        ++it;
      }
    }
  }

  /**
   * @brief Get the number of active entries
   */
  static size_t size() {
    cleanup();
    return the_store_.size();
  }

private:
  // Use T::create() if T is not directly constructible
  template <typename... Args>
  static std::enable_if_t<!std::is_constructible_v<T, Args...>,
                          std::shared_ptr<T>>
  create_impl(Args &&...args) {
    return T::create(std::forward<Args>(args)...);
  }

  // Use direct construction if T is constructible
  template <typename... Args>
  static std::enable_if_t<std::is_constructible_v<T, Args...>,
                          std::shared_ptr<T>>
  create_impl(Args &&...args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
  }
};

// Static member definition
template <typename K, typename T>
std::unordered_map<K, std::weak_ptr<T>> WeakStore<K, T>::the_store_;

} // namespace morphizen::utils
