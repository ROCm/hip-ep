/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
 *
 *      Redistribution and use in binary form only, without modification, is
 * permitted provided that the following conditions are met:
 *
 *      1. Redistributions must reproduce the above copyright notice, this list
 * of conditions and the following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 *
 *      2. The name of Xilinx, Inc. may not be used to endorse or promote
 * products redistributed with this software without specific prior written
 * permission.
 *
 *      THIS SOFTWARE IS PROVIDED BY XILINX, INC. "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL XILINX, INC. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *      PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */
/**
 * @brief A template class representing an immutable map.
 *
 * @tparam key_type The type of the keys in the map.
 * @tparam value_t The type of the values in the map.
 */
/**
 * @file immutable_map.hpp
 * @brief Defines the ImmutableMap class template.
 */
#pragma once
#include <cassert>
#include <iostream>
#include <memory>
#include <utility>

namespace vaip_core {
namespace immutable_map {
template <typename Key, typename T, typename Compare> class ImmutableMap;

enum Color { RED, BLACK };
template <typename Key, typename T, class Compare = std::less<Key>> class Node {
public:
  using key_type = Key;
  using mapped_type = T;
  using value_type = std::pair<const key_type, mapped_type>;

  using NodePtr = std::shared_ptr<const Node<Key, T, Compare>>;
  Node(Color color, const NodePtr& left, const value_type& value,
       const NodePtr& right)
      : color(color), left(left), value(value), right(right) {}

private:
  const Color color;
  const NodePtr left;
  const value_type value;
  const NodePtr right;

private:
  static const NodePtr nil;
  friend class ImmutableMap<Key, T, Compare>;

  static NodePtr make_node(Color color, const NodePtr& left,
                           const value_type& value, const NodePtr& right) {
    return std::make_shared<Node>(Node{color, left, value, right});
  }

  static NodePtr ins(const NodePtr& x, const value_type& value) {
    Compare less;
    if (x == nil) {
      return make_node(RED, nil, value, nil);
    } else if (less(x->value.first, value.first)) {
      return balance(
          make_node(x->color, x->left, x->value, ins(x->right, value)));
    } else if (less(value.first, x->value.first)) {
      return balance(
          make_node(x->color, ins(x->left, value), x->value, x->right));
    } else {
      return make_node(x->color, x->left, value, x->right);
    }
  }

  /*
    balance :: Color -> Tree a -> a -> Tree a -> Tree a
    balance B (T R (T R a x b) y c) z d = T R (T B a x b) y (T B c z d)
    balance B (T R a x (T R b y c)) z d = T R (T B a x b) y (T B c z d)
    balance B a x (T R (T R b y c) z d) = T R (T B a x b) y (T B c z d)
    balance B a x (T R b y (T R c z d)) = T R (T B a x b) y (T B c z d)
    balance color a x b = T color a x b
  */
  static NodePtr balance(const NodePtr& node) {
    if (node->color == BLACK) {
      if (node->left && node->left->left && node->left->color == RED &&
          node->left->left->color == RED) {
        return make_node(
            RED,
            make_node(BLACK, node->left->left->left, node->left->left->value,
                      node->left->left->right),
            node->left->value,
            make_node(BLACK, node->left->right, node->value, node->right));
      } else if (node->left && node->left->right && node->left->color == RED &&
                 node->left->right->color == RED) {
        return make_node(RED,
                         make_node(BLACK, node->left->left, node->left->value,
                                   node->left->right->left),
                         node->left->right->value,
                         make_node(BLACK, node->left->right->right, node->value,
                                   node->right));
      } else if (node->right && node->right->left &&
                 node->right->color == RED && node->right->left->color == RED) {
        return make_node(
            RED,
            make_node(BLACK, node->left, node->value, node->right->left->left),
            node->right->left->value,
            make_node(BLACK, node->right->left->right, node->right->value,
                      node->right->right));
      } else if (node->right && node->right->right &&
                 node->right->color == RED &&
                 node->right->right->color == RED) {
        return make_node(
            RED, make_node(BLACK, node->left, node->value, node->right->left),
            node->right->value,
            make_node(BLACK, node->right->right->left,
                      node->right->right->value, node->right->right->right));
      }
    }
    return node;
  }

  static NodePtr insert(const NodePtr& x, const value_type& value) {
    auto tree = ins(x, value);
    return make_node(BLACK, tree->left, tree->value, tree->right);
  }
  static mapped_type* find(const NodePtr& x, key_type key) {
    Compare less;
    if (x == nil) {
      return nullptr;
    } else if (less(x->value.first, key)) {
      return find(x->right, key);
    } else if (less(key, x->value.first)) {
      return find(x->left, key);
    } else {
      return const_cast<mapped_type*>(&x->value.second);
    }
    return nullptr;
  }
  template <typename key_type, typename mapped_type>
  friend std::ostream& operator<<(std::ostream& str,
                                  const Node<key_type, mapped_type>* node) {
    if (node == nullptr) {
      str << "nil";
    } else {
      str << "(" << (node->color == RED ? "R" : "B") << " " << node->left.get()
          << " " << node->value.first << " " << node->right.get() << ")";
    }
    return str;
  }
};

template <typename Key, typename T, typename Compare>
const typename Node<Key, T, Compare>::NodePtr Node<Key, T, Compare>::nil =
    nullptr;

/**
 * @brief A class template that represents an immutable map.
 *
 * The ImmutableMap class template provides a way to store key-value pairs in an
 * immutable map data structure. It supports operations such as insertion and
 * finding of key-value pairs.
 *
 * @tparam key_type The type of the keys in the map.
 * @tparam mapped_type The type of the values in the map.
 */
template <typename Key, typename T, typename Compare = std::less<Key>>
class ImmutableMap {
public:
  using key_type = Key;
  using mapped_type = T;
  using value_type = std::pair<const key_type, mapped_type>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using key_compare = Compare;
  using reference = value_type&;
  using const_reference = const value_type&;
  using node_type = Node<key_type, mapped_type>;
  struct constant_iterator {
  public:
    constant_iterator(const node_type* node)
        : stack_{std::make_unique<std::vector<const node_type*>>()} {
      stack_->reserve(10);
      for (auto p = node; p != nullptr; p = p->left.get()) {
        stack_->push_back(p);
      }
      if (this->stack_->empty()) {
        this->stack_ = nullptr;
      }
    }
    // for the ::end iterator.
    constant_iterator() : stack_{nullptr} {}

  public:
    const value_type& operator*() const {
      const value_type* ret = nullptr;
      if (stack_ && !stack_->empty()) {
        ret = &this->stack_->back()->value;
      }
      assert(ret != nullptr);
      return *ret;
    }

    bool operator!=(const constant_iterator& other) const {
      return stack_ != other.stack_;
    }

    constant_iterator& operator++() {
      if (this->stack_) {
        auto p = stack_->back();
        stack_->pop_back();
        for (p = p->right.get(); p != nullptr; p = p->left.get()) {
          stack_->push_back(p);
        }
        if (this->stack_->empty()) {
          this->stack_ = nullptr;
        }
      }
      return *this;
    }

  private:
    std::unique_ptr<std::vector<const node_type*>> stack_;
  };
  /**
   * @brief Default constructor.
   *
   * Initializes an empty ImmutableMap.
   */
  ImmutableMap() : root_(node_type::nil), size_{0} {}

  /**
   * Inserts a new key-value pair into the immutable map.
   *
   * @param value The key-value pair to be inserted.
   * @return A new immutable map with the key-value pair inserted.
   */
  ImmutableMap insert(const value_type& value) const {
    return ImmutableMap(node_type::insert(root_, value), size() + 1);
  }

  /**
   * @brief Finds the value associated with the given key in the ImmutableMap.
   *
   * @param key The key to search for.
   * @return mapped_type* A pointer to the value associated with the key, or
   * nullptr if the key is not found.
   */
  mapped_type* find(key_type key) const { return node_type::find(root_, key); }

  constant_iterator begin() const { return root_.get(); }
  constant_iterator end() const { return node_type::nil.get(); }

  size_type size() const { return size_; }
  /**
   * @brief Overloaded stream insertion operator for printing the
   * ImmutableMap.
   *
   * @param str The output stream.
   * @param x The ImmutableMap to print.
   * @return std::ostream& The output stream after printing the
   * ImmutableMap.
   */
  template <typename K2, typename T2>
  friend std::ostream& operator<<(std::ostream& str,
                                  const ImmutableMap<K2, T2>& x) {
    str << x.root_.get();
    return str;
  }

private:
  /**
   * @brief Private constructor used to create a new ImmutableMap with a given
   * root node.
   *
   * @param root The root node of the ImmutableMap.
   */
  ImmutableMap(const typename node_type::NodePtr& root, size_type size)
      : root_(root), size_(size) {}

private:
  const typename node_type::NodePtr root_;
  const size_type size_;
};
} // namespace immutable_map
} // namespace vaip_core
