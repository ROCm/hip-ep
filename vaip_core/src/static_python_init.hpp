/*
 *     The Xilinx Vitis AI Vaip in this distribution are provided under the
 * following free and permissive binary-only license, but are not provided in
 * source code form.  While the following free and permissive license is similar
 * to the BSD open source license, it is NOT the BSD open source license nor
 * other OSI-approved open source license.
 *
 *      Copyright (C) 2022 Xilinx, Inc. All rights reserved.
 *      Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights
 * reserved.
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

#include "vaip/util.hpp"
#include <Python.h>
#include <cwchar>
#include <memory>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <vector>

inline void initialize_interpreter(int argc, const char* const* argv) {
  if (Py_IsInitialized() != 0) {
    throw std::runtime_error("The interpreter is already running");
  }

  PyConfig config;
  PyConfig_InitIsolatedConfig(&config);

  PyStatus status =
      PyConfig_SetBytesArgv(&config, argc, const_cast<char* const*>(argv));
  if (PyStatus_Exception(status)) {
    // A failure here indicates a character-encoding failure or the python
    // interpreter out of memory. Give up.
    PyConfig_Clear(&config);
    throw std::runtime_error(PyStatus_IsError(status)
                                 ? status.err_msg
                                 : "Failed to prepare CPython");
  }
  config.module_search_paths_set = 1;
  config.site_import = 0;
  status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status)) {
    throw std::runtime_error(
        PyStatus_IsError(status) ? status.err_msg : "Failed to init CPython");
  }
}

template <typename T, pybind11::detail::enable_if_t<
                          std::is_base_of<pybind11::object, T>::value, int> = 0>
bool isinstance(pybind11::handle obj) {
  return T::check_(obj);
}

inline void finalize_interpreter() {
  pybind11::handle builtins(PyEval_GetBuiltins());
  const char* id = PYBIND11_INTERNALS_ID;

  // Get the internals pointer (without creating it if it doesn't exist).  It's
  // possible for the internals to be created during Py_Finalize() (e.g. if a
  // py::capsule calls `get_internals()` during destruction), so we get the
  // pointer-pointer here and check it after Py_Finalize().
  pybind11::detail::internals** internals_ptr_ptr =
      pybind11::detail::get_internals_pp();
  // It could also be stashed in builtins, so look there too:
  if (builtins.contains(id) && isinstance<pybind11::capsule>(builtins[id])) {
    internals_ptr_ptr = pybind11::capsule(builtins[id]);
  }
  // Local internals contains data managed by the current interpreter, so we
  // must clear them to avoid undefined behaviors when initializing another
  // interpreter
  pybind11::detail::get_local_internals().registered_types_cpp.clear();
  pybind11::detail::get_local_internals()
      .registered_exception_translators.clear();

  Py_Finalize();

  if (internals_ptr_ptr) {
    delete *internals_ptr_ptr;
    *internals_ptr_ptr = nullptr;
  }
}

class static_scoped_interpreter {

public:
  explicit static_scoped_interpreter(int argc = 0,
                                     const char* const* argv = nullptr) {
    initialize_interpreter(argc, argv);
  }

  static_scoped_interpreter(const static_scoped_interpreter&) = delete;
  static_scoped_interpreter(static_scoped_interpreter&& other) noexcept {
    other.is_valid = false;
  }
  static_scoped_interpreter&
  operator=(const static_scoped_interpreter&) = delete;
  static_scoped_interpreter& operator=(static_scoped_interpreter&&) = delete;

  ~static_scoped_interpreter() {
    if (is_valid) {
      finalize_interpreter();
    }
  }

private:
  bool is_valid = true;
};