##
# ** Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ** Licensed under the MIT License.
##

# ============================================================================
# hip_add_pdll_patterns
# ============================================================================
# Compiles a PDLL pattern set and embeds it into the calling library as a byte
# array, so a pass can reach its patterns through two `extern "C"` accessors
# and nothing has to be shipped beside the library.
#
# The pipeline is:
#   mlir-pdll -x mlir           ->  one PDL module (textual)
#   mlir-opt --emit-bytecode    ->  condensed bytecode (optional)
#   cmake/xxd.py                ->  a .cpp holding the bytes + accessors
#
# Signature:
#   hip_add_pdll_patterns(
#     NAME       <target>      # custom target created for the generated .cpp
#     ENTRY      <file.pdll>   # aggregate entry point, relative to SOURCE_DIR
#     SOURCE_DIR <dir>         # searched by -I and globbed for dependencies
#     SYMBOL     <prefix>      # accessors: <prefix>_data(), <prefix>_size()
#     OUT_VAR    <var>)        # receives the generated .cpp path
#
# Every caller gets its own target, byte array and accessor names, so several
# independent pattern sets can coexist in one build.
#
# When mlir-pdll is unavailable the generated .cpp still defines both
# accessors and reports size 0. Whether that is fatal is up to the consuming
# pass: convert-onnx-to-hip rejects it because QDQ fusion is load-bearing for
# it, while hip-fusion-transform degrades to a no-op.
include_guard(GLOBAL)

function(hip_add_pdll_patterns)
    cmake_parse_arguments(ARG "" "NAME;ENTRY;SOURCE_DIR;SYMBOL;OUT_VAR" "" ${ARGN})
    foreach(required NAME ENTRY SOURCE_DIR SYMBOL OUT_VAR)
        if(NOT ARG_${required})
            message(FATAL_ERROR "hip_add_pdll_patterns: ${required} is required")
        endif()
    endforeach()

    set(_label "${ARG_NAME}")
    set(_blob_var "${ARG_SYMBOL}_blob")
    set(_embed_cpp "${CMAKE_CURRENT_BINARY_DIR}/${ARG_NAME}_data.cpp")
    set(${ARG_OUT_VAR} "${_embed_cpp}" PARENT_SCOPE)

    # Locate the MLIR tools. When LLVM/MLIR is built from source as a
    # subproject the tool targets exist but their .exe files are not present at
    # configure time, so find_program cannot see them — use the target directly
    # (the custom commands below then gain a build-time dependency on it). Fall
    # back to find_program for the prebuilt / installed-LLVM case:
    # LLVM_TOOLS_BINARY_DIR is exported by find_package(MLIR) and covers both
    # the prebuilt SDK and an installed LLVM; the _deps path covers a
    # FetchContent tree that has already been built. Default search paths stay
    # enabled so a tool on PATH is usable.
    set(_tool_hints
        "${CMAKE_BINARY_DIR}/_deps/llvm-project-build/bin"
        "${LLVM_TOOLS_BINARY_DIR}"
        "${LLVM_BINARY_DIR}/bin"
        "${MLIR_BINARY_DIR}/bin"
    )

    if(TARGET mlir-pdll)
        set(_pdll_exe mlir-pdll)
        message(STATUS "${_label}: using in-tree mlir-pdll target")
    else()
        find_program(MLIR_PDLL_EXE NAMES mlir-pdll HINTS ${_tool_hints})
        set(_pdll_exe "${MLIR_PDLL_EXE}")
    endif()

    if(NOT _pdll_exe)
        # No mlir-pdll -> empty pattern blob. Unrelated targets still build.
        file(WRITE "${_embed_cpp}"
            "#include <cstddef>\n"
            "static const unsigned char ${_blob_var}[] = {0};\n"
            "extern \"C\" const unsigned char *${ARG_SYMBOL}_data(void) {\n"
            "  return ${_blob_var};\n"
            "}\n"
            "extern \"C\" size_t ${ARG_SYMBOL}_size(void) { return 0; }\n"
        )
        message(WARNING
            "${_label}: mlir-pdll not found, so ${ARG_ENTRY} cannot be "
            "compiled and the pattern set is embedded empty.\n"
            "\n"
            "To fix:\n"
            "  - From-source LLVM: ensure mlir is in LLVM_ENABLE_PROJECTS "
            "(cmake/deps.cmake already does this) and reconfigure after LLVM "
            "is in the build tree.\n"
            "  - Prebuilt LLVM: install the MLIR tools (mlir-pdll) into the "
            "same prefix as find_package(MLIR), or put mlir-pdll on PATH.\n"
            "\n"
            "Searched in:\n"
            "  - ${CMAKE_BINARY_DIR}/_deps/llvm-project-build/bin\n"
            "  - ${LLVM_TOOLS_BINARY_DIR}\n"
            "  - ${LLVM_BINARY_DIR}/bin\n"
            "  - ${MLIR_BINARY_DIR}/bin\n"
            "  - System PATH")
        return()
    endif()

    if(NOT TARGET mlir-pdll)
        message(STATUS "${_label}: mlir-pdll found: ${_pdll_exe}")
    endif()

    # mlir-opt condenses the patterns before they are embedded. It is optional:
    # without it the textual module is embedded as-is, which the parser accepts
    # because it dispatches on the bytecode magic bytes.
    if(TARGET mlir-opt)
        set(_opt_exe mlir-opt)
    else()
        find_program(MLIR_OPT_EXE NAMES mlir-opt HINTS ${_tool_hints})
        set(_opt_exe "${MLIR_OPT_EXE}")
    endif()

    # Embedding goes through cmake/xxd.py, which needs an interpreter. Sibling
    # directories run find_package(Python3) separately because it only
    # populates the directory scope it is called from.
    find_package(Python3 COMPONENTS Interpreter)
    if(NOT Python3_FOUND)
        find_program(Python3_EXECUTABLE python)
    endif()
    if(NOT Python3_EXECUTABLE)
        message(FATAL_ERROR
            "${_label}: Python not found, so the PDL patterns cannot be "
            "embedded into the EP library.")
    endif()

    # The entry point #includes every pattern file, so mlir-pdll runs once and
    # emits one PDL module — collecting the patterns into a single
    # RewritePatternSet also makes them share a fixpoint. Every .pdll is still
    # globbed as a dependency, otherwise editing an included pattern would not
    # rebuild it.
    file(GLOB _pdll_sources CONFIGURE_DEPENDS "${ARG_SOURCE_DIR}/*.pdll")
    get_filename_component(_entry_stem "${ARG_ENTRY}" NAME_WE)
    set(_pdl_text "${CMAKE_CURRENT_BINARY_DIR}/${_entry_stem}.pdl.mlir")

    if(TARGET mlir-pdll)
        set(_pdll_command "$<TARGET_FILE:mlir-pdll>")
        set(_pdll_depends mlir-pdll)
    else()
        set(_pdll_command "${_pdll_exe}")
        set(_pdll_depends "${_pdll_exe}")
    endif()
    add_custom_command(OUTPUT "${_pdl_text}"
        COMMAND "${_pdll_command}" -x mlir "-I${ARG_SOURCE_DIR}"
                "${ARG_SOURCE_DIR}/${ARG_ENTRY}" -o "${_pdl_text}"
        DEPENDS ${_pdll_sources} ${_pdll_depends}
        COMMENT "Compiling PDLL patterns -> ${_entry_stem}.pdl.mlir"
        VERBATIM)

    if(_opt_exe)
        if(TARGET mlir-opt)
            set(_opt_command "$<TARGET_FILE:mlir-opt>")
            set(_opt_depends mlir-opt)
        else()
            set(_opt_command "${_opt_exe}")
            set(_opt_depends "${_opt_exe}")
            message(STATUS "${_label}: mlir-opt found: ${_opt_exe}")
        endif()
        # mlir-pdll hardcodes enableDebugInfo(), so two thirds of its output is
        # location aliases repeating an absolute source path. Stripping them
        # also keeps the build directory out of the shipped library. Bytecode
        # then folds the remaining names into a string table.
        set(_embed_source "${CMAKE_CURRENT_BINARY_DIR}/${_entry_stem}.pdl.mlirbc")
        add_custom_command(OUTPUT "${_embed_source}"
            COMMAND "${_opt_command}" --strip-debuginfo --emit-bytecode
                    "${_pdl_text}" -o "${_embed_source}"
            DEPENDS "${_pdl_text}" ${_opt_depends}
            COMMENT "Condensing PDL patterns -> ${_entry_stem}.pdl.mlirbc"
            VERBATIM)
    else()
        message(STATUS
            "${_label}: mlir-opt not found; embedding the textual PDL module.")
        set(_embed_source "${_pdl_text}")
    endif()

    # xxd.py terminates the array with an extra 0x00, so the accessor reports
    # the real file size instead of sizeof().
    add_custom_command(OUTPUT "${_embed_cpp}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/cmake/xxd.py"
                --var ${_blob_var}
                --output "${_embed_cpp}.tmp"
                "${_embed_source}"
        COMMAND "${Python3_EXECUTABLE}" -c
                "import os,sys; tmp,out,blob = sys.argv[1:4]; n = os.path.getsize(blob); open(out,'w').write('#include <cstddef>\\n' + open(tmp).read() + f'\\nextern \"C\" const unsigned char *${ARG_SYMBOL}_data(void) {{ return ${_blob_var}; }}\\nextern \"C\" size_t ${ARG_SYMBOL}_size(void) {{ return {n}; }}\\n')"
                "${_embed_cpp}.tmp" "${_embed_cpp}" "${_embed_source}"
        COMMAND "${CMAKE_COMMAND}" -E remove "${_embed_cpp}.tmp"
        DEPENDS "${_embed_source}"
        COMMENT "Embedding PDL patterns via ${ARG_NAME}_data.cpp"
        VERBATIM)
    add_custom_target(${ARG_NAME} DEPENDS "${_embed_cpp}")
    message(STATUS "${_label}: PDLL enabled — embedding ${_embed_source}")
endfunction()
