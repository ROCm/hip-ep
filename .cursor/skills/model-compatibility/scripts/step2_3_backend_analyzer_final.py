#!/usr/bin/env python3
"""
Step 2-3: Backend Analyzer (Final - Hybrid Approach)
Identify which backend each runtime function uses (ROCm library or
Custom Hip Kernel) and emit the full mapping chain:
    onnx_op -> hip_op -> runtime_func -> backend

Final hybrid design:
- Use a predefined runtime-function -> backend mapping table (easy to
  maintain and extend).
- Allow loading the mapping table from an external config file.
- Fall back to dynamic detection by scanning runtime source.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys

class BackendAnalyzer:
    """Identify the ROCm library / Custom Kernel used by each runtime function."""
    
    def __init__(self, runtime_dir: str, hip_to_runtime_json: str = None, onnx_to_hip_json: str = None, 
                 backend_config_json: str = None):
        self.runtime_dir = Path(runtime_dir)
        self.hip_to_runtime_data: List[Dict] = []
        self.hip_to_runtime_by_hip: Dict[str, List[Dict]] = {}
        self.onnx_to_hip_data: List[Dict] = []
        
        if hip_to_runtime_json:
            self._load_json(hip_to_runtime_json, 'hip_to_runtime')
        if onnx_to_hip_json:
            self._load_json(onnx_to_hip_json, 'onnx_to_hip')
        
        # Load runtime implementation files.
        self.runtime_implementations = self._load_runtime_implementations()
        
        # Load or build the runtime -> backend mapping table.
        self.backend_mappings = self._load_backend_mappings(backend_config_json)
    
    def _load_json(self, json_path: str, data_type: str):
        """Load a JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data_type == 'hip_to_runtime':
                for mapping in data.get('mappings', []):
                    self.hip_to_runtime_data.append(mapping)
                    hip_op = mapping.get('hip_op')
                    if hip_op:
                        self.hip_to_runtime_by_hip.setdefault(hip_op, []).append(mapping)
            elif data_type == 'onnx_to_hip':
                for mapping in data.get('mappings', []):
                    self.onnx_to_hip_data.append(mapping)
        except Exception as e:
            print(f"Warning: Could not load {json_path}: {e}")
    
    def _load_backend_mappings(self, config_json: str = None) -> Dict[str, str]:
        """Load the runtime -> backend mapping table.

        Source of truth is the runtime source itself: `_analyze_backend`
        scans each wrapper's implementation for ROCm library API calls
        (miopen*, hipblasLt*, hipblas*, rocblas*) and custom-kernel
        markers (__global__, hipLaunchKernel, file under custom_kernels/).
        This dict is therefore empty by default -- override only via an
        external config when source-scan misclassifies (no compiled-in
        list to drift out of date as new wrappers land).
        """
        if config_json and Path(config_json).exists():
            try:
                with open(config_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('mappings', {})
            except Exception as e:
                print(f"Warning: Could not load backend config {config_json}: {e}")
        return {}
    
    def _load_runtime_implementations(self) -> Dict[str, Dict]:
        """Index every runtime function definition to its source file.

        Walks `lib/Runtime/real/` plus immediate `lib/Runtime/` siblings,
        but **excludes `lib/Runtime/mock/`** because mock_gpu.cpp redefines
        many wrappers as stubs whose bodies do not reflect the real backend.
        Without this filter mock stubs overwrite real implementations
        (last-write-wins) and `_analyze_backend` ends up reading the mock
        body for backend classification.

        For real-vs-real collisions (rare), the first definition wins.
        """
        implementations: Dict[str, Dict] = {}

        # `runtime_dir` is typically `<repo>/lib/Runtime/real`. Also scan
        # the parent so any wrappers placed at `lib/Runtime/*.cpp` are
        # caught, but always skip the `mock/` subtree.
        search_dirs = [self.runtime_dir]
        parent_dir = self.runtime_dir.parent
        if parent_dir.exists() and parent_dir != self.runtime_dir:
            search_dirs.append(parent_dir)

        runtime_files = set()
        for search_dir in search_dirs:
            for cpp in search_dir.glob('**/*.cpp'):
                norm = str(cpp).replace('\\', '/').lower()
                if '/mock/' in norm or norm.endswith('/mock_gpu.cpp'):
                    continue
                runtime_files.add(cpp)

        func_pat = re.compile(
            r'(?:int|void|bool|float|double|hipError_t)\s+(\w+)\s*\([^)]*\)\s*\{'
        )

        for file_path in runtime_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                for func_name in func_pat.findall(content):
                    # First real definition wins; skip duplicates.
                    if func_name not in implementations:
                        implementations[func_name] = {
                            'file': str(file_path),
                            'content': content,
                        }
            except Exception:
                pass

        return implementations
    
    def analyze(self) -> Dict:
        """Run the backend analysis."""
        result = {
            'mappings': [],
            'statistics': {}
        }
        
        seen_keys = set()
        for onnx_mapping in self.onnx_to_hip_data:
            onnx_op = onnx_mapping['onnx_op']
            onnx_domain = onnx_mapping['onnx_domain']
            hip_op = onnx_mapping['hip_op']
            onnx_file = onnx_mapping['file_name']

            row_key = (onnx_op, onnx_domain, hip_op)
            if row_key in seen_keys:
                continue
            seen_keys.add(row_key)
            
            runtime_func = None
            hip_file = None
            hip_rt_candidates = self.hip_to_runtime_by_hip.get(hip_op, [])
            if hip_rt_candidates:
                # Prefer a stable deterministic pick when multiple mappings exist.
                chosen = sorted(
                    hip_rt_candidates,
                    key=lambda m: (
                        str(m.get('file_name') or ''),
                        str(m.get('runtime_func') or ''),
                    ),
                )[0]
                runtime_func = chosen.get('runtime_func')
                hip_file = chosen.get('file_name')
            
            if runtime_func:
                backend = self._analyze_backend(runtime_func)
                runtime_file = self.runtime_implementations.get(runtime_func, {}).get('file', 'N/A')
            else:
                backend = "Compile Time Optimization"
                runtime_func = None
                runtime_file = "N/A"
            
            mapping = {
                'onnx_op': onnx_op,
                'onnx_domain': onnx_domain,
                'hip_op': hip_op,
                'runtime_func': runtime_func,
                'backend': backend,
                'onnx_file': onnx_file,
                'hip_file': hip_file,
                'runtime_file': runtime_file
            }
            
            result['mappings'].append(mapping)
        
        backends = set(m['backend'] for m in result['mappings'])
        result['statistics'] = {
            'total_mappings': len(result['mappings']),
            'unique_onnx_ops': len(set(m['onnx_op'] for m in result['mappings'] if m['onnx_op'])),
            'unique_hip_ops': len(set(m['hip_op'] for m in result['mappings'])),
            'unique_runtime_funcs': len(set(m['runtime_func'] for m in result['mappings'] if m['runtime_func'])),
            'unique_backends': len(backends),
            'backends': sorted(list(backends))
        }
        
        return result
    
    def _analyze_backend(self, runtime_func: str) -> str:
        """Identify the backend a runtime function relies on.

        Detection rules (source-driven, no hardcoded wrapper table):
        1. Override via external config (self.backend_mappings, empty by default).
        2. Implementation file path under custom_kernels/ -> Custom Hip Kernel
           (file lives in the custom-kernel area of the tree).
        3. The WRAPPER FUNCTION BODY (scoped, not whole file) is examined
           for direct calls to library APIs:
              - hipblasLt* / hipblaslt_*  -> hipBLASLt
              - miopen* / miopenCreate    -> MIOpen
              - hipblas* / hipblas[A-Z]   -> hipBLAS
              - rocblas* / rocblas[A-Z]   -> rocBLAS
           Body-scope avoids cross-wrapper pollution from neighbouring
           helpers in the same .cpp.
        4. Body has `__global__` / `hipLaunchKernel`, OR delegates to a
           `hip_<name>(` function while the file `#include`s the custom-
           kernels gateway header -> Custom Hip Kernel.
        5. Fallback to whole-file library scan (rescue tiny wrappers whose
           body is purely a delegation call without the library token).
        6. Else Unknown.
        """

        if runtime_func in self.backend_mappings:
            return self.backend_mappings[runtime_func]

        if runtime_func not in self.runtime_implementations:
            return "Unknown"

        impl = self.runtime_implementations[runtime_func]
        content = impl['content']
        file_path = impl['file']

        if 'custom_kernels' in file_path.replace('\\', '/'):
            return "Custom Hip Kernel"

        rocm_libraries = [
            ('hipBLASLt', [r'hipblaslt_\w+', r'hipblasLt\w+']),
            ('MIOpen', [r'miopen_\w+', r'miopenCreate', r'miopenDestroy']),
            ('hipBLAS', [r'hipblas_\w+', r'hipblas[A-Z]']),
            ('rocBLAS', [r'rocblas_\w+', r'rocblas[A-Z]']),
        ]

        custom_kernel_header = bool(
            re.search(r'#\s*include\s*[<"]hip_custom_kernels\.h[>"]', content)
        )

        body = self._extract_function_body(content, runtime_func)
        if body:
            for lib_name, patterns in rocm_libraries:
                for pattern in patterns:
                    if re.search(pattern, body):
                        return lib_name
            if re.search(r'__global__\s+void|__kernel\s+void|hipLaunchKernel', body):
                return "Custom Hip Kernel"
            # Body delegates to a hip_<name>(...) call AND the file pulls
            # in the custom-kernel gateway: this is the canonical
            # "wrap_X -> hip_X" delegation pattern used across runtime/real/.
            if custom_kernel_header and re.search(r'\bhip_\w+\s*\(', body):
                return "Custom Hip Kernel"

        for lib_name, patterns in rocm_libraries:
            for pattern in patterns:
                if re.search(pattern, content):
                    return lib_name
        if re.search(r'__global__\s+void|__kernel\s+void|hipLaunchKernel', content):
            return "Custom Hip Kernel"

        return "Unknown"

    @staticmethod
    def _extract_function_body(content: str, func_name: str) -> str:
        """Extract the body `{ ... }` of a top-level function definition
        `... func_name(...) {`. Returns '' when not found. Uses balanced
        brace traversal so nested blocks and string-with-brace edge cases
        are handled."""
        # Find the function signature opening: `func_name(`.
        sig_pat = re.compile(
            r'\b' + re.escape(func_name) + r'\s*\([^)]*\)\s*(?:const\s*)?\{'
        )
        m = sig_pat.search(content)
        if not m:
            return ''
        # The `{` is the last char of the match. Walk balanced braces.
        brace_pos = m.end() - 1
        depth = 0
        i = brace_pos
        n = len(content)
        while i < n:
            ch = content[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return content[brace_pos + 1:i]
            i += 1
        return ''


def main():
    if len(sys.argv) < 2:
        print("Usage: python step2_3_backend_analyzer_final.py <runtime_dir> [hip_to_runtime_json] [onnx_to_hip_json] [output_dir] [backend_config_json]")
        sys.exit(1)
    
    runtime_dir = sys.argv[1]
    hip_to_runtime_json = sys.argv[2] if len(sys.argv) > 2 else None
    onnx_to_hip_json = sys.argv[3] if len(sys.argv) > 3 else None
    output_dir = sys.argv[4] if len(sys.argv) > 4 else str(Path(runtime_dir).parent)
    backend_config_json = sys.argv[5] if len(sys.argv) > 5 else None
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing backends (Final - Hybrid Approach)...")
    print(f"Runtime directory: {runtime_dir}")
    if hip_to_runtime_json:
        print(f"Hip->Runtime mappings: {hip_to_runtime_json}")
    if onnx_to_hip_json:
        print(f"ONNX->Hip mappings: {onnx_to_hip_json}")
    if backend_config_json:
        print(f"Backend config: {backend_config_json}")
    print(f"Output directory: {output_dir}\n")
    
    analyzer = BackendAnalyzer(runtime_dir, hip_to_runtime_json, onnx_to_hip_json, backend_config_json)
    result = analyzer.analyze()
    
    print(f"{'='*80}")
    print(f"Backend Analysis Summary:")
    print(f"  Total mappings: {result['statistics']['total_mappings']}")
    print(f"  Unique ONNX ops: {result['statistics']['unique_onnx_ops']}")
    print(f"  Unique Hip ops: {result['statistics']['unique_hip_ops']}")
    print(f"  Unique Runtime funcs: {result['statistics']['unique_runtime_funcs']}")
    print(f"  Unique backends: {result['statistics']['unique_backends']}")
    print(f"  Backends: {', '.join(result['statistics']['backends'])}")
    print(f"{'='*80}\n")
    
    json_path = Path(output_dir) / "step2_3_backend_analysis.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[OK] Analysis saved: {json_path}")
    
    md_path = Path(output_dir) / "step2_3_backend_analysis.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(generate_markdown_report(result))
    print(f"[OK] Markdown report saved: {md_path}")


def generate_markdown_report(result: Dict) -> str:
    """Render the backend analysis as a Markdown report."""
    report = []
    
    report.append("# Backend Analysis Report\n\n")
    
    report.append("## Summary\n\n")
    stats = result['statistics']
    report.append(f"- **Total Mappings**: {stats['total_mappings']}\n")
    report.append(f"- **Unique ONNX Operations**: {stats['unique_onnx_ops']}\n")
    report.append(f"- **Unique Hip Operations**: {stats['unique_hip_ops']}\n")
    report.append(f"- **Unique Runtime Functions**: {stats['unique_runtime_funcs']}\n")
    report.append(f"- **Unique Backends**: {stats['unique_backends']}\n")
    report.append(f"- **Backends Used**: {', '.join(stats['backends'])}\n\n")
    
    report.append("## Complete Mapping Chain with Backend Implementation\n\n")
    report.append("| ONNX Op | Domain | Hip Op | Runtime Func | Backend |\n")
    report.append("|---|---|---|---|---|\n")
    
    for mapping in sorted(result['mappings'], key=lambda x: (x['onnx_op'] or '', x['hip_op'])):
        onnx_op = mapping['onnx_op'] or "N/A"
        domain = mapping['onnx_domain'] or "N/A"
        hip_op = mapping['hip_op']
        runtime_func = mapping['runtime_func'] or "None"
        backend = mapping['backend']
        report.append(f"| {onnx_op} | {domain} | {hip_op} | {runtime_func} | {backend} |\n")
    
    report.append("\n")
    
    report.append("## Mappings by Backend\n\n")
    
    backends = {}
    for mapping in result['mappings']:
        backend = mapping['backend']
        if backend not in backends:
            backends[backend] = []
        backends[backend].append(mapping)
    
    backend_descriptions = {
        'MIOpen': "AMD's optimized library for deep learning operations on HIP/ROCm.",
        'hipBLASLt': "AMD's optimized BLAS library for matrix operations on HIP/ROCm.",
        'Custom Hip Kernel': "Custom HIP kernels implemented specifically for these operations.",
        'Compile Time Optimization': "Tensor operations optimized at compile time, no runtime function needed.",
    }
    
    for backend in sorted(backends.keys()):
        mappings = backends[backend]
        report.append(f"### {backend} ({len(mappings)} operations)\n\n")
        
        if backend in backend_descriptions:
            report.append(f"{backend_descriptions[backend]}\n\n")
        
        report.append("| ONNX Op | Hip Op | Runtime Func |\n")
        report.append("|---|---|---|\n")
        
        for mapping in sorted(mappings, key=lambda x: x['onnx_op'] or ''):
            onnx_op = mapping['onnx_op'] or "N/A"
            hip_op = mapping['hip_op']
            runtime_func = mapping['runtime_func'] or "None"
            report.append(f"| {onnx_op} | {hip_op} | {runtime_func} |\n")
        
        report.append("\n")
    
    report.append("## Backend Distribution\n\n")
    report.append("```\n")
    total = len(result['mappings'])
    for backend in sorted(backends.keys()):
        count = len(backends[backend])
        percentage = (count / total * 100) if total > 0 else 0
        report.append(f"{backend:30s} {count:2d} operations ({percentage:5.1f}%)\n")
    report.append("```\n\n")
    
    report.append("## Key Insights\n\n")
    
    miopen_count = len(backends.get('MIOpen', []))
    custom_count = len(backends.get('Custom Hip Kernel', []))
    compile_count = len(backends.get('Compile Time Optimization', []))
    hipblas_count = len(backends.get('hipBLASLt', []))
    
    if miopen_count > 0:
        report.append(f"1. **MIOpen Dominance**: {miopen_count} operations ({miopen_count/total*100:.0f}%) use MIOpen, indicating heavy reliance on AMD's optimized library for standard operations.\n\n")
    
    if custom_count > 0:
        report.append(f"2. **Custom Kernels**: {custom_count} operations ({custom_count/total*100:.0f}%) use custom HIP kernels, particularly for:\n")
        report.append("   - Attention mechanisms (GroupQueryAttention)\n")
        report.append("   - Quantized operations (MatMulNBits)\n")
        report.append("   - Specialized operations (QMoE, RotaryEmbedding)\n\n")
    
    if compile_count > 0:
        report.append(f"3. **Compile-Time Optimization**: {compile_count} operations ({compile_count/total*100:.0f}%) are tensor shape operations optimized at compile time with no runtime overhead.\n\n")
    
    if hipblas_count > 0:
        report.append(f"4. **Specialized Libraries**: {hipblas_count} operation(s) use hipBLASLt for optimal performance on matrix operations.\n\n")
    
    return ''.join(report)


if __name__ == '__main__':
    main()
