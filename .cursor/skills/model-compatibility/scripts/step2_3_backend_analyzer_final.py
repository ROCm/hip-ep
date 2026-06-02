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

        Priority:
        1. External config file (if provided).
        2. Built-in predefined mapping table.
        """
        
        if config_json and Path(config_json).exists():
            try:
                with open(config_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('mappings', {})
            except Exception as e:
                print(f"Warning: Could not load backend config {config_json}: {e}")
        
        # Built-in predefined mapping table.
        return {
            # MIOpen functions
            'wrap_miopenActivationForward': 'MIOpen',
            'wrap_miopenConvolutionForward': 'MIOpen',
            'wrap_miopenOpTensor': 'MIOpen',
            'wrap_miopenT5LayerNormForward': 'MIOpen',
            'wrap_reduce_sum': 'MIOpen',
            'hip_miopen_softmax': 'MIOpen',
            
            # hipBLASLt functions
            'wrap_hipblasLtMatmul': 'hipBLASLt',
            
            # Custom Hip Kernel functions
            'wrap_cast': 'Custom Hip Kernel',
            'wrap_gather': 'Custom Hip Kernel',
            'wrap_gemm': 'Custom Hip Kernel',
            'wrap_group_query_attention': 'Custom Hip Kernel',
            'wrap_matmul_nbits': 'Custom Hip Kernel',
            'wrap_qmoe': 'Custom Hip Kernel',
            'wrap_rotary_embedding': 'Custom Hip Kernel',
            'wrap_skip_simplified_layer_norm': 'Custom Hip Kernel',
            'wrap_elementwise_sub': 'MIOpen',
            'hip_transpose': 'Custom Hip Kernel',
        }
    
    def _load_runtime_implementations(self) -> Dict[str, Dict]:
        """Load the content of all runtime implementation files."""
        implementations = {}
        
        # Search both the runtime dir and its parent (broader search).
        search_dirs = [self.runtime_dir]
        parent_dir = self.runtime_dir.parent
        if parent_dir.exists():
            search_dirs.append(parent_dir)
        
        runtime_files = []
        for search_dir in search_dirs:
            runtime_files.extend(list(search_dir.glob('**/*.cpp')))
        
        runtime_files = list(set(runtime_files))
        
        for file_path in runtime_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                func_patterns = [
                    r'(?:int|void|bool|float|double|hipError_t)\s+(\w+)\s*\([^)]*\)\s*\{',
                ]
                
                for pattern in func_patterns:
                    matches = re.findall(pattern, content)
                    for func_name in matches:
                        if func_name not in implementations:
                            implementations[func_name] = {
                                'file': str(file_path),
                                'content': content
                            }
            except Exception as e:
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

        Priority:
        1. Look up in the predefined mapping table.
        2. Inspect implementation file path (e.g. 3rd-party/custom_kernels).
        3. Dynamic detection by scanning the implementation source.
        """
        
        # 1. Check the predefined mapping table.
        if runtime_func in self.backend_mappings:
            return self.backend_mappings[runtime_func]
        
        # 2. Dynamic detection (requires the implementation source).
        if runtime_func not in self.runtime_implementations:
            return "Unknown"
        
        impl = self.runtime_implementations[runtime_func]
        content = impl['content']
        file_path = impl['file']
        
        # 2.1 Path heuristic: files under 3rd-party/custom_kernels are Custom Hip Kernels.
        if '3rd-party/custom_kernels' in file_path or '3rd-party\\custom_kernels' in file_path:
            return "Custom Hip Kernel"
        if 'custom_kernels' in file_path:
            return "Custom Hip Kernel"
        
        # 2.2 Library-API detection in source.
        rocm_libraries = [
            ('MIOpen', [r'miopen_\w+', r'miopenCreate', r'miopenDestroy']),
            ('hipBLASLt', [r'hipblaslt_\w+', r'hipblasLt\w+']),
            ('hipBLAS', [r'hipblas_\w+', r'hipblas[A-Z]']),
            ('rocBLAS', [r'rocblas_\w+', r'rocblas[A-Z]']),
        ]
        
        for lib_name, patterns in rocm_libraries:
            for pattern in patterns:
                if re.search(pattern, content):
                    return lib_name
        
        # 2.3 Custom kernel detection.
        if re.search(r'__global__\s+void|__kernel\s+void|hipLaunchKernel', content):
            return "Custom Hip Kernel"
        
        return "Unknown"


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
