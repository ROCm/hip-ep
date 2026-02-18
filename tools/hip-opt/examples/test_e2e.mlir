// End-to-end test: self-contained transformer layer exercising all HIP dialect ops.
// All buffers allocated internally via hip.alloc (zero-filled by hipMalloc).
// No function arguments -- avoids memref calling convention complexity.
//
// Dimensions: seq=4, hidden=128, ffn=344, heads=..., layers=2, vocab=1000
//
// Run:  hip-opt test_e2e.mlir --convert-hip-to-llvm --convert-scf-to-cf \
//         --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts

module {
  func.func @run() {
    // ================================================================
    // 1. INITIALIZE
    // ================================================================
    %handle = hip.create_handle() : !hip.handle

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // ================================================================
    // 2. ALLOCATE ALL BUFFERS (static shapes, zero-filled by hipMalloc)
    // ================================================================

    // Residual stream & norm
    %x         = hip.alloc(%handle) : memref<4x128xf32, 1>
    %x_norm    = hip.alloc(%handle) : memref<4x128xf32, 1>
    %residual  = hip.alloc(%handle) : memref<4x128xf32, 1>

    // Attention intermediates
    %q         = hip.alloc(%handle) : memref<4x128xf32, 1>
    %k         = hip.alloc(%handle) : memref<4x128xf32, 1>
    %v         = hip.alloc(%handle) : memref<4x128xf32, 1>
    %attn_out  = hip.alloc(%handle) : memref<4x128xf32, 1>
    %proj      = hip.alloc(%handle) : memref<4x128xf32, 1>

    // MLP intermediates
    %gate_buf  = hip.alloc(%handle) : memref<4x344xf32, 1>
    %up_buf    = hip.alloc(%handle) : memref<4x344xf32, 1>
    %silu_buf  = hip.alloc(%handle) : memref<4x344xf32, 1>
    %mlp_buf   = hip.alloc(%handle) : memref<4x344xf32, 1>
    %down_buf  = hip.alloc(%handle) : memref<4x128xf32, 1>

    // Weights
    %w_rms_attn  = hip.alloc(%handle) : memref<128xf32, 1>
    %w_rms_mlp   = hip.alloc(%handle) : memref<128xf32, 1>
    %w_rms_final = hip.alloc(%handle) : memref<128xf32, 1>
    %w_q         = hip.alloc(%handle) : memref<128x128xf32, 1>
    %w_k         = hip.alloc(%handle) : memref<128x128xf32, 1>
    %w_v         = hip.alloc(%handle) : memref<128x128xf32, 1>
    %w_o         = hip.alloc(%handle) : memref<128x128xf32, 1>
    %w_gate      = hip.alloc(%handle) : memref<128x344xf32, 1>
    %w_up        = hip.alloc(%handle) : memref<128x344xf32, 1>
    %w_down      = hip.alloc(%handle) : memref<344x128xf32, 1>
    %w_head      = hip.alloc(%handle) : memref<128x1000xf32, 1>

    // Embedding & output
    %input_ids = hip.alloc(%handle) : memref<4xi32, 1>
    %emb_table = hip.alloc(%handle) : memref<1000x128xf32, 1>
    %logits    = hip.alloc(%handle) : memref<4x1000xf32, 1>

    // KV cache & RoPE
    %kv_cache  = hip.alloc(%handle) : memref<2x64x128xf32, 1>
    %cos_cache = hip.alloc(%handle) : memref<64x64xf32, 1>
    %sin_cache = hip.alloc(%handle) : memref<64x64xf32, 1>

    // ================================================================
    // 3. EMBEDDING LOOKUP
    // ================================================================
    hip.gather(%handle, %input_ids, %emb_table, %x)
      : (memref<4xi32, 1>, memref<1000x128xf32, 1>, memref<4x128xf32, 1>) -> ()

    // ================================================================
    // 4. TRANSFORMER LAYER LOOP (2 iterations)
    // ================================================================
    scf.for %layer = %c0 to %c2 step %c1 {

      // --- A. Pre-Attention RMS Norm ---
      hip.miopen.graph {
        hip.miopen.rms_norm(%handle, %x, %w_rms_attn, %x_norm)
          : (memref<4x128xf32, 1>, memref<128xf32, 1>, memref<4x128xf32, 1>) -> ()
      }

      // --- B. Q/K/V Projections ---
      hip.hipblaslt.graph {
        hip.hipblaslt.matmul(%handle, %x_norm, %w_q, %q)
          : (memref<4x128xf32, 1>, memref<128x128xf32, 1>, memref<4x128xf32, 1>) -> ()
        hip.hipblaslt.matmul(%handle, %x_norm, %w_k, %k)
          : (memref<4x128xf32, 1>, memref<128x128xf32, 1>, memref<4x128xf32, 1>) -> ()
        hip.hipblaslt.matmul(%handle, %x_norm, %w_v, %v)
          : (memref<4x128xf32, 1>, memref<128x128xf32, 1>, memref<4x128xf32, 1>) -> ()
      }

      // --- C. Rotary Positional Embeddings ---
      hip.miopen.graph {
        hip.miopen.rope(%handle, %q, %k, %cos_cache, %sin_cache, %c0)
          : (memref<4x128xf32, 1>, memref<4x128xf32, 1>,
             memref<64x64xf32, 1>, memref<64x64xf32, 1>, index) -> ()
      }

      // --- D. Grouped Query Attention ---
      hip.gqa(%handle, %q, %k, %v, %kv_cache, %attn_out, %layer, %c0, %c0)
        : (memref<4x128xf32, 1>, memref<4x128xf32, 1>, memref<4x128xf32, 1>,
           memref<2x64x128xf32, 1>, memref<4x128xf32, 1>,
           index, index, index) -> ()

      // --- E. Output Projection ---
      hip.hipblaslt.graph {
        hip.hipblaslt.matmul(%handle, %attn_out, %w_o, %proj)
          : (memref<4x128xf32, 1>, memref<128x128xf32, 1>, memref<4x128xf32, 1>) -> ()
      }

      // --- F. Fused Residual Add + Pre-MLP RMS Norm ---
      hip.miopen.graph {
        hip.miopen.skip_rms_norm(%handle, %x, %proj, %w_rms_mlp, %x_norm, %residual)
          : (memref<4x128xf32, 1>, memref<4x128xf32, 1>, memref<128xf32, 1>,
             memref<4x128xf32, 1>, memref<4x128xf32, 1>) -> ()
      }

      // --- G. MLP: Gate + Up Projections ---
      hip.hipblaslt.graph {
        hip.hipblaslt.matmul(%handle, %x_norm, %w_gate, %gate_buf)
          : (memref<4x128xf32, 1>, memref<128x344xf32, 1>, memref<4x344xf32, 1>) -> ()
        hip.hipblaslt.matmul(%handle, %x_norm, %w_up, %up_buf)
          : (memref<4x128xf32, 1>, memref<128x344xf32, 1>, memref<4x344xf32, 1>) -> ()
      }

      // --- H. SiLU Activation + Gate ---
      hip.silu(%handle, %gate_buf, %silu_buf)
        : (memref<4x344xf32, 1>, memref<4x344xf32, 1>) -> ()
      hip.miopen.graph {
        hip.miopen.mul(%handle, %silu_buf, %up_buf, %mlp_buf)
          : (memref<4x344xf32, 1>, memref<4x344xf32, 1>, memref<4x344xf32, 1>) -> ()
      }

      // --- I. Down Projection ---
      hip.hipblaslt.graph {
        hip.hipblaslt.matmul(%handle, %mlp_buf, %w_down, %down_buf)
          : (memref<4x344xf32, 1>, memref<344x128xf32, 1>, memref<4x128xf32, 1>) -> ()
      }

      // --- J. Residual Add ---
      hip.miopen.graph {
        hip.miopen.add(%handle, %residual, %down_buf, %x)
          : (memref<4x128xf32, 1>, memref<4x128xf32, 1>, memref<4x128xf32, 1>) -> ()
      }
    }

    // ================================================================
    // 5. FINAL NORM + LM HEAD
    // ================================================================
    hip.miopen.graph {
      hip.miopen.rms_norm(%handle, %x, %w_rms_final, %x_norm)
        : (memref<4x128xf32, 1>, memref<128xf32, 1>, memref<4x128xf32, 1>) -> ()
    }
    hip.hipblaslt.graph {
      hip.hipblaslt.matmul(%handle, %x_norm, %w_head, %logits)
        : (memref<4x128xf32, 1>, memref<128x1000xf32, 1>, memref<4x1000xf32, 1>) -> ()
    }

    // ================================================================
    // 6. CLEANUP
    // ================================================================
    hip.free(%handle, %x) : memref<4x128xf32, 1>
    hip.free(%handle, %x_norm) : memref<4x128xf32, 1>
    hip.free(%handle, %residual) : memref<4x128xf32, 1>
    hip.free(%handle, %q) : memref<4x128xf32, 1>
    hip.free(%handle, %k) : memref<4x128xf32, 1>
    hip.free(%handle, %v) : memref<4x128xf32, 1>
    hip.free(%handle, %attn_out) : memref<4x128xf32, 1>
    hip.free(%handle, %proj) : memref<4x128xf32, 1>
    hip.free(%handle, %gate_buf) : memref<4x344xf32, 1>
    hip.free(%handle, %up_buf) : memref<4x344xf32, 1>
    hip.free(%handle, %silu_buf) : memref<4x344xf32, 1>
    hip.free(%handle, %mlp_buf) : memref<4x344xf32, 1>
    hip.free(%handle, %down_buf) : memref<4x128xf32, 1>
    hip.free(%handle, %w_rms_attn) : memref<128xf32, 1>
    hip.free(%handle, %w_rms_mlp) : memref<128xf32, 1>
    hip.free(%handle, %w_rms_final) : memref<128xf32, 1>
    hip.free(%handle, %w_q) : memref<128x128xf32, 1>
    hip.free(%handle, %w_k) : memref<128x128xf32, 1>
    hip.free(%handle, %w_v) : memref<128x128xf32, 1>
    hip.free(%handle, %w_o) : memref<128x128xf32, 1>
    hip.free(%handle, %w_gate) : memref<128x344xf32, 1>
    hip.free(%handle, %w_up) : memref<128x344xf32, 1>
    hip.free(%handle, %w_down) : memref<344x128xf32, 1>
    hip.free(%handle, %w_head) : memref<128x1000xf32, 1>
    hip.free(%handle, %input_ids) : memref<4xi32, 1>
    hip.free(%handle, %emb_table) : memref<1000x128xf32, 1>
    hip.free(%handle, %logits) : memref<4x1000xf32, 1>
    hip.free(%handle, %kv_cache) : memref<2x64x128xf32, 1>
    hip.free(%handle, %cos_cache) : memref<64x64xf32, 1>
    hip.free(%handle, %sin_cache) : memref<64x64xf32, 1>
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
