module {
  func.func @main(
    %input_ids: memref<?xi32, 1>,          // Input Tokens [N]
    %logits:    memref<?x32000xf32, 1>,    // Output Logits [N, Vocab]
    %kv_cache:  memref<32x4096x128xf32, 1>,// Layers x MaxSeq x Dim
    %start_pos: index,                     // Current Sequence Position
    // Weights (Device Pointers)
    %w_emb:  memref<32000x128xf32, 1>,
    %w_q:    memref<128x128xf32, 1>,
    %w_k:    memref<128x128xf32, 1>,
    %w_v:    memref<128x128xf32, 1>,
    %w_o:    memref<128x128xf32, 1>,
    %w_gate: memref<128x344xf32, 1>,
    %w_up:   memref<128x344xf32, 1>,
    %w_down: memref<344x128xf32, 1>,
    %w_rms_attn:  memref<128xf32, 1>,      // RMS norm weight (pre-attention)
    %w_rms_mlp:   memref<128xf32, 1>,      // RMS norm weight (pre-MLP)
    %w_rms_final: memref<128xf32, 1>,      // RMS norm weight (final)
    %w_head: memref<128x32000xf32, 1>,
    %cos_cache: memref<4096x64xf32, 1>,    // RoPE cos cache
    %sin_cache: memref<4096x64xf32, 1>     // RoPE sin cache
  ) {
    // 1. INITIALIZE LIBRARY HANDLE
    // --------------------------------------------------------
    %handle = hip.create_handle() : !hip.handle
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %N = memref.dim %input_ids, %c0 : memref<?xi32, 1>

    // 2. DEVICE MEMORY PRE-ALLOCATION
    // --------------------------------------------------------
    // Residual Stream & Norm
    %x         = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
    %x_norm    = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
    %residual  = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
    // Attention Intermediates
    %q         = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
    %k         = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
    %v         = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
    %attn_out  = hip.alloc(%handle, %N) : memref<?x128xf32, 1>
    // MLP Intermediates
    %gate_buf  = hip.alloc(%handle, %N) : memref<?x344xf32, 1>
    %up_buf    = hip.alloc(%handle, %N) : memref<?x344xf32, 1>
    %silu_buf  = hip.alloc(%handle, %N) : memref<?x344xf32, 1>
    %mlp_buf   = hip.alloc(%handle, %N) : memref<?x344xf32, 1>
    %down_buf  = hip.alloc(%handle, %N) : memref<?x128xf32, 1>

    // 3. EMBEDDING LOOKUP
    // --------------------------------------------------------
    hip.gather(%handle, %input_ids, %w_emb, %x)
      : (memref<?xi32, 1>, memref<32000x128xf32, 1>, memref<?x128xf32, 1>) -> ()

    // 4. TRANSFORMER LAYER LOOP
    // --------------------------------------------------------
    scf.for %layer = %c0 to %c32 step %c1 {

      // --- A. Pre-Attention RMS Norm ---
      // x_norm = RMSNorm(x) * w_rms_attn
      // Maps to: miopenT5LayerNormForward(mode=MIOPEN_ELEMENTWISE_AFFINE_T5)
      hip.miopen.graph {
        hip.miopen.rms_norm(%handle, %x, %w_rms_attn, %x_norm)
          : (memref<?x128xf32, 1>, memref<128xf32, 1>, memref<?x128xf32, 1>) -> ()
      }

      // --- B. Q/K/V Projections ---
      // Maps to: hipblasLtMatmul
      hip.hipblaslt.graph {
        hip.hipblaslt.matmul(%handle, %x_norm, %w_q, %q)
          : (memref<?x128xf32, 1>, memref<128x128xf32, 1>, memref<?x128xf32, 1>) -> ()
        hip.hipblaslt.matmul(%handle, %x_norm, %w_k, %k)
          : (memref<?x128xf32, 1>, memref<128x128xf32, 1>, memref<?x128xf32, 1>) -> ()
        hip.hipblaslt.matmul(%handle, %x_norm, %w_v, %v)
          : (memref<?x128xf32, 1>, memref<128x128xf32, 1>, memref<?x128xf32, 1>) -> ()
      }

      // --- C. Rotary Positional Embeddings (RoPE) ---
      // Maps to: MIOpen experimental RotaryPositionalEmbeddings
      hip.miopen.graph {
        hip.miopen.rope(%handle, %q, %k, %cos_cache, %sin_cache, %start_pos)
          : (memref<?x128xf32, 1>, memref<?x128xf32, 1>,
             memref<4096x64xf32, 1>, memref<4096x64xf32, 1>, index) -> ()
      }

      // --- D. Grouped Query Attention ---
      hip.gqa(%handle, %q, %k, %v, %kv_cache, %attn_out, %layer, %start_pos, %N)
        : (memref<?x128xf32, 1>, memref<?x128xf32, 1>, memref<?x128xf32, 1>,
           memref<32x4096x128xf32, 1>, memref<?x128xf32, 1>,
           index, index, index) -> ()

      // --- E. Output Projection ---
      hip.hipblaslt.graph {
        hip.hipblaslt.matmul(%handle, %attn_out, %w_o, %down_buf)
          : (memref<?x128xf32, 1>, memref<128x128xf32, 1>, memref<?x128xf32, 1>) -> ()
      }

      // --- F. Residual Add + Pre-MLP RMS Norm (fused) ---
      // x_norm, residual = SkipRMSNorm(x, attn_o_proj)
      // Maps to: miopenAddLayerNormForward(mode=MIOPEN_ELEMENTWISE_AFFINE_T5)
      // Single kernel: residual = x + down_buf; x_norm = RMSNorm(residual) * w_rms_mlp
      hip.miopen.graph {
        hip.miopen.skip_rms_norm(%handle, %x, %down_buf, %w_rms_mlp, %x_norm, %residual)
          : (memref<?x128xf32, 1>, memref<?x128xf32, 1>, memref<128xf32, 1>,
             memref<?x128xf32, 1>, memref<?x128xf32, 1>) -> ()
      }

      // --- G. MLP: Gate + Up Projections ---
      hip.hipblaslt.graph {
        hip.hipblaslt.matmul(%handle, %x_norm, %w_gate, %gate_buf)
          : (memref<?x128xf32, 1>, memref<128x344xf32, 1>, memref<?x344xf32, 1>) -> ()
        hip.hipblaslt.matmul(%handle, %x_norm, %w_up, %up_buf)
          : (memref<?x128xf32, 1>, memref<128x344xf32, 1>, memref<?x344xf32, 1>) -> ()
      }

      // --- H. SiLU Activation + Gate ---
      // silu_buf = SiLU(gate_buf) = gate_buf * sigmoid(gate_buf)
      hip.silu(%handle, %gate_buf, %silu_buf)
        : (memref<?x344xf32, 1>, memref<?x344xf32, 1>) -> ()
      // mlp_buf = silu_buf * up_buf
      // Maps to: miopenOpTensor(miopenTensorOpMul)
      hip.miopen.graph {
        hip.miopen.mul(%handle, %silu_buf, %up_buf, %mlp_buf)
          : (memref<?x344xf32, 1>, memref<?x344xf32, 1>, memref<?x344xf32, 1>) -> ()
      }

      // --- I. Down Projection ---
      hip.hipblaslt.graph {
        hip.hipblaslt.matmul(%handle, %mlp_buf, %w_down, %down_buf)
          : (memref<?x344xf32, 1>, memref<344x128xf32, 1>, memref<?x128xf32, 1>) -> ()
      }

      // --- J. Residual Add (x = residual + mlp_out) ---
      // Maps to: miopenOpTensor(miopenTensorOpAdd)
      hip.miopen.graph {
        hip.miopen.add(%handle, %residual, %down_buf, %x)
          : (memref<?x128xf32, 1>, memref<?x128xf32, 1>, memref<?x128xf32, 1>) -> ()
      }

    } // End Layer Loop

    // 5. FINAL NORM + LM HEAD
    // --------------------------------------------------------
    hip.miopen.graph {
      hip.miopen.rms_norm(%handle, %x, %w_rms_final, %x_norm)
        : (memref<?x128xf32, 1>, memref<128xf32, 1>, memref<?x128xf32, 1>) -> ()
    }
    hip.hipblaslt.graph {
      hip.hipblaslt.matmul(%handle, %x_norm, %w_head, %logits)
        : (memref<?x128xf32, 1>, memref<128x32000xf32, 1>, memref<?x32000xf32, 1>) -> ()
    }

    // 6. CLEANUP
    // --------------------------------------------------------
    hip.free(%handle, %x) : memref<?x128xf32, 1>
    hip.free(%handle, %x_norm) : memref<?x128xf32, 1>
    hip.free(%handle, %residual) : memref<?x128xf32, 1>
    hip.free(%handle, %q) : memref<?x128xf32, 1>
    hip.free(%handle, %k) : memref<?x128xf32, 1>
    hip.free(%handle, %v) : memref<?x128xf32, 1>
    hip.free(%handle, %attn_out) : memref<?x128xf32, 1>
    hip.free(%handle, %gate_buf) : memref<?x344xf32, 1>
    hip.free(%handle, %up_buf) : memref<?x344xf32, 1>
    hip.free(%handle, %silu_buf) : memref<?x344xf32, 1>
    hip.free(%handle, %mlp_buf) : memref<?x344xf32, 1>
    hip.free(%handle, %down_buf) : memref<?x128xf32, 1>
    hip.destroy_handle(%handle) : !hip.handle
    return
  }
}
