module {
  func.func @main_graph(
      %input_ids: memref<?x?xi64, 1>,
    %attention_mask: memref<?x?xi64, 1>,
    %past_key_values_0_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_0_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_1_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_1_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_2_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_2_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_3_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_3_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_4_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_4_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_5_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_5_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_6_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_6_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_7_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_7_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_8_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_8_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_9_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_9_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_10_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_10_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_11_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_11_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_12_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_12_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_13_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_13_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_14_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_14_value: memref<?x8x?x64xf32, 1>,
    %past_key_values_15_key: memref<?x8x?x64xf32, 1>,
    %past_key_values_15_value: memref<?x8x?x64xf32, 1>,
    %model_embed_tokens_weight: memref<128256x2048xf32, 1>  /* weight: model.embed_tokens.weight */,
    %model_layers_0_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.0.input_layernorm.weight */,
    %model_layers_0_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.0.attn.q_proj.MatMul.weight */,
    %model_layers_0_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.0.attn.k_proj.MatMul.weight */,
    %model_layers_0_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.0.attn.v_proj.MatMul.weight */,
    %cos_cache: memref<131072x32xf32, 1>  /* weight: cos_cache */,
    %sin_cache: memref<131072x32xf32, 1>  /* weight: sin_cache */,
    %model_layers_0_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.0.attn.o_proj.MatMul.weight */,
    %model_layers_0_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.0.post_attention_layernorm.weight */,
    %model_layers_0_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.0.mlp.gate_proj.MatMul.weight */,
    %model_layers_0_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.0.mlp.up_proj.MatMul.weight */,
    %model_layers_0_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.0.mlp.down_proj.MatMul.weight */,
    %model_layers_1_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.1.input_layernorm.weight */,
    %model_layers_1_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.1.attn.q_proj.MatMul.weight */,
    %model_layers_1_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.1.attn.k_proj.MatMul.weight */,
    %model_layers_1_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.1.attn.v_proj.MatMul.weight */,
    %model_layers_1_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.1.attn.o_proj.MatMul.weight */,
    %model_layers_1_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.1.post_attention_layernorm.weight */,
    %model_layers_1_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.1.mlp.gate_proj.MatMul.weight */,
    %model_layers_1_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.1.mlp.up_proj.MatMul.weight */,
    %model_layers_1_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.1.mlp.down_proj.MatMul.weight */,
    %model_layers_2_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.2.input_layernorm.weight */,
    %model_layers_2_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.2.attn.q_proj.MatMul.weight */,
    %model_layers_2_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.2.attn.k_proj.MatMul.weight */,
    %model_layers_2_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.2.attn.v_proj.MatMul.weight */,
    %model_layers_2_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.2.attn.o_proj.MatMul.weight */,
    %model_layers_2_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.2.post_attention_layernorm.weight */,
    %model_layers_2_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.2.mlp.gate_proj.MatMul.weight */,
    %model_layers_2_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.2.mlp.up_proj.MatMul.weight */,
    %model_layers_2_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.2.mlp.down_proj.MatMul.weight */,
    %model_layers_3_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.3.input_layernorm.weight */,
    %model_layers_3_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.3.attn.q_proj.MatMul.weight */,
    %model_layers_3_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.3.attn.k_proj.MatMul.weight */,
    %model_layers_3_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.3.attn.v_proj.MatMul.weight */,
    %model_layers_3_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.3.attn.o_proj.MatMul.weight */,
    %model_layers_3_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.3.post_attention_layernorm.weight */,
    %model_layers_3_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.3.mlp.gate_proj.MatMul.weight */,
    %model_layers_3_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.3.mlp.up_proj.MatMul.weight */,
    %model_layers_3_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.3.mlp.down_proj.MatMul.weight */,
    %model_layers_4_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.4.input_layernorm.weight */,
    %model_layers_4_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.4.attn.q_proj.MatMul.weight */,
    %model_layers_4_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.4.attn.k_proj.MatMul.weight */,
    %model_layers_4_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.4.attn.v_proj.MatMul.weight */,
    %model_layers_4_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.4.attn.o_proj.MatMul.weight */,
    %model_layers_4_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.4.post_attention_layernorm.weight */,
    %model_layers_4_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.4.mlp.gate_proj.MatMul.weight */,
    %model_layers_4_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.4.mlp.up_proj.MatMul.weight */,
    %model_layers_4_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.4.mlp.down_proj.MatMul.weight */,
    %model_layers_5_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.5.input_layernorm.weight */,
    %model_layers_5_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.5.attn.q_proj.MatMul.weight */,
    %model_layers_5_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.5.attn.k_proj.MatMul.weight */,
    %model_layers_5_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.5.attn.v_proj.MatMul.weight */,
    %model_layers_5_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.5.attn.o_proj.MatMul.weight */,
    %model_layers_5_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.5.post_attention_layernorm.weight */,
    %model_layers_5_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.5.mlp.gate_proj.MatMul.weight */,
    %model_layers_5_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.5.mlp.up_proj.MatMul.weight */,
    %model_layers_5_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.5.mlp.down_proj.MatMul.weight */,
    %model_layers_6_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.6.input_layernorm.weight */,
    %model_layers_6_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.6.attn.q_proj.MatMul.weight */,
    %model_layers_6_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.6.attn.k_proj.MatMul.weight */,
    %model_layers_6_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.6.attn.v_proj.MatMul.weight */,
    %model_layers_6_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.6.attn.o_proj.MatMul.weight */,
    %model_layers_6_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.6.post_attention_layernorm.weight */,
    %model_layers_6_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.6.mlp.gate_proj.MatMul.weight */,
    %model_layers_6_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.6.mlp.up_proj.MatMul.weight */,
    %model_layers_6_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.6.mlp.down_proj.MatMul.weight */,
    %model_layers_7_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.7.input_layernorm.weight */,
    %model_layers_7_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.7.attn.q_proj.MatMul.weight */,
    %model_layers_7_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.7.attn.k_proj.MatMul.weight */,
    %model_layers_7_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.7.attn.v_proj.MatMul.weight */,
    %model_layers_7_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.7.attn.o_proj.MatMul.weight */,
    %model_layers_7_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.7.post_attention_layernorm.weight */,
    %model_layers_7_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.7.mlp.gate_proj.MatMul.weight */,
    %model_layers_7_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.7.mlp.up_proj.MatMul.weight */,
    %model_layers_7_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.7.mlp.down_proj.MatMul.weight */,
    %model_layers_8_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.8.input_layernorm.weight */,
    %model_layers_8_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.8.attn.q_proj.MatMul.weight */,
    %model_layers_8_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.8.attn.k_proj.MatMul.weight */,
    %model_layers_8_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.8.attn.v_proj.MatMul.weight */,
    %model_layers_8_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.8.attn.o_proj.MatMul.weight */,
    %model_layers_8_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.8.post_attention_layernorm.weight */,
    %model_layers_8_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.8.mlp.gate_proj.MatMul.weight */,
    %model_layers_8_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.8.mlp.up_proj.MatMul.weight */,
    %model_layers_8_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.8.mlp.down_proj.MatMul.weight */,
    %model_layers_9_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.9.input_layernorm.weight */,
    %model_layers_9_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.9.attn.q_proj.MatMul.weight */,
    %model_layers_9_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.9.attn.k_proj.MatMul.weight */,
    %model_layers_9_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.9.attn.v_proj.MatMul.weight */,
    %model_layers_9_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.9.attn.o_proj.MatMul.weight */,
    %model_layers_9_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.9.post_attention_layernorm.weight */,
    %model_layers_9_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.9.mlp.gate_proj.MatMul.weight */,
    %model_layers_9_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.9.mlp.up_proj.MatMul.weight */,
    %model_layers_9_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.9.mlp.down_proj.MatMul.weight */,
    %model_layers_10_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.10.input_layernorm.weight */,
    %model_layers_10_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.10.attn.q_proj.MatMul.weight */,
    %model_layers_10_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.10.attn.k_proj.MatMul.weight */,
    %model_layers_10_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.10.attn.v_proj.MatMul.weight */,
    %model_layers_10_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.10.attn.o_proj.MatMul.weight */,
    %model_layers_10_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.10.post_attention_layernorm.weight */,
    %model_layers_10_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.10.mlp.gate_proj.MatMul.weight */,
    %model_layers_10_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.10.mlp.up_proj.MatMul.weight */,
    %model_layers_10_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.10.mlp.down_proj.MatMul.weight */,
    %model_layers_11_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.11.input_layernorm.weight */,
    %model_layers_11_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.11.attn.q_proj.MatMul.weight */,
    %model_layers_11_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.11.attn.k_proj.MatMul.weight */,
    %model_layers_11_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.11.attn.v_proj.MatMul.weight */,
    %model_layers_11_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.11.attn.o_proj.MatMul.weight */,
    %model_layers_11_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.11.post_attention_layernorm.weight */,
    %model_layers_11_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.11.mlp.gate_proj.MatMul.weight */,
    %model_layers_11_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.11.mlp.up_proj.MatMul.weight */,
    %model_layers_11_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.11.mlp.down_proj.MatMul.weight */,
    %model_layers_12_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.12.input_layernorm.weight */,
    %model_layers_12_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.12.attn.q_proj.MatMul.weight */,
    %model_layers_12_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.12.attn.k_proj.MatMul.weight */,
    %model_layers_12_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.12.attn.v_proj.MatMul.weight */,
    %model_layers_12_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.12.attn.o_proj.MatMul.weight */,
    %model_layers_12_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.12.post_attention_layernorm.weight */,
    %model_layers_12_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.12.mlp.gate_proj.MatMul.weight */,
    %model_layers_12_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.12.mlp.up_proj.MatMul.weight */,
    %model_layers_12_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.12.mlp.down_proj.MatMul.weight */,
    %model_layers_13_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.13.input_layernorm.weight */,
    %model_layers_13_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.13.attn.q_proj.MatMul.weight */,
    %model_layers_13_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.13.attn.k_proj.MatMul.weight */,
    %model_layers_13_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.13.attn.v_proj.MatMul.weight */,
    %model_layers_13_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.13.attn.o_proj.MatMul.weight */,
    %model_layers_13_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.13.post_attention_layernorm.weight */,
    %model_layers_13_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.13.mlp.gate_proj.MatMul.weight */,
    %model_layers_13_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.13.mlp.up_proj.MatMul.weight */,
    %model_layers_13_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.13.mlp.down_proj.MatMul.weight */,
    %model_layers_14_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.14.input_layernorm.weight */,
    %model_layers_14_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.14.attn.q_proj.MatMul.weight */,
    %model_layers_14_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.14.attn.k_proj.MatMul.weight */,
    %model_layers_14_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.14.attn.v_proj.MatMul.weight */,
    %model_layers_14_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.14.attn.o_proj.MatMul.weight */,
    %model_layers_14_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.14.post_attention_layernorm.weight */,
    %model_layers_14_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.14.mlp.gate_proj.MatMul.weight */,
    %model_layers_14_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.14.mlp.up_proj.MatMul.weight */,
    %model_layers_14_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.14.mlp.down_proj.MatMul.weight */,
    %model_layers_15_input_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.15.input_layernorm.weight */,
    %model_layers_15_attn_q_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.15.attn.q_proj.MatMul.weight */,
    %model_layers_15_attn_k_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.15.attn.k_proj.MatMul.weight */,
    %model_layers_15_attn_v_proj_MatMul_weight: memref<2048x512xf32, 1>  /* weight: model.layers.15.attn.v_proj.MatMul.weight */,
    %model_layers_15_attn_o_proj_MatMul_weight: memref<2048x2048xf32, 1>  /* weight: model.layers.15.attn.o_proj.MatMul.weight */,
    %model_layers_15_post_attention_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.15.post_attention_layernorm.weight */,
    %model_layers_15_mlp_gate_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.15.mlp.gate_proj.MatMul.weight */,
    %model_layers_15_mlp_up_proj_MatMul_weight: memref<2048x8192xf32, 1>  /* weight: model.layers.15.mlp.up_proj.MatMul.weight */,
    %model_layers_15_mlp_down_proj_MatMul_weight: memref<8192x2048xf32, 1>  /* weight: model.layers.15.mlp.down_proj.MatMul.weight */,
    %model_layers_16_final_norm_layernorm_weight: memref<2048xf32, 1>  /* weight: model.layers.16.final_norm_layernorm.weight */
    ) -> (memref<?x?x128256xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>) {
    %handle = hip.create_handle() : !hip.handle

    %_model_constants_INT64__1_ = "hip.constant"() {value = dense<0> : tensor<1xi64>} : () -> memref<1xi64, 1>  // /model/constant_nodes/INT64/[1]
    %_model_attn_mask_reformat_attn_mask_subgraph_ReduceSum_output_0 = "hip.ReduceSum"(%attention_mask, %_model_constants_INT64__1_) {keepdims = 1 : i64} : (memref<?x?xi64, 1>, memref<1xi64, 1>) -> memref<?x1xi64, 1>  // /model/attn_mask_reformat/attn_mask_subgraph/ReduceSum
    %_model_attn_mask_reformat_attn_mask_subgraph_Sub_output_0 = "hip.Sub"(%_model_attn_mask_reformat_attn_mask_subgraph_ReduceSum_output_0, %_model_constants_INT64__1_) : (memref<?x1xi64, 1>, memref<1xi64, 1>) -> memref<?x1xi64, 1>  // /model/attn_mask_reformat/attn_mask_subgraph/Sub
    %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0 = "hip.cast"(%_model_attn_mask_reformat_attn_mask_subgraph_Sub_output_0) {to = 6 : i64} : (memref<?x1xi64, 1>) -> memref<?x1xi32, 1>  // /model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast
    %_model_attn_mask_reformat_attn_mask_subgraph_Shape_output_0 = "hip.Shape"(%attention_mask) : (memref<?x?xi64, 1>) -> memref<2xi64, 1>  // /model/attn_mask_reformat/attn_mask_subgraph/Shape
    %_model_constants_INT64_1 = "hip.constant"() {value = dense<0> : tensor<i64>} : () -> memref<i64, 1>  // /model/constant_nodes/INT64/1
    %_model_attn_mask_reformat_attn_mask_subgraph_Gather_1_output_0 = "hip.gather"(%_model_attn_mask_reformat_attn_mask_subgraph_Shape_output_0, %_model_constants_INT64_1) {axis = 0 : i64} : (memref<2xi64, 1>, memref<i64, 1>) -> memref<i64, 1>  // /model/attn_mask_reformat/attn_mask_subgraph/Gather_1
    %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0 = "hip.cast"(%_model_attn_mask_reformat_attn_mask_subgraph_Gather_1_output_0) {to = 6 : i64} : (memref<i64, 1>) -> memref<i32, 1>  // /model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast
    %_model_embed_tokens_Gather_output_0 = "hip.gather"(%model_embed_tokens_weight, %input_ids) : (memref<128256x2048xf32, 1>, memref<?x?xi64, 1>) -> memref<?x?x2048xf32, 1>  // /model/embed_tokens/Gather
    hip.miopen.graph {
      %_model_layers_0_input_layernorm_output_0 = "hip.miopen.t5_layer_norm"(%_model_embed_tokens_Gather_output_0, %model_layers_0_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32, axis = -1 : i64, stash_type = 1 : i64} : (memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.0/input_layernorm/LayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_0_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_0_input_layernorm_output_0, %model_layers_0_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.0/attn/q_proj/MatMul
      %_model_layers_0_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_0_input_layernorm_output_0, %model_layers_0_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.0/attn/k_proj/MatMul
      %_model_layers_0_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_0_input_layernorm_output_0, %model_layers_0_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.0/attn/v_proj/MatMul
    }
    %_model_layers_0_attn_GroupQueryAttention_output_0, %present_0_key, %present_0_value = "hip.gqa"(%_model_layers_0_attn_q_proj_MatMul_output_0, %_model_layers_0_attn_k_proj_MatMul_output_0, %_model_layers_0_attn_v_proj_MatMul_output_0, %past_key_values_0_key, %past_key_values_0_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.0/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_0_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_0_attn_GroupQueryAttention_output_0, %model_layers_0_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.0/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_0_post_attention_layernorm_output_0, %v, %v_2, %_model_layers_0_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_embed_tokens_Gather_output_0, %_model_layers_0_attn_o_proj_MatMul_output_0, %model_layers_0_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.0/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_0_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_0_post_attention_layernorm_output_0, %model_layers_0_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.0/mlp/gate_proj/MatMul
      %_model_layers_0_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_0_post_attention_layernorm_output_0, %model_layers_0_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.0/mlp/up_proj/MatMul
    }
    %_model_layers_0_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_0_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_0_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_0_mlp_act_fn_Mul_output_0, %_model_layers_0_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.0/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_0_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_0_mlp_Mul_output_0, %model_layers_0_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.0/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_1_input_layernorm_output_0, %v_3, %v_4, %_model_layers_1_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_0_post_attention_layernorm_output_3, %_model_layers_0_mlp_down_proj_MatMul_output_0, %model_layers_1_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.1/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_1_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_1_input_layernorm_output_0, %model_layers_1_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.1/attn/q_proj/MatMul
      %_model_layers_1_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_1_input_layernorm_output_0, %model_layers_1_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.1/attn/k_proj/MatMul
      %_model_layers_1_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_1_input_layernorm_output_0, %model_layers_1_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.1/attn/v_proj/MatMul
    }
    %_model_layers_1_attn_GroupQueryAttention_output_0, %present_1_key, %present_1_value = "hip.gqa"(%_model_layers_1_attn_q_proj_MatMul_output_0, %_model_layers_1_attn_k_proj_MatMul_output_0, %_model_layers_1_attn_v_proj_MatMul_output_0, %past_key_values_1_key, %past_key_values_1_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.1/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_1_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_1_attn_GroupQueryAttention_output_0, %model_layers_1_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.1/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_1_post_attention_layernorm_output_0, %v_5, %v_6, %_model_layers_1_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_1_input_layernorm_output_3, %_model_layers_1_attn_o_proj_MatMul_output_0, %model_layers_1_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.1/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_1_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_1_post_attention_layernorm_output_0, %model_layers_1_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.1/mlp/gate_proj/MatMul
      %_model_layers_1_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_1_post_attention_layernorm_output_0, %model_layers_1_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.1/mlp/up_proj/MatMul
    }
    %_model_layers_1_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_1_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_1_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_1_mlp_act_fn_Mul_output_0, %_model_layers_1_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.1/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_1_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_1_mlp_Mul_output_0, %model_layers_1_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.1/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_2_input_layernorm_output_0, %v_7, %v_8, %_model_layers_2_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_1_post_attention_layernorm_output_3, %_model_layers_1_mlp_down_proj_MatMul_output_0, %model_layers_2_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.2/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_2_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_2_input_layernorm_output_0, %model_layers_2_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.2/attn/q_proj/MatMul
      %_model_layers_2_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_2_input_layernorm_output_0, %model_layers_2_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.2/attn/k_proj/MatMul
      %_model_layers_2_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_2_input_layernorm_output_0, %model_layers_2_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.2/attn/v_proj/MatMul
    }
    %_model_layers_2_attn_GroupQueryAttention_output_0, %present_2_key, %present_2_value = "hip.gqa"(%_model_layers_2_attn_q_proj_MatMul_output_0, %_model_layers_2_attn_k_proj_MatMul_output_0, %_model_layers_2_attn_v_proj_MatMul_output_0, %past_key_values_2_key, %past_key_values_2_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.2/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_2_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_2_attn_GroupQueryAttention_output_0, %model_layers_2_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.2/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_2_post_attention_layernorm_output_0, %v_9, %v_10, %_model_layers_2_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_2_input_layernorm_output_3, %_model_layers_2_attn_o_proj_MatMul_output_0, %model_layers_2_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.2/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_2_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_2_post_attention_layernorm_output_0, %model_layers_2_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.2/mlp/gate_proj/MatMul
      %_model_layers_2_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_2_post_attention_layernorm_output_0, %model_layers_2_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.2/mlp/up_proj/MatMul
    }
    %_model_layers_2_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_2_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_2_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_2_mlp_act_fn_Mul_output_0, %_model_layers_2_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.2/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_2_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_2_mlp_Mul_output_0, %model_layers_2_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.2/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_3_input_layernorm_output_0, %v_11, %v_12, %_model_layers_3_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_2_post_attention_layernorm_output_3, %_model_layers_2_mlp_down_proj_MatMul_output_0, %model_layers_3_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.3/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_3_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_3_input_layernorm_output_0, %model_layers_3_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.3/attn/q_proj/MatMul
      %_model_layers_3_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_3_input_layernorm_output_0, %model_layers_3_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.3/attn/k_proj/MatMul
      %_model_layers_3_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_3_input_layernorm_output_0, %model_layers_3_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.3/attn/v_proj/MatMul
    }
    %_model_layers_3_attn_GroupQueryAttention_output_0, %present_3_key, %present_3_value = "hip.gqa"(%_model_layers_3_attn_q_proj_MatMul_output_0, %_model_layers_3_attn_k_proj_MatMul_output_0, %_model_layers_3_attn_v_proj_MatMul_output_0, %past_key_values_3_key, %past_key_values_3_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.3/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_3_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_3_attn_GroupQueryAttention_output_0, %model_layers_3_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.3/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_3_post_attention_layernorm_output_0, %v_13, %v_14, %_model_layers_3_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_3_input_layernorm_output_3, %_model_layers_3_attn_o_proj_MatMul_output_0, %model_layers_3_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.3/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_3_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_3_post_attention_layernorm_output_0, %model_layers_3_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.3/mlp/gate_proj/MatMul
      %_model_layers_3_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_3_post_attention_layernorm_output_0, %model_layers_3_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.3/mlp/up_proj/MatMul
    }
    %_model_layers_3_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_3_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_3_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_3_mlp_act_fn_Mul_output_0, %_model_layers_3_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.3/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_3_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_3_mlp_Mul_output_0, %model_layers_3_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.3/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_4_input_layernorm_output_0, %v_15, %v_16, %_model_layers_4_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_3_post_attention_layernorm_output_3, %_model_layers_3_mlp_down_proj_MatMul_output_0, %model_layers_4_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.4/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_4_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_4_input_layernorm_output_0, %model_layers_4_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.4/attn/q_proj/MatMul
      %_model_layers_4_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_4_input_layernorm_output_0, %model_layers_4_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.4/attn/k_proj/MatMul
      %_model_layers_4_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_4_input_layernorm_output_0, %model_layers_4_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.4/attn/v_proj/MatMul
    }
    %_model_layers_4_attn_GroupQueryAttention_output_0, %present_4_key, %present_4_value = "hip.gqa"(%_model_layers_4_attn_q_proj_MatMul_output_0, %_model_layers_4_attn_k_proj_MatMul_output_0, %_model_layers_4_attn_v_proj_MatMul_output_0, %past_key_values_4_key, %past_key_values_4_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.4/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_4_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_4_attn_GroupQueryAttention_output_0, %model_layers_4_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.4/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_4_post_attention_layernorm_output_0, %v_17, %v_18, %_model_layers_4_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_4_input_layernorm_output_3, %_model_layers_4_attn_o_proj_MatMul_output_0, %model_layers_4_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.4/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_4_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_4_post_attention_layernorm_output_0, %model_layers_4_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.4/mlp/gate_proj/MatMul
      %_model_layers_4_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_4_post_attention_layernorm_output_0, %model_layers_4_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.4/mlp/up_proj/MatMul
    }
    %_model_layers_4_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_4_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_4_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_4_mlp_act_fn_Mul_output_0, %_model_layers_4_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.4/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_4_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_4_mlp_Mul_output_0, %model_layers_4_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.4/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_5_input_layernorm_output_0, %v_19, %v_20, %_model_layers_5_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_4_post_attention_layernorm_output_3, %_model_layers_4_mlp_down_proj_MatMul_output_0, %model_layers_5_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.5/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_5_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_5_input_layernorm_output_0, %model_layers_5_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.5/attn/q_proj/MatMul
      %_model_layers_5_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_5_input_layernorm_output_0, %model_layers_5_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.5/attn/k_proj/MatMul
      %_model_layers_5_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_5_input_layernorm_output_0, %model_layers_5_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.5/attn/v_proj/MatMul
    }
    %_model_layers_5_attn_GroupQueryAttention_output_0, %present_5_key, %present_5_value = "hip.gqa"(%_model_layers_5_attn_q_proj_MatMul_output_0, %_model_layers_5_attn_k_proj_MatMul_output_0, %_model_layers_5_attn_v_proj_MatMul_output_0, %past_key_values_5_key, %past_key_values_5_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.5/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_5_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_5_attn_GroupQueryAttention_output_0, %model_layers_5_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.5/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_5_post_attention_layernorm_output_0, %v_21, %v_22, %_model_layers_5_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_5_input_layernorm_output_3, %_model_layers_5_attn_o_proj_MatMul_output_0, %model_layers_5_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.5/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_5_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_5_post_attention_layernorm_output_0, %model_layers_5_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.5/mlp/gate_proj/MatMul
      %_model_layers_5_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_5_post_attention_layernorm_output_0, %model_layers_5_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.5/mlp/up_proj/MatMul
    }
    %_model_layers_5_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_5_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_5_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_5_mlp_act_fn_Mul_output_0, %_model_layers_5_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.5/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_5_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_5_mlp_Mul_output_0, %model_layers_5_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.5/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_6_input_layernorm_output_0, %v_23, %v_24, %_model_layers_6_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_5_post_attention_layernorm_output_3, %_model_layers_5_mlp_down_proj_MatMul_output_0, %model_layers_6_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.6/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_6_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_6_input_layernorm_output_0, %model_layers_6_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.6/attn/q_proj/MatMul
      %_model_layers_6_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_6_input_layernorm_output_0, %model_layers_6_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.6/attn/k_proj/MatMul
      %_model_layers_6_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_6_input_layernorm_output_0, %model_layers_6_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.6/attn/v_proj/MatMul
    }
    %_model_layers_6_attn_GroupQueryAttention_output_0, %present_6_key, %present_6_value = "hip.gqa"(%_model_layers_6_attn_q_proj_MatMul_output_0, %_model_layers_6_attn_k_proj_MatMul_output_0, %_model_layers_6_attn_v_proj_MatMul_output_0, %past_key_values_6_key, %past_key_values_6_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.6/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_6_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_6_attn_GroupQueryAttention_output_0, %model_layers_6_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.6/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_6_post_attention_layernorm_output_0, %v_25, %v_26, %_model_layers_6_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_6_input_layernorm_output_3, %_model_layers_6_attn_o_proj_MatMul_output_0, %model_layers_6_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.6/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_6_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_6_post_attention_layernorm_output_0, %model_layers_6_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.6/mlp/gate_proj/MatMul
      %_model_layers_6_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_6_post_attention_layernorm_output_0, %model_layers_6_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.6/mlp/up_proj/MatMul
    }
    %_model_layers_6_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_6_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_6_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_6_mlp_act_fn_Mul_output_0, %_model_layers_6_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.6/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_6_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_6_mlp_Mul_output_0, %model_layers_6_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.6/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_7_input_layernorm_output_0, %v_27, %v_28, %_model_layers_7_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_6_post_attention_layernorm_output_3, %_model_layers_6_mlp_down_proj_MatMul_output_0, %model_layers_7_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.7/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_7_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_7_input_layernorm_output_0, %model_layers_7_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.7/attn/q_proj/MatMul
      %_model_layers_7_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_7_input_layernorm_output_0, %model_layers_7_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.7/attn/k_proj/MatMul
      %_model_layers_7_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_7_input_layernorm_output_0, %model_layers_7_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.7/attn/v_proj/MatMul
    }
    %_model_layers_7_attn_GroupQueryAttention_output_0, %present_7_key, %present_7_value = "hip.gqa"(%_model_layers_7_attn_q_proj_MatMul_output_0, %_model_layers_7_attn_k_proj_MatMul_output_0, %_model_layers_7_attn_v_proj_MatMul_output_0, %past_key_values_7_key, %past_key_values_7_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.7/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_7_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_7_attn_GroupQueryAttention_output_0, %model_layers_7_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.7/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_7_post_attention_layernorm_output_0, %v_29, %v_30, %_model_layers_7_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_7_input_layernorm_output_3, %_model_layers_7_attn_o_proj_MatMul_output_0, %model_layers_7_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.7/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_7_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_7_post_attention_layernorm_output_0, %model_layers_7_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.7/mlp/gate_proj/MatMul
      %_model_layers_7_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_7_post_attention_layernorm_output_0, %model_layers_7_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.7/mlp/up_proj/MatMul
    }
    %_model_layers_7_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_7_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_7_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_7_mlp_act_fn_Mul_output_0, %_model_layers_7_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.7/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_7_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_7_mlp_Mul_output_0, %model_layers_7_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.7/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_8_input_layernorm_output_0, %v_31, %v_32, %_model_layers_8_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_7_post_attention_layernorm_output_3, %_model_layers_7_mlp_down_proj_MatMul_output_0, %model_layers_8_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.8/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_8_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_8_input_layernorm_output_0, %model_layers_8_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.8/attn/q_proj/MatMul
      %_model_layers_8_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_8_input_layernorm_output_0, %model_layers_8_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.8/attn/k_proj/MatMul
      %_model_layers_8_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_8_input_layernorm_output_0, %model_layers_8_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.8/attn/v_proj/MatMul
    }
    %_model_layers_8_attn_GroupQueryAttention_output_0, %present_8_key, %present_8_value = "hip.gqa"(%_model_layers_8_attn_q_proj_MatMul_output_0, %_model_layers_8_attn_k_proj_MatMul_output_0, %_model_layers_8_attn_v_proj_MatMul_output_0, %past_key_values_8_key, %past_key_values_8_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.8/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_8_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_8_attn_GroupQueryAttention_output_0, %model_layers_8_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.8/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_8_post_attention_layernorm_output_0, %v_33, %v_34, %_model_layers_8_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_8_input_layernorm_output_3, %_model_layers_8_attn_o_proj_MatMul_output_0, %model_layers_8_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.8/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_8_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_8_post_attention_layernorm_output_0, %model_layers_8_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.8/mlp/gate_proj/MatMul
      %_model_layers_8_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_8_post_attention_layernorm_output_0, %model_layers_8_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.8/mlp/up_proj/MatMul
    }
    %_model_layers_8_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_8_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_8_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_8_mlp_act_fn_Mul_output_0, %_model_layers_8_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.8/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_8_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_8_mlp_Mul_output_0, %model_layers_8_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.8/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_9_input_layernorm_output_0, %v_35, %v_36, %_model_layers_9_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_8_post_attention_layernorm_output_3, %_model_layers_8_mlp_down_proj_MatMul_output_0, %model_layers_9_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.9/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_9_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_9_input_layernorm_output_0, %model_layers_9_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.9/attn/q_proj/MatMul
      %_model_layers_9_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_9_input_layernorm_output_0, %model_layers_9_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.9/attn/k_proj/MatMul
      %_model_layers_9_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_9_input_layernorm_output_0, %model_layers_9_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.9/attn/v_proj/MatMul
    }
    %_model_layers_9_attn_GroupQueryAttention_output_0, %present_9_key, %present_9_value = "hip.gqa"(%_model_layers_9_attn_q_proj_MatMul_output_0, %_model_layers_9_attn_k_proj_MatMul_output_0, %_model_layers_9_attn_v_proj_MatMul_output_0, %past_key_values_9_key, %past_key_values_9_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.9/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_9_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_9_attn_GroupQueryAttention_output_0, %model_layers_9_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.9/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_9_post_attention_layernorm_output_0, %v_37, %v_38, %_model_layers_9_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_9_input_layernorm_output_3, %_model_layers_9_attn_o_proj_MatMul_output_0, %model_layers_9_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.9/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_9_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_9_post_attention_layernorm_output_0, %model_layers_9_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.9/mlp/gate_proj/MatMul
      %_model_layers_9_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_9_post_attention_layernorm_output_0, %model_layers_9_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.9/mlp/up_proj/MatMul
    }
    %_model_layers_9_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_9_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_9_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_9_mlp_act_fn_Mul_output_0, %_model_layers_9_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.9/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_9_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_9_mlp_Mul_output_0, %model_layers_9_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.9/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_10_input_layernorm_output_0, %v_39, %v_40, %_model_layers_10_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_9_post_attention_layernorm_output_3, %_model_layers_9_mlp_down_proj_MatMul_output_0, %model_layers_10_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.10/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_10_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_10_input_layernorm_output_0, %model_layers_10_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.10/attn/q_proj/MatMul
      %_model_layers_10_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_10_input_layernorm_output_0, %model_layers_10_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.10/attn/k_proj/MatMul
      %_model_layers_10_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_10_input_layernorm_output_0, %model_layers_10_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.10/attn/v_proj/MatMul
    }
    %_model_layers_10_attn_GroupQueryAttention_output_0, %present_10_key, %present_10_value = "hip.gqa"(%_model_layers_10_attn_q_proj_MatMul_output_0, %_model_layers_10_attn_k_proj_MatMul_output_0, %_model_layers_10_attn_v_proj_MatMul_output_0, %past_key_values_10_key, %past_key_values_10_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.10/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_10_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_10_attn_GroupQueryAttention_output_0, %model_layers_10_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.10/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_10_post_attention_layernorm_output_0, %v_41, %v_42, %_model_layers_10_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_10_input_layernorm_output_3, %_model_layers_10_attn_o_proj_MatMul_output_0, %model_layers_10_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.10/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_10_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_10_post_attention_layernorm_output_0, %model_layers_10_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.10/mlp/gate_proj/MatMul
      %_model_layers_10_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_10_post_attention_layernorm_output_0, %model_layers_10_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.10/mlp/up_proj/MatMul
    }
    %_model_layers_10_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_10_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_10_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_10_mlp_act_fn_Mul_output_0, %_model_layers_10_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.10/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_10_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_10_mlp_Mul_output_0, %model_layers_10_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.10/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_11_input_layernorm_output_0, %v_43, %v_44, %_model_layers_11_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_10_post_attention_layernorm_output_3, %_model_layers_10_mlp_down_proj_MatMul_output_0, %model_layers_11_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.11/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_11_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_11_input_layernorm_output_0, %model_layers_11_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.11/attn/q_proj/MatMul
      %_model_layers_11_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_11_input_layernorm_output_0, %model_layers_11_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.11/attn/k_proj/MatMul
      %_model_layers_11_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_11_input_layernorm_output_0, %model_layers_11_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.11/attn/v_proj/MatMul
    }
    %_model_layers_11_attn_GroupQueryAttention_output_0, %present_11_key, %present_11_value = "hip.gqa"(%_model_layers_11_attn_q_proj_MatMul_output_0, %_model_layers_11_attn_k_proj_MatMul_output_0, %_model_layers_11_attn_v_proj_MatMul_output_0, %past_key_values_11_key, %past_key_values_11_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.11/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_11_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_11_attn_GroupQueryAttention_output_0, %model_layers_11_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.11/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_11_post_attention_layernorm_output_0, %v_45, %v_46, %_model_layers_11_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_11_input_layernorm_output_3, %_model_layers_11_attn_o_proj_MatMul_output_0, %model_layers_11_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.11/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_11_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_11_post_attention_layernorm_output_0, %model_layers_11_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.11/mlp/gate_proj/MatMul
      %_model_layers_11_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_11_post_attention_layernorm_output_0, %model_layers_11_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.11/mlp/up_proj/MatMul
    }
    %_model_layers_11_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_11_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_11_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_11_mlp_act_fn_Mul_output_0, %_model_layers_11_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.11/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_11_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_11_mlp_Mul_output_0, %model_layers_11_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.11/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_12_input_layernorm_output_0, %v_47, %v_48, %_model_layers_12_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_11_post_attention_layernorm_output_3, %_model_layers_11_mlp_down_proj_MatMul_output_0, %model_layers_12_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.12/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_12_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_12_input_layernorm_output_0, %model_layers_12_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.12/attn/q_proj/MatMul
      %_model_layers_12_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_12_input_layernorm_output_0, %model_layers_12_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.12/attn/k_proj/MatMul
      %_model_layers_12_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_12_input_layernorm_output_0, %model_layers_12_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.12/attn/v_proj/MatMul
    }
    %_model_layers_12_attn_GroupQueryAttention_output_0, %present_12_key, %present_12_value = "hip.gqa"(%_model_layers_12_attn_q_proj_MatMul_output_0, %_model_layers_12_attn_k_proj_MatMul_output_0, %_model_layers_12_attn_v_proj_MatMul_output_0, %past_key_values_12_key, %past_key_values_12_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.12/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_12_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_12_attn_GroupQueryAttention_output_0, %model_layers_12_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.12/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_12_post_attention_layernorm_output_0, %v_49, %v_50, %_model_layers_12_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_12_input_layernorm_output_3, %_model_layers_12_attn_o_proj_MatMul_output_0, %model_layers_12_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.12/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_12_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_12_post_attention_layernorm_output_0, %model_layers_12_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.12/mlp/gate_proj/MatMul
      %_model_layers_12_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_12_post_attention_layernorm_output_0, %model_layers_12_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.12/mlp/up_proj/MatMul
    }
    %_model_layers_12_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_12_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_12_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_12_mlp_act_fn_Mul_output_0, %_model_layers_12_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.12/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_12_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_12_mlp_Mul_output_0, %model_layers_12_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.12/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_13_input_layernorm_output_0, %v_51, %v_52, %_model_layers_13_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_12_post_attention_layernorm_output_3, %_model_layers_12_mlp_down_proj_MatMul_output_0, %model_layers_13_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.13/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_13_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_13_input_layernorm_output_0, %model_layers_13_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.13/attn/q_proj/MatMul
      %_model_layers_13_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_13_input_layernorm_output_0, %model_layers_13_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.13/attn/k_proj/MatMul
      %_model_layers_13_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_13_input_layernorm_output_0, %model_layers_13_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.13/attn/v_proj/MatMul
    }
    %_model_layers_13_attn_GroupQueryAttention_output_0, %present_13_key, %present_13_value = "hip.gqa"(%_model_layers_13_attn_q_proj_MatMul_output_0, %_model_layers_13_attn_k_proj_MatMul_output_0, %_model_layers_13_attn_v_proj_MatMul_output_0, %past_key_values_13_key, %past_key_values_13_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.13/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_13_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_13_attn_GroupQueryAttention_output_0, %model_layers_13_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.13/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_13_post_attention_layernorm_output_0, %v_53, %v_54, %_model_layers_13_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_13_input_layernorm_output_3, %_model_layers_13_attn_o_proj_MatMul_output_0, %model_layers_13_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.13/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_13_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_13_post_attention_layernorm_output_0, %model_layers_13_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.13/mlp/gate_proj/MatMul
      %_model_layers_13_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_13_post_attention_layernorm_output_0, %model_layers_13_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.13/mlp/up_proj/MatMul
    }
    %_model_layers_13_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_13_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_13_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_13_mlp_act_fn_Mul_output_0, %_model_layers_13_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.13/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_13_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_13_mlp_Mul_output_0, %model_layers_13_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.13/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_14_input_layernorm_output_0, %v_55, %v_56, %_model_layers_14_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_13_post_attention_layernorm_output_3, %_model_layers_13_mlp_down_proj_MatMul_output_0, %model_layers_14_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.14/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_14_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_14_input_layernorm_output_0, %model_layers_14_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.14/attn/q_proj/MatMul
      %_model_layers_14_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_14_input_layernorm_output_0, %model_layers_14_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.14/attn/k_proj/MatMul
      %_model_layers_14_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_14_input_layernorm_output_0, %model_layers_14_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.14/attn/v_proj/MatMul
    }
    %_model_layers_14_attn_GroupQueryAttention_output_0, %present_14_key, %present_14_value = "hip.gqa"(%_model_layers_14_attn_q_proj_MatMul_output_0, %_model_layers_14_attn_k_proj_MatMul_output_0, %_model_layers_14_attn_v_proj_MatMul_output_0, %past_key_values_14_key, %past_key_values_14_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.14/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_14_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_14_attn_GroupQueryAttention_output_0, %model_layers_14_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.14/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_14_post_attention_layernorm_output_0, %v_57, %v_58, %_model_layers_14_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_14_input_layernorm_output_3, %_model_layers_14_attn_o_proj_MatMul_output_0, %model_layers_14_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.14/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_14_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_14_post_attention_layernorm_output_0, %model_layers_14_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.14/mlp/gate_proj/MatMul
      %_model_layers_14_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_14_post_attention_layernorm_output_0, %model_layers_14_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.14/mlp/up_proj/MatMul
    }
    %_model_layers_14_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_14_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_14_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_14_mlp_act_fn_Mul_output_0, %_model_layers_14_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.14/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_14_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_14_mlp_Mul_output_0, %model_layers_14_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.14/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_15_input_layernorm_output_0, %v_59, %v_60, %_model_layers_15_input_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_14_post_attention_layernorm_output_3, %_model_layers_14_mlp_down_proj_MatMul_output_0, %model_layers_15_input_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.15/input_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_15_attn_q_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_15_input_layernorm_output_0, %model_layers_15_attn_q_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.15/attn/q_proj/MatMul
      %_model_layers_15_attn_k_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_15_input_layernorm_output_0, %model_layers_15_attn_k_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.15/attn/k_proj/MatMul
      %_model_layers_15_attn_v_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_15_input_layernorm_output_0, %model_layers_15_attn_v_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x512xf32, 1>) -> memref<?x?x512xf32, 1>  // /model/layers.15/attn/v_proj/MatMul
    }
    %_model_layers_15_attn_GroupQueryAttention_output_0, %present_15_key, %present_15_value = "hip.gqa"(%_model_layers_15_attn_q_proj_MatMul_output_0, %_model_layers_15_attn_k_proj_MatMul_output_0, %_model_layers_15_attn_v_proj_MatMul_output_0, %past_key_values_15_key, %past_key_values_15_value, %_model_attn_mask_reformat_attn_mask_subgraph_Sub_Cast_output_0, %_model_attn_mask_reformat_attn_mask_subgraph_Gather_Cast_output_0, %cos_cache, %sin_cache) {num_heads = 32 : i64, kv_num_heads = 8 : i64, scale = 0.125 : f32, local_window_size = -1 : i64, softcap = 0.0 : f32, do_rotary = 1 : i64, rotary_interleaved = 0 : i64} : (memref<?x?x2048xf32, 1>, memref<?x?x512xf32, 1>, memref<?x?x512xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x1xi32, 1>, memref<i32, 1>, memref<131072x32xf32, 1>, memref<131072x32xf32, 1>) -> memref<?x?x2048xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>  // /model/layers.15/attn/GroupQueryAttention
    hip.hipblaslt.graph {
      %_model_layers_15_attn_o_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_15_attn_GroupQueryAttention_output_0, %model_layers_15_attn_o_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.15/attn/o_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_15_post_attention_layernorm_output_0, %v_61, %v_62, %_model_layers_15_post_attention_layernorm_output_3 = "hip.miopen.skip_rms_norm"(%_model_layers_15_input_layernorm_output_3, %_model_layers_15_attn_o_proj_MatMul_output_0, %model_layers_15_post_attention_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>, tensor<*xf32>, tensor<*xf32>, memref<?x?x2048xf32, 1>  // /model/layers.15/post_attention_layernorm/SkipLayerNorm
    }
    hip.hipblaslt.graph {
      %_model_layers_15_mlp_gate_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_15_post_attention_layernorm_output_0, %model_layers_15_mlp_gate_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.15/mlp/gate_proj/MatMul
      %_model_layers_15_mlp_up_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_15_post_attention_layernorm_output_0, %model_layers_15_mlp_up_proj_MatMul_weight) : (memref<?x?x2048xf32, 1>, memref<2048x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.15/mlp/up_proj/MatMul
    }
    %_model_layers_15_mlp_act_fn_Mul_output_0 = "hip.silu"(%_model_layers_15_mlp_gate_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>
    hip.miopen.graph {
      %_model_layers_15_mlp_Mul_output_0 = "hip.miopen.mul"(%_model_layers_15_mlp_act_fn_Mul_output_0, %_model_layers_15_mlp_up_proj_MatMul_output_0) : (memref<?x?x8192xf32, 1>, memref<?x?x8192xf32, 1>) -> memref<?x?x8192xf32, 1>  // /model/layers.15/mlp/Mul
    }
    hip.hipblaslt.graph {
      %_model_layers_15_mlp_down_proj_MatMul_output_0 = "hip.hipblaslt.matmul"(%_model_layers_15_mlp_Mul_output_0, %model_layers_15_mlp_down_proj_MatMul_weight) : (memref<?x?x8192xf32, 1>, memref<8192x2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.15/mlp/down_proj/MatMul
    }
    hip.miopen.graph {
      %_model_layers_16_final_norm_layernorm_output_0 = "hip.miopen.skip_rms_norm"(%_model_layers_15_post_attention_layernorm_output_3, %_model_layers_15_mlp_down_proj_MatMul_output_0, %model_layers_16_final_norm_layernorm_weight) {epsilon = 9.9999997e-06 : f32} : (memref<?x?x2048xf32, 1>, memref<?x?x2048xf32, 1>, memref<2048xf32, 1>) -> memref<?x?x2048xf32, 1>  // /model/layers.16/final_norm_layernorm/SkipLayerNorm
    }
    %_lm_head_Transpose_output_0 = "hip.transpose"(%model_embed_tokens_weight) {perm = [1, 0]} : (memref<128256x2048xf32, 1>) -> memref<2048x128256xf32, 1>  // /lm_head/Transpose
    hip.hipblaslt.graph {
      %logits = "hip.hipblaslt.matmul"(%_model_layers_16_final_norm_layernorm_output_0, %_lm_head_Transpose_output_0) : (memref<?x?x2048xf32, 1>, memref<2048x128256xf32, 1>) -> memref<?x?x128256xf32, 1>  // /lm_head/MatMul
    }

    hip.destroy_handle(%handle) : !hip.handle

    return %logits, %present_0_key, %present_0_value, %present_1_key, %present_1_value, %present_2_key, %present_2_value, %present_3_key, %present_3_value, %present_4_key, %present_4_value, %present_5_key, %present_5_value, %present_6_key, %present_6_value, %present_7_key, %present_7_value, %present_8_key, %present_8_value, %present_9_key, %present_9_value, %present_10_key, %present_10_value, %present_11_key, %present_11_value, %present_12_key, %present_12_value, %present_13_key, %present_13_value, %present_14_key, %present_14_value, %present_15_key, %present_15_value : memref<?x?x128256xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>, memref<?x8x?x64xf32, 1>
  }
}
