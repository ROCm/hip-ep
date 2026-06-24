// === ck_dsl lower_hip prologue (auto-generated) ===
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>
#include <stdint.h>

using fp16 = _Float16;
#if defined(__BF16__) || defined(__bfloat16)
using bf16 = __bf16;
#else
using bf16 = __bf16;
#endif
using fp8e4m3 = signed char; // raw byte storage; converted via amdgcn intrinsics
using bf8e5m2 = signed char; // raw byte storage; e5m2 variant (different cvt intrinsic)

// AMDGPU vector typedefs via Clang's ext_vector_type. Names match the
// fNxM convention used throughout the handlers below.
#define _CKDSL_VEC(elem_t, name, n) \
 using name##n = elem_t __attribute__((ext_vector_type(n)))
_CKDSL_VEC(fp16, f16x, 1); _CKDSL_VEC(fp16, f16x, 2); _CKDSL_VEC(fp16, f16x, 4);
_CKDSL_VEC(fp16, f16x, 8); _CKDSL_VEC(fp16, f16x, 16);
_CKDSL_VEC(bf16, bf16x, 1); _CKDSL_VEC(bf16, bf16x, 2); _CKDSL_VEC(bf16, bf16x, 4);
_CKDSL_VEC(bf16, bf16x, 8); _CKDSL_VEC(bf16, bf16x, 16);
_CKDSL_VEC(float, f32x, 1); _CKDSL_VEC(float, f32x, 2); _CKDSL_VEC(float, f32x, 4);
_CKDSL_VEC(float, f32x, 8); _CKDSL_VEC(float, f32x, 16);
_CKDSL_VEC(int, i32x, 1); _CKDSL_VEC(int, i32x, 2); _CKDSL_VEC(int, i32x, 3);
_CKDSL_VEC(int, i32x, 4); _CKDSL_VEC(int, i32x, 8);
_CKDSL_VEC(int16_t, i16x, 1); _CKDSL_VEC(int16_t, i16x, 2);
_CKDSL_VEC(int16_t, i16x, 4); _CKDSL_VEC(int16_t, i16x, 8);
_CKDSL_VEC(int8_t, i8x, 4); _CKDSL_VEC(int8_t, i8x, 8); _CKDSL_VEC(int8_t, i8x, 16);
_CKDSL_VEC(bool, boolx, 2); _CKDSL_VEC(bool, boolx, 4); _CKDSL_VEC(bool, boolx, 8);
_CKDSL_VEC(bool, boolx, 16);
#undef _CKDSL_VEC

// Buffer-resource descriptor opaque type. ``__builtin_amdgcn_make_buffer_rsrc``
// returns this; the ``_ptr_`` family of buffer-load / store builtins takes
// it as the first argument. Although the IR uses ``<4 x i32>`` to model the
// 128-bit descriptor, at the C++ level we use the opaque builtin type so
// type checking lines up with the intrinsics.
using rsrc_t = __amdgpu_buffer_rsrc_t;

// LLVM intrinsics that clang 20 does NOT expose as ``__builtin_amdgcn_*``
// builtins (or whose builtins reject the size values we need). We declare
// them as ``__device__ extern "C"`` with an ``__asm`` mangling that names
// the LLVM intrinsic directly; clang lowers the call through the AMDGPU
// backend the same way it would the missing builtin. The ``__device__``
// attribute is required so HIP allows the call from a ``__global__``
// kernel context.
typedef short i16x4_raw __attribute__((ext_vector_type(4)));
__device__ extern "C" i16x4_raw _llvm_amdgcn_ds_read_tr16_b64(
 const __attribute__((address_space(3))) void*)
 __asm("llvm.amdgcn.ds.read.tr16.b64");
typedef short i16x8_raw __attribute__((ext_vector_type(8)));
__device__ extern "C" i16x8_raw _llvm_amdgcn_ds_read_tr16_b128(
 const __attribute__((address_space(3))) void*)
 __asm("llvm.amdgcn.ds.read.tr16.b128");
// ``__builtin_amdgcn_raw_ptr_buffer_load_lds`` restricts the size arg to
// {1, 2, 4} bytes; the LLVM intrinsic itself accepts {1, 2, 4, 12, 16},
// which is what async-DMA pipelines (compv4 / split-KV attention) need.
// Calling the intrinsic directly bypasses the builtin's validation.
__device__ extern "C" void _llvm_amdgcn_raw_ptr_buffer_load_lds(
 __amdgpu_buffer_rsrc_t,
 __attribute__((address_space(3))) void*,
 int /*size_bytes*/,
 int /*voffset*/,
 int /*soffset*/,
 int /*offset_imm*/,
 int /*aux_imm*/)
 __asm("llvm.amdgcn.raw.ptr.buffer.load.lds");

extern "C" __global__ __launch_bounds__(256)
void ck_dsl_rmsnorm2d_fwd_f16_N4096_b256_v4(fp16* __restrict__ X, fp16* __restrict__ Gamma, fp16* __restrict__ Y, int M, int N, float eps)
{
    __shared__ float lds_red5_storage[256];
 int tid1 = (int)threadIdx.x;
 int bid2 = (int)blockIdx.x;
 int c3 = 0;
 int c4 = 0;
 float c6 = 0.0f;
 int c7 = 4;
 int c8 = 0;
 int mul9 = c8 * c7;
 int mul10 = tid1 * c7;
 int add11 = mul9 + mul10;
 int c12 = 0;
 int add13 = bid2 + c12;
 int add14 = c3 + add11;
 int c15 = 4096;
 int mul16 = add13 * c15;
 int add17 = mul16 + add14;
 f16x4 gv418 = *reinterpret_cast<const f16x4*>(X + add17);
 fp16 e19 = gv418[0];
 float f3220 = (float)e19;
 fp16 e21 = gv418[1];
 float f3222 = (float)e21;
 fp16 e23 = gv418[2];
 float f3224 = (float)e23;
 fp16 e25 = gv418[3];
 float f3226 = (float)e25;
 float fmul27 = f3220 * f3220;
 float fmul28 = f3222 * f3222;
 float fmul29 = f3224 * f3224;
 float fmul30 = f3226 * f3226;
 float fadd31 = fmul27 + fmul28;
 float fadd32 = fmul29 + fmul30;
 float fadd33 = fadd31 + fadd32;
 float fadd34 = c6 + fadd33;
 int c35 = 256;
 int mul36 = c35 * c7;
 int mul37 = tid1 * c7;
 int add38 = mul36 + mul37;
 int c39 = 0;
 int add40 = bid2 + c39;
 int add41 = c3 + add38;
 int c42 = 4096;
 int mul43 = add40 * c42;
 int add44 = mul43 + add41;
 f16x4 gv445 = *reinterpret_cast<const f16x4*>(X + add44);
 fp16 e46 = gv445[0];
 float f3247 = (float)e46;
 fp16 e48 = gv445[1];
 float f3249 = (float)e48;
 fp16 e50 = gv445[2];
 float f3251 = (float)e50;
 fp16 e52 = gv445[3];
 float f3253 = (float)e52;
 float fmul54 = f3247 * f3247;
 float fmul55 = f3249 * f3249;
 float fmul56 = f3251 * f3251;
 float fmul57 = f3253 * f3253;
 float fadd58 = fmul54 + fmul55;
 float fadd59 = fmul56 + fmul57;
 float fadd60 = fadd58 + fadd59;
 float fadd61 = fadd34 + fadd60;
 int c62 = 512;
 int mul63 = c62 * c7;
 int mul64 = tid1 * c7;
 int add65 = mul63 + mul64;
 int c66 = 0;
 int add67 = bid2 + c66;
 int add68 = c3 + add65;
 int c69 = 4096;
 int mul70 = add67 * c69;
 int add71 = mul70 + add68;
 f16x4 gv472 = *reinterpret_cast<const f16x4*>(X + add71);
 fp16 e73 = gv472[0];
 float f3274 = (float)e73;
 fp16 e75 = gv472[1];
 float f3276 = (float)e75;
 fp16 e77 = gv472[2];
 float f3278 = (float)e77;
 fp16 e79 = gv472[3];
 float f3280 = (float)e79;
 float fmul81 = f3274 * f3274;
 float fmul82 = f3276 * f3276;
 float fmul83 = f3278 * f3278;
 float fmul84 = f3280 * f3280;
 float fadd85 = fmul81 + fmul82;
 float fadd86 = fmul83 + fmul84;
 float fadd87 = fadd85 + fadd86;
 float fadd88 = fadd61 + fadd87;
 int c89 = 768;
 int mul90 = c89 * c7;
 int mul91 = tid1 * c7;
 int add92 = mul90 + mul91;
 int c93 = 0;
 int add94 = bid2 + c93;
 int add95 = c3 + add92;
 int c96 = 4096;
 int mul97 = add94 * c96;
 int add98 = mul97 + add95;
 f16x4 gv499 = *reinterpret_cast<const f16x4*>(X + add98);
 fp16 e100 = gv499[0];
 float f32101 = (float)e100;
 fp16 e102 = gv499[1];
 float f32103 = (float)e102;
 fp16 e104 = gv499[2];
 float f32105 = (float)e104;
 fp16 e106 = gv499[3];
 float f32107 = (float)e106;
 float fmul108 = f32101 * f32101;
 float fmul109 = f32103 * f32103;
 float fmul110 = f32105 * f32105;
 float fmul111 = f32107 * f32107;
 float fadd112 = fmul108 + fmul109;
 float fadd113 = fmul110 + fmul111;
 float fadd114 = fadd112 + fadd113;
 float fadd115 = fadd88 + fadd114;
 int bc116; __builtin_memcpy(&bc116, &fadd115, sizeof(int));
 int sw117 = __builtin_amdgcn_ds_swizzle(bc116, 1055);
 float bc118; __builtin_memcpy(&bc118, &sw117, sizeof(float));
 float fadd119 = fadd115 + bc118;
 int bc120; __builtin_memcpy(&bc120, &fadd119, sizeof(int));
 int sw121 = __builtin_amdgcn_ds_swizzle(bc120, 2079);
 float bc122; __builtin_memcpy(&bc122, &sw121, sizeof(float));
 float fadd123 = fadd119 + bc122;
 int bc124; __builtin_memcpy(&bc124, &fadd123, sizeof(int));
 int sw125 = __builtin_amdgcn_ds_swizzle(bc124, 4127);
 float bc126; __builtin_memcpy(&bc126, &sw125, sizeof(float));
 float fadd127 = fadd123 + bc126;
 int bc128; __builtin_memcpy(&bc128, &fadd127, sizeof(int));
 int sw129 = __builtin_amdgcn_ds_swizzle(bc128, 8223);
 float bc130; __builtin_memcpy(&bc130, &sw129, sizeof(float));
 float fadd131 = fadd127 + bc130;
 int bc132; __builtin_memcpy(&bc132, &fadd131, sizeof(int));
 int sw133 = __builtin_amdgcn_ds_swizzle(bc132, 16415);
 float bc134; __builtin_memcpy(&bc134, &sw133, sizeof(float));
 float fadd135 = fadd131 + bc134;
 int c136 = 32;
 int mod137 = tid1 % c136;
 int div138 = tid1 / c136;
 int c139 = 0;
 bool eq140 = mod137 == c139;
 if(eq140) {
  *reinterpret_cast<f32x1*>(&lds_red5_storage[div138]) = fadd135;
 }
 __syncthreads();
 int c141 = 0;
 f32x1 av1f32142 = *reinterpret_cast<const f32x1*>(&lds_red5_storage[c141]);
 float e143 = av1f32142[0];
 int c144 = 1;
 f32x1 av1f32145 = *reinterpret_cast<const f32x1*>(&lds_red5_storage[c144]);
 float e146 = av1f32145[0];
 int c147 = 2;
 f32x1 av1f32148 = *reinterpret_cast<const f32x1*>(&lds_red5_storage[c147]);
 float e149 = av1f32148[0];
 int c150 = 3;
 f32x1 av1f32151 = *reinterpret_cast<const f32x1*>(&lds_red5_storage[c150]);
 float e152 = av1f32151[0];
 int c153 = 4;
 f32x1 av1f32154 = *reinterpret_cast<const f32x1*>(&lds_red5_storage[c153]);
 float e155 = av1f32154[0];
 int c156 = 5;
 f32x1 av1f32157 = *reinterpret_cast<const f32x1*>(&lds_red5_storage[c156]);
 float e158 = av1f32157[0];
 int c159 = 6;
 f32x1 av1f32160 = *reinterpret_cast<const f32x1*>(&lds_red5_storage[c159]);
 float e161 = av1f32160[0];
 int c162 = 7;
 f32x1 av1f32163 = *reinterpret_cast<const f32x1*>(&lds_red5_storage[c162]);
 float e164 = av1f32163[0];
 float fadd165 = e143 + e146;
 float fadd166 = e149 + e152;
 float fadd167 = e155 + e158;
 float fadd168 = e161 + e164;
 float fadd169 = fadd165 + fadd166;
 float fadd170 = fadd167 + fadd168;
 float fadd171 = fadd169 + fadd170;
 float c172 = 4096.0f;
 float rcp173 = __builtin_amdgcn_rcpf(c172);
 float fmul174 = fadd171 * rcp173;
 float fadd175 = fmul174 + eps;
 float rsq176 = __builtin_amdgcn_rsqf(fadd175);
 int c177 = 4;
 int c178 = 0;
 int mul179 = c178 * c177;
 int mul180 = tid1 * c177;
 int add181 = mul179 + mul180;
 f16x4 gv4182 = *reinterpret_cast<const f16x4*>(Gamma + add181);
 fp16 e183 = gv4182[0];
 float f32184 = (float)e183;
 fp16 e185 = gv4182[1];
 float f32186 = (float)e185;
 fp16 e187 = gv4182[2];
 float f32188 = (float)e187;
 fp16 e189 = gv4182[3];
 float f32190 = (float)e189;
 float fmul191 = rsq176 * f32184;
 float fmul192 = f3220 * fmul191;
 float fmul193 = rsq176 * f32186;
 float fmul194 = f3222 * fmul193;
 float fmul195 = rsq176 * f32188;
 float fmul196 = f3224 * fmul195;
 float fmul197 = rsq176 * f32190;
 float fmul198 = f3226 * fmul197;
 int c199 = 0;
 fp16 cast200 = (fp16)fmul192;
 fp16 cast201 = (fp16)fmul194;
 fp16 cast202 = (fp16)fmul196;
 fp16 cast203 = (fp16)fmul198;
 f16x4 vp204;
 vp204[0] = cast200;
 vp204[1] = cast201;
 vp204[2] = cast202;
 vp204[3] = cast203;
 int add205 = bid2 + c199;
 int add206 = c4 + add181;
 int c207 = 4096;
 int mul208 = add205 * c207;
 int add209 = mul208 + add206;
 *reinterpret_cast<f16x4*>(Y + add209) = vp204;
 int c210 = 256;
 int mul211 = c210 * c177;
 int mul212 = tid1 * c177;
 int add213 = mul211 + mul212;
 f16x4 gv4214 = *reinterpret_cast<const f16x4*>(Gamma + add213);
 fp16 e215 = gv4214[0];
 float f32216 = (float)e215;
 fp16 e217 = gv4214[1];
 float f32218 = (float)e217;
 fp16 e219 = gv4214[2];
 float f32220 = (float)e219;
 fp16 e221 = gv4214[3];
 float f32222 = (float)e221;
 float fmul223 = rsq176 * f32216;
 float fmul224 = f3247 * fmul223;
 float fmul225 = rsq176 * f32218;
 float fmul226 = f3249 * fmul225;
 float fmul227 = rsq176 * f32220;
 float fmul228 = f3251 * fmul227;
 float fmul229 = rsq176 * f32222;
 float fmul230 = f3253 * fmul229;
 int c231 = 0;
 fp16 cast232 = (fp16)fmul224;
 fp16 cast233 = (fp16)fmul226;
 fp16 cast234 = (fp16)fmul228;
 fp16 cast235 = (fp16)fmul230;
 f16x4 vp236;
 vp236[0] = cast232;
 vp236[1] = cast233;
 vp236[2] = cast234;
 vp236[3] = cast235;
 int add237 = bid2 + c231;
 int add238 = c4 + add213;
 int c239 = 4096;
 int mul240 = add237 * c239;
 int add241 = mul240 + add238;
 *reinterpret_cast<f16x4*>(Y + add241) = vp236;
 int c242 = 512;
 int mul243 = c242 * c177;
 int mul244 = tid1 * c177;
 int add245 = mul243 + mul244;
 f16x4 gv4246 = *reinterpret_cast<const f16x4*>(Gamma + add245);
 fp16 e247 = gv4246[0];
 float f32248 = (float)e247;
 fp16 e249 = gv4246[1];
 float f32250 = (float)e249;
 fp16 e251 = gv4246[2];
 float f32252 = (float)e251;
 fp16 e253 = gv4246[3];
 float f32254 = (float)e253;
 float fmul255 = rsq176 * f32248;
 float fmul256 = f3274 * fmul255;
 float fmul257 = rsq176 * f32250;
 float fmul258 = f3276 * fmul257;
 float fmul259 = rsq176 * f32252;
 float fmul260 = f3278 * fmul259;
 float fmul261 = rsq176 * f32254;
 float fmul262 = f3280 * fmul261;
 int c263 = 0;
 fp16 cast264 = (fp16)fmul256;
 fp16 cast265 = (fp16)fmul258;
 fp16 cast266 = (fp16)fmul260;
 fp16 cast267 = (fp16)fmul262;
 f16x4 vp268;
 vp268[0] = cast264;
 vp268[1] = cast265;
 vp268[2] = cast266;
 vp268[3] = cast267;
 int add269 = bid2 + c263;
 int add270 = c4 + add245;
 int c271 = 4096;
 int mul272 = add269 * c271;
 int add273 = mul272 + add270;
 *reinterpret_cast<f16x4*>(Y + add273) = vp268;
 int c274 = 768;
 int mul275 = c274 * c177;
 int mul276 = tid1 * c177;
 int add277 = mul275 + mul276;
 f16x4 gv4278 = *reinterpret_cast<const f16x4*>(Gamma + add277);
 fp16 e279 = gv4278[0];
 float f32280 = (float)e279;
 fp16 e281 = gv4278[1];
 float f32282 = (float)e281;
 fp16 e283 = gv4278[2];
 float f32284 = (float)e283;
 fp16 e285 = gv4278[3];
 float f32286 = (float)e285;
 float fmul287 = rsq176 * f32280;
 float fmul288 = f32101 * fmul287;
 float fmul289 = rsq176 * f32282;
 float fmul290 = f32103 * fmul289;
 float fmul291 = rsq176 * f32284;
 float fmul292 = f32105 * fmul291;
 float fmul293 = rsq176 * f32286;
 float fmul294 = f32107 * fmul293;
 int c295 = 0;
 fp16 cast296 = (fp16)fmul288;
 fp16 cast297 = (fp16)fmul290;
 fp16 cast298 = (fp16)fmul292;
 fp16 cast299 = (fp16)fmul294;
 f16x4 vp300;
 vp300[0] = cast296;
 vp300[1] = cast297;
 vp300[2] = cast298;
 vp300[3] = cast299;
 int add301 = bid2 + c295;
 int add302 = c4 + add277;
 int c303 = 4096;
 int mul304 = add301 * c303;
 int add305 = mul304 + add302;
 *reinterpret_cast<f16x4*>(Y + add305) = vp300;
}
