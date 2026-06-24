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
void ck_dsl_add_rmsnorm2d_bf16_f16_N4096_b256_v4_sr(fp16* __restrict__ A, fp16* __restrict__ B, fp16* __restrict__ Gamma, fp16* __restrict__ X, fp16* __restrict__ Y, int M, int N, float eps)
{
    __shared__ float lds_red7_storage[256];
 int tid1 = (int)threadIdx.x;
 int bid2 = (int)blockIdx.x;
 int c3 = 0;
 int c4 = 0;
 int c5 = 0;
 int c6 = 0;
 float c8 = 0.0f;
 int c9 = 4;
 int c10 = 0;
 int mul11 = c10 * c9;
 int mul12 = tid1 * c9;
 int add13 = mul11 + mul12;
 int c14 = 0;
 int add15 = bid2 + c14;
 int add16 = c3 + add13;
 int c17 = 4096;
 int mul18 = add15 * c17;
 int add19 = mul18 + add16;
 f16x4 gv420 = *reinterpret_cast<const f16x4*>(A + add19);
 fp16 e21 = gv420[0];
 float f3222 = (float)e21;
 fp16 e23 = gv420[1];
 float f3224 = (float)e23;
 fp16 e25 = gv420[2];
 float f3226 = (float)e25;
 fp16 e27 = gv420[3];
 float f3228 = (float)e27;
 int c29 = 0;
 int add30 = bid2 + c29;
 int add31 = c4 + add13;
 int c32 = 4096;
 int mul33 = add30 * c32;
 int add34 = mul33 + add31;
 f16x4 gv435 = *reinterpret_cast<const f16x4*>(B + add34);
 fp16 e36 = gv435[0];
 float f3237 = (float)e36;
 fp16 e38 = gv435[1];
 float f3239 = (float)e38;
 fp16 e40 = gv435[2];
 float f3241 = (float)e40;
 fp16 e42 = gv435[3];
 float f3243 = (float)e42;
 float fadd44 = f3222 + f3237;
 float fmul45 = fadd44 * fadd44;
 float fadd46 = f3224 + f3239;
 float fmul47 = fadd46 * fadd46;
 float fadd48 = f3226 + f3241;
 float fmul49 = fadd48 * fadd48;
 float fadd50 = f3228 + f3243;
 float fmul51 = fadd50 * fadd50;
 float fadd52 = fmul45 + fmul47;
 float fadd53 = fmul49 + fmul51;
 float fadd54 = fadd52 + fadd53;
 float fadd55 = c8 + fadd54;
 int c56 = 0;
 fp16 cast57 = (fp16)fadd44;
 fp16 cast58 = (fp16)fadd46;
 fp16 cast59 = (fp16)fadd48;
 fp16 cast60 = (fp16)fadd50;
 f16x4 vp61;
 vp61[0] = cast57;
 vp61[1] = cast58;
 vp61[2] = cast59;
 vp61[3] = cast60;
 int add62 = bid2 + c56;
 int add63 = c6 + add13;
 int c64 = 4096;
 int mul65 = add62 * c64;
 int add66 = mul65 + add63;
 *reinterpret_cast<f16x4*>(X + add66) = vp61;
 int c67 = 256;
 int mul68 = c67 * c9;
 int mul69 = tid1 * c9;
 int add70 = mul68 + mul69;
 int c71 = 0;
 int add72 = bid2 + c71;
 int add73 = c3 + add70;
 int c74 = 4096;
 int mul75 = add72 * c74;
 int add76 = mul75 + add73;
 f16x4 gv477 = *reinterpret_cast<const f16x4*>(A + add76);
 fp16 e78 = gv477[0];
 float f3279 = (float)e78;
 fp16 e80 = gv477[1];
 float f3281 = (float)e80;
 fp16 e82 = gv477[2];
 float f3283 = (float)e82;
 fp16 e84 = gv477[3];
 float f3285 = (float)e84;
 int c86 = 0;
 int add87 = bid2 + c86;
 int add88 = c4 + add70;
 int c89 = 4096;
 int mul90 = add87 * c89;
 int add91 = mul90 + add88;
 f16x4 gv492 = *reinterpret_cast<const f16x4*>(B + add91);
 fp16 e93 = gv492[0];
 float f3294 = (float)e93;
 fp16 e95 = gv492[1];
 float f3296 = (float)e95;
 fp16 e97 = gv492[2];
 float f3298 = (float)e97;
 fp16 e99 = gv492[3];
 float f32100 = (float)e99;
 float fadd101 = f3279 + f3294;
 float fmul102 = fadd101 * fadd101;
 float fadd103 = f3281 + f3296;
 float fmul104 = fadd103 * fadd103;
 float fadd105 = f3283 + f3298;
 float fmul106 = fadd105 * fadd105;
 float fadd107 = f3285 + f32100;
 float fmul108 = fadd107 * fadd107;
 float fadd109 = fmul102 + fmul104;
 float fadd110 = fmul106 + fmul108;
 float fadd111 = fadd109 + fadd110;
 float fadd112 = fadd55 + fadd111;
 int c113 = 0;
 fp16 cast114 = (fp16)fadd101;
 fp16 cast115 = (fp16)fadd103;
 fp16 cast116 = (fp16)fadd105;
 fp16 cast117 = (fp16)fadd107;
 f16x4 vp118;
 vp118[0] = cast114;
 vp118[1] = cast115;
 vp118[2] = cast116;
 vp118[3] = cast117;
 int add119 = bid2 + c113;
 int add120 = c6 + add70;
 int c121 = 4096;
 int mul122 = add119 * c121;
 int add123 = mul122 + add120;
 *reinterpret_cast<f16x4*>(X + add123) = vp118;
 int c124 = 512;
 int mul125 = c124 * c9;
 int mul126 = tid1 * c9;
 int add127 = mul125 + mul126;
 int c128 = 0;
 int add129 = bid2 + c128;
 int add130 = c3 + add127;
 int c131 = 4096;
 int mul132 = add129 * c131;
 int add133 = mul132 + add130;
 f16x4 gv4134 = *reinterpret_cast<const f16x4*>(A + add133);
 fp16 e135 = gv4134[0];
 float f32136 = (float)e135;
 fp16 e137 = gv4134[1];
 float f32138 = (float)e137;
 fp16 e139 = gv4134[2];
 float f32140 = (float)e139;
 fp16 e141 = gv4134[3];
 float f32142 = (float)e141;
 int c143 = 0;
 int add144 = bid2 + c143;
 int add145 = c4 + add127;
 int c146 = 4096;
 int mul147 = add144 * c146;
 int add148 = mul147 + add145;
 f16x4 gv4149 = *reinterpret_cast<const f16x4*>(B + add148);
 fp16 e150 = gv4149[0];
 float f32151 = (float)e150;
 fp16 e152 = gv4149[1];
 float f32153 = (float)e152;
 fp16 e154 = gv4149[2];
 float f32155 = (float)e154;
 fp16 e156 = gv4149[3];
 float f32157 = (float)e156;
 float fadd158 = f32136 + f32151;
 float fmul159 = fadd158 * fadd158;
 float fadd160 = f32138 + f32153;
 float fmul161 = fadd160 * fadd160;
 float fadd162 = f32140 + f32155;
 float fmul163 = fadd162 * fadd162;
 float fadd164 = f32142 + f32157;
 float fmul165 = fadd164 * fadd164;
 float fadd166 = fmul159 + fmul161;
 float fadd167 = fmul163 + fmul165;
 float fadd168 = fadd166 + fadd167;
 float fadd169 = fadd112 + fadd168;
 int c170 = 0;
 fp16 cast171 = (fp16)fadd158;
 fp16 cast172 = (fp16)fadd160;
 fp16 cast173 = (fp16)fadd162;
 fp16 cast174 = (fp16)fadd164;
 f16x4 vp175;
 vp175[0] = cast171;
 vp175[1] = cast172;
 vp175[2] = cast173;
 vp175[3] = cast174;
 int add176 = bid2 + c170;
 int add177 = c6 + add127;
 int c178 = 4096;
 int mul179 = add176 * c178;
 int add180 = mul179 + add177;
 *reinterpret_cast<f16x4*>(X + add180) = vp175;
 int c181 = 768;
 int mul182 = c181 * c9;
 int mul183 = tid1 * c9;
 int add184 = mul182 + mul183;
 int c185 = 0;
 int add186 = bid2 + c185;
 int add187 = c3 + add184;
 int c188 = 4096;
 int mul189 = add186 * c188;
 int add190 = mul189 + add187;
 f16x4 gv4191 = *reinterpret_cast<const f16x4*>(A + add190);
 fp16 e192 = gv4191[0];
 float f32193 = (float)e192;
 fp16 e194 = gv4191[1];
 float f32195 = (float)e194;
 fp16 e196 = gv4191[2];
 float f32197 = (float)e196;
 fp16 e198 = gv4191[3];
 float f32199 = (float)e198;
 int c200 = 0;
 int add201 = bid2 + c200;
 int add202 = c4 + add184;
 int c203 = 4096;
 int mul204 = add201 * c203;
 int add205 = mul204 + add202;
 f16x4 gv4206 = *reinterpret_cast<const f16x4*>(B + add205);
 fp16 e207 = gv4206[0];
 float f32208 = (float)e207;
 fp16 e209 = gv4206[1];
 float f32210 = (float)e209;
 fp16 e211 = gv4206[2];
 float f32212 = (float)e211;
 fp16 e213 = gv4206[3];
 float f32214 = (float)e213;
 float fadd215 = f32193 + f32208;
 float fmul216 = fadd215 * fadd215;
 float fadd217 = f32195 + f32210;
 float fmul218 = fadd217 * fadd217;
 float fadd219 = f32197 + f32212;
 float fmul220 = fadd219 * fadd219;
 float fadd221 = f32199 + f32214;
 float fmul222 = fadd221 * fadd221;
 float fadd223 = fmul216 + fmul218;
 float fadd224 = fmul220 + fmul222;
 float fadd225 = fadd223 + fadd224;
 float fadd226 = fadd169 + fadd225;
 int c227 = 0;
 fp16 cast228 = (fp16)fadd215;
 fp16 cast229 = (fp16)fadd217;
 fp16 cast230 = (fp16)fadd219;
 fp16 cast231 = (fp16)fadd221;
 f16x4 vp232;
 vp232[0] = cast228;
 vp232[1] = cast229;
 vp232[2] = cast230;
 vp232[3] = cast231;
 int add233 = bid2 + c227;
 int add234 = c6 + add184;
 int c235 = 4096;
 int mul236 = add233 * c235;
 int add237 = mul236 + add234;
 *reinterpret_cast<f16x4*>(X + add237) = vp232;
 int bc238; __builtin_memcpy(&bc238, &fadd226, sizeof(int));
 int sw239 = __builtin_amdgcn_ds_swizzle(bc238, 1055);
 float bc240; __builtin_memcpy(&bc240, &sw239, sizeof(float));
 float fadd241 = fadd226 + bc240;
 int bc242; __builtin_memcpy(&bc242, &fadd241, sizeof(int));
 int sw243 = __builtin_amdgcn_ds_swizzle(bc242, 2079);
 float bc244; __builtin_memcpy(&bc244, &sw243, sizeof(float));
 float fadd245 = fadd241 + bc244;
 int bc246; __builtin_memcpy(&bc246, &fadd245, sizeof(int));
 int sw247 = __builtin_amdgcn_ds_swizzle(bc246, 4127);
 float bc248; __builtin_memcpy(&bc248, &sw247, sizeof(float));
 float fadd249 = fadd245 + bc248;
 int bc250; __builtin_memcpy(&bc250, &fadd249, sizeof(int));
 int sw251 = __builtin_amdgcn_ds_swizzle(bc250, 8223);
 float bc252; __builtin_memcpy(&bc252, &sw251, sizeof(float));
 float fadd253 = fadd249 + bc252;
 int bc254; __builtin_memcpy(&bc254, &fadd253, sizeof(int));
 int sw255 = __builtin_amdgcn_ds_swizzle(bc254, 16415);
 float bc256; __builtin_memcpy(&bc256, &sw255, sizeof(float));
 float fadd257 = fadd253 + bc256;
 int c258 = 32;
 int mod259 = tid1 % c258;
 int div260 = tid1 / c258;
 int c261 = 0;
 bool eq262 = mod259 == c261;
 if(eq262) {
  *reinterpret_cast<f32x1*>(&lds_red7_storage[div260]) = fadd257;
 }
 __syncthreads();
 int c263 = 0;
 f32x1 av1f32264 = *reinterpret_cast<const f32x1*>(&lds_red7_storage[c263]);
 float e265 = av1f32264[0];
 int c266 = 1;
 f32x1 av1f32267 = *reinterpret_cast<const f32x1*>(&lds_red7_storage[c266]);
 float e268 = av1f32267[0];
 int c269 = 2;
 f32x1 av1f32270 = *reinterpret_cast<const f32x1*>(&lds_red7_storage[c269]);
 float e271 = av1f32270[0];
 int c272 = 3;
 f32x1 av1f32273 = *reinterpret_cast<const f32x1*>(&lds_red7_storage[c272]);
 float e274 = av1f32273[0];
 int c275 = 4;
 f32x1 av1f32276 = *reinterpret_cast<const f32x1*>(&lds_red7_storage[c275]);
 float e277 = av1f32276[0];
 int c278 = 5;
 f32x1 av1f32279 = *reinterpret_cast<const f32x1*>(&lds_red7_storage[c278]);
 float e280 = av1f32279[0];
 int c281 = 6;
 f32x1 av1f32282 = *reinterpret_cast<const f32x1*>(&lds_red7_storage[c281]);
 float e283 = av1f32282[0];
 int c284 = 7;
 f32x1 av1f32285 = *reinterpret_cast<const f32x1*>(&lds_red7_storage[c284]);
 float e286 = av1f32285[0];
 float fadd287 = e265 + e268;
 float fadd288 = e271 + e274;
 float fadd289 = e277 + e280;
 float fadd290 = e283 + e286;
 float fadd291 = fadd287 + fadd288;
 float fadd292 = fadd289 + fadd290;
 float fadd293 = fadd291 + fadd292;
 float c294 = 4096.0f;
 float rcp295 = __builtin_amdgcn_rcpf(c294);
 float fmul296 = fadd293 * rcp295;
 float fadd297 = fmul296 + eps;
 float rsq298 = __builtin_amdgcn_rsqf(fadd297);
 int c299 = 0;
 int mul300 = c299 * c9;
 int mul301 = tid1 * c9;
 int add302 = mul300 + mul301;
 f16x4 gv4303 = *reinterpret_cast<const f16x4*>(Gamma + add302);
 fp16 e304 = gv4303[0];
 float f32305 = (float)e304;
 fp16 e306 = gv4303[1];
 float f32307 = (float)e306;
 fp16 e308 = gv4303[2];
 float f32309 = (float)e308;
 fp16 e310 = gv4303[3];
 float f32311 = (float)e310;
 float fmul312 = rsq298 * f32305;
 float fmul313 = fadd44 * fmul312;
 float fmul314 = rsq298 * f32307;
 float fmul315 = fadd46 * fmul314;
 float fmul316 = rsq298 * f32309;
 float fmul317 = fadd48 * fmul316;
 float fmul318 = rsq298 * f32311;
 float fmul319 = fadd50 * fmul318;
 int c320 = 0;
 fp16 cast321 = (fp16)fmul313;
 fp16 cast322 = (fp16)fmul315;
 fp16 cast323 = (fp16)fmul317;
 fp16 cast324 = (fp16)fmul319;
 f16x4 vp325;
 vp325[0] = cast321;
 vp325[1] = cast322;
 vp325[2] = cast323;
 vp325[3] = cast324;
 int add326 = bid2 + c320;
 int add327 = c5 + add302;
 int c328 = 4096;
 int mul329 = add326 * c328;
 int add330 = mul329 + add327;
 *reinterpret_cast<f16x4*>(Y + add330) = vp325;
 int c331 = 256;
 int mul332 = c331 * c9;
 int mul333 = tid1 * c9;
 int add334 = mul332 + mul333;
 f16x4 gv4335 = *reinterpret_cast<const f16x4*>(Gamma + add334);
 fp16 e336 = gv4335[0];
 float f32337 = (float)e336;
 fp16 e338 = gv4335[1];
 float f32339 = (float)e338;
 fp16 e340 = gv4335[2];
 float f32341 = (float)e340;
 fp16 e342 = gv4335[3];
 float f32343 = (float)e342;
 float fmul344 = rsq298 * f32337;
 float fmul345 = fadd101 * fmul344;
 float fmul346 = rsq298 * f32339;
 float fmul347 = fadd103 * fmul346;
 float fmul348 = rsq298 * f32341;
 float fmul349 = fadd105 * fmul348;
 float fmul350 = rsq298 * f32343;
 float fmul351 = fadd107 * fmul350;
 int c352 = 0;
 fp16 cast353 = (fp16)fmul345;
 fp16 cast354 = (fp16)fmul347;
 fp16 cast355 = (fp16)fmul349;
 fp16 cast356 = (fp16)fmul351;
 f16x4 vp357;
 vp357[0] = cast353;
 vp357[1] = cast354;
 vp357[2] = cast355;
 vp357[3] = cast356;
 int add358 = bid2 + c352;
 int add359 = c5 + add334;
 int c360 = 4096;
 int mul361 = add358 * c360;
 int add362 = mul361 + add359;
 *reinterpret_cast<f16x4*>(Y + add362) = vp357;
 int c363 = 512;
 int mul364 = c363 * c9;
 int mul365 = tid1 * c9;
 int add366 = mul364 + mul365;
 f16x4 gv4367 = *reinterpret_cast<const f16x4*>(Gamma + add366);
 fp16 e368 = gv4367[0];
 float f32369 = (float)e368;
 fp16 e370 = gv4367[1];
 float f32371 = (float)e370;
 fp16 e372 = gv4367[2];
 float f32373 = (float)e372;
 fp16 e374 = gv4367[3];
 float f32375 = (float)e374;
 float fmul376 = rsq298 * f32369;
 float fmul377 = fadd158 * fmul376;
 float fmul378 = rsq298 * f32371;
 float fmul379 = fadd160 * fmul378;
 float fmul380 = rsq298 * f32373;
 float fmul381 = fadd162 * fmul380;
 float fmul382 = rsq298 * f32375;
 float fmul383 = fadd164 * fmul382;
 int c384 = 0;
 fp16 cast385 = (fp16)fmul377;
 fp16 cast386 = (fp16)fmul379;
 fp16 cast387 = (fp16)fmul381;
 fp16 cast388 = (fp16)fmul383;
 f16x4 vp389;
 vp389[0] = cast385;
 vp389[1] = cast386;
 vp389[2] = cast387;
 vp389[3] = cast388;
 int add390 = bid2 + c384;
 int add391 = c5 + add366;
 int c392 = 4096;
 int mul393 = add390 * c392;
 int add394 = mul393 + add391;
 *reinterpret_cast<f16x4*>(Y + add394) = vp389;
 int c395 = 768;
 int mul396 = c395 * c9;
 int mul397 = tid1 * c9;
 int add398 = mul396 + mul397;
 f16x4 gv4399 = *reinterpret_cast<const f16x4*>(Gamma + add398);
 fp16 e400 = gv4399[0];
 float f32401 = (float)e400;
 fp16 e402 = gv4399[1];
 float f32403 = (float)e402;
 fp16 e404 = gv4399[2];
 float f32405 = (float)e404;
 fp16 e406 = gv4399[3];
 float f32407 = (float)e406;
 float fmul408 = rsq298 * f32401;
 float fmul409 = fadd215 * fmul408;
 float fmul410 = rsq298 * f32403;
 float fmul411 = fadd217 * fmul410;
 float fmul412 = rsq298 * f32405;
 float fmul413 = fadd219 * fmul412;
 float fmul414 = rsq298 * f32407;
 float fmul415 = fadd221 * fmul414;
 int c416 = 0;
 fp16 cast417 = (fp16)fmul409;
 fp16 cast418 = (fp16)fmul411;
 fp16 cast419 = (fp16)fmul413;
 fp16 cast420 = (fp16)fmul415;
 f16x4 vp421;
 vp421[0] = cast417;
 vp421[1] = cast418;
 vp421[2] = cast419;
 vp421[3] = cast420;
 int add422 = bid2 + c416;
 int add423 = c5 + add398;
 int c424 = 4096;
 int mul425 = add422 * c424;
 int add426 = mul425 + add423;
 *reinterpret_cast<f16x4*>(Y + add426) = vp421;
}
