# FFT3dGPU — Exhaustive technical documentation of the FFT implementation

> Document prepared for the OpenCL port of the AviSynth **FFT3dGPU** plugin (DirectX 9).
> Source root: `upstream\FFT3dGPU\FFT3dGPU`. All references are `file:line`.
> The HLSL/C formulas are copied **verbatim** (original indentation preserved in the code blocks).
> Analysis date: 2026-10-17. No source file was modified.

---

## Table of contents

1. [Overview](#1-overview)
2. [Architecture: what runs on the GPU, what runs on the CPU](#2-architecture)
3. [Common conventions (addressing, textures, shader macros)](#3-common-conventions)
4. [Data encoding (RGBA packing, real FFT trick)](#4-data-encoding)
5. [Algorithm 1 — RADIX2LUT (`FFT2dRR`, file `FFT.cpp`)](#5-algorithm-1--radix2lut-fft2drr)
6. [Algorithm 2 — STOCKHAM (`FFT2dRR2`, file `fft2.cpp`)](#6-algorithm-2--stockham-fft2drr2)
7. [1D temporal FFT (2 and 3 points)](#7-1d-temporal-fft-2-and-3-points)
8. [The `precision` parameter (0/1/2) and the texture formats](#8-the-precision-parameter-012-and-the-texture-formats)
9. [MEASURE mode (automatic choice of implementation)](#9-measure-mode)
10. [Numerical reproducibility — critical points for the port](#10-numerical-reproducibility)
11. [Appendix A — HLSL shader summary](#appendix-a--hlsl-shader-summary)
12. [Appendix B — Vertex geometry summary](#appendix-b--vertex-geometry-summary)
13. [Appendix C — DX9 → OpenCL mapping](#appendix-c--dx9--opencl-mapping)

---

## 1. Overview

The plugin implements the 2D (block-based) and 3D (spatial 2D + temporal 1D) FFT **entirely on the GPU**, through DirectX 9 pixel shaders and vertex shaders compiled at runtime from the `fft3dgpu.hlsl` file located next to the DLL (`core/shader.h:28` defines `SRC_SHADER`; `core/shader.cpp:54-76` compiles the source at runtime with D3DXCompileShader).

There are **two complete GPU implementations** of the 2D FFT, selected by the `FFTCODE` enum:

```cpp
// FFT3dGPU.h:58-61
enum FFTCODE {
  MEASURE,     // = 0 : benchmark les deux, garde la plus rapide
  RADIX2LUT,   // = 1 : FFT radix-2 avec tables d'indices (LUT) en textures  (classe FFT2dRR,  FFT.cpp)
  STOCKHAM,    // = 2 : FFT de Stockham (ping-pong), twiddles encodés dans les sommets (classe FFT2dRR2, fft2.cpp)
};
```

Dispatch happens in `FFT3dGPU.cpp:363-387`: `RADIX2LUT` creates a `FFT2dRR` object, `STOCKHAM` creates a `FFT2dRR2`, `MEASURE` creates both, times them (`MeasureFFT`, `FFT3dGPU.cpp:103-127`) and keeps the fastest.

**Key point: there is no CPU FFT at all.** The files `FFT.cpp` and `fft2.cpp` contain only host-side orchestration (texture creation, shader compilation, issuing the draw calls). The "CPU code" is used only to **build the tables** (twiddles, bit-reversal, load/save maps) in `double` on the CPU side, and then upload them into GPU textures as `float` (or `half` depending on `precision`).

Supported block sizes: `bw`/`bh` rounded to the nearest power of two, clamped to **[4, 512]** (`FFT3dGPU.cpp:214-231`).

---

## 2. Architecture

### 2.1 What runs on the GPU

- All FFT butterflies, in **pixel shaders**.
- The twiddles (rotation factors): either read from **LUT textures** (RADIX2LUT), or **interpolated from the vertex coordinates** (STOCKHAM), or computed by **hardware** `sincos` in the shader (STOCKHAM stage 0, C2Rfft/R2Cfft).
- The complex↔real conversions (C2R/R2C), the final collection, the normalization.

### 2.2 What runs on the CPU

Only table generation, in `double`, then uploaded as `float`/`half`:

| Table | Constructor | File:line | Contents |
|---|---|---|---|
| `TwiddleLUTforward/backward[i]` | `CreateTwiddleMap` | `FFT.cpp:743-768` | (cos, sin) per butterfly, **Numerical Recipes recurrence** |
| `LoadLUT2[i]` | `CreateBitReverseMap` + `CreateLoadMap` + `CreateSaveMap` + `CreateLoadMap2` | `FFT.cpp:714-723`, `633-646`, `566-582`, `605-615` | read texture coordinates + I/J texture selector |
| `SaveLUT[i]` | `CreateSaveMap` | `FFT.cpp:566-582` | write texture coordinates + I/J selector |
| `R2CLUT` / `C2RLUT` | `CreateC2RMap` | `FFT.cpp:788-820` | mirror coordinates + (cos, sin) of the real transform |
| STOCKHAM geometries | `FFTQuad0/2`, `C2RQuad`, `R2CFFTQuad` constructors | `core/vertexbuffer.cpp:2871+`, `2723+`, `2249+`, `2598+` | twiddles/angles stored as vertex data (`cos(-pi*m/n)` etc., computed in `double` by the C compiler, stored as `float32`) |

### 2.3 Files

| File | Role |
|---|---|
| `fft3dgpu.hlsl` | **All** the shaders (pixel + vertex), both algorithms. The file `hlsl/original/fft3dgpu.hlsl` is an older copy, not used. |
| `FFT.h` / `FFT.cpp` | `FFT2dRR` class = RADIX2LUT orchestration + all the CPU table constructors. |
| `FFTPS.h` / `FFTPS.cpp` | C++ wrappers for the RADIX2LUT shaders (`psBitReverseButterFly*`, `psButterflyCollect*`, `psCollect*`, `psC2R`, `psR2C`, `psFFT2p/3p`), MRT and multipass variants. |
| `fft2.h` / `fft2.cpp` | `FFT2dRR2` class = STOCKHAM orchestration. |
| `fft2ps.h` / `fft2ps.cpp` | C++ wrappers for the STOCKHAM shaders (`psButterflyHSP/HMP/VSP/VMP`, `psC2Rfft`, `psR2Cfft`, `psCombine`). |
| `core/vertexbuffer.cpp` | Geometries: `NQuad` (full screen), `C2RQuad`, `CollectQuad`, `R2CFFTQuad`, `FFTQuad2`, `FFTQuad0`. |
| `core/texture.cpp/.h` | `GPUTYPES` (formats), `TextureM` (LUT), `TextureRT` (rendering), `UploadToTexture` (half conversion). |
| `core/shader.cpp/.h` | Runtime compilation of the HLSL, macro helpers (`FloatToMacroArray`, `VecToMacroArray`, `SetMacroArray`, `CreateMacroArray`, `core/shader.cpp:136-262`). |
| `FFT3dGPU.cpp` | AviSynth plugin: `FFTCODE` dispatch, bw/bh clamping, precision override, sampler state. |

---

## 3. Common conventions

### 3.1 "Texel center" addressing

`NQuad` (`core/vertexbuffer.cpp:40-169`) generates a full-screen quad whose texture coordinates are **pre-transformed** so that they land exactly at the center of the texels: `texCoord = (i + 0.5) / W` (`core/vertexbuffer.cpp:87-100`). All the LUTs use the same `+0.5` offset (see `CreateBitReverseMap` `FFT.cpp:720`, `CreateSaveMap` `FFT.cpp:573/575`, `CreateC2RMap` `FFT.cpp:795-796`, etc.). The sampler filter is **POINT** (default); the addressing is set to `MIRROR` during execution (`FFT3dGPU.cpp:816-820`, created as `BORDER` in `core/ManageDirect3DWindow.cpp:221-225`).

> ⚠ For the OpenCL port: replace all of this with integer indices. The `+0.5` exists only to work around the interpolation of DX9 textures.

### 3.2 MRT vs multipass

If the card supports only one simultaneous render target (`Cap.NumSimultaneousRTs == 1`), the plugin compiles `_MP` variants of the shaders with the `MULTIPASS`/`PASS0`/`PASS1`/`PASS2` macros and issues **two draw calls** instead of one (one per output). The choice is made in `FFT.cpp:225-246` (RADIX2LUT) and `fft2.cpp:46-58` (STOCKHAM). The HLSL shaders contain both branches (`#ifndef MULTIPASS` / `#ifdef PASS1`…).

> For OpenCL: not applicable — one kernel writes everywhere. But the **semantics must be preserved**: each output is computed by the **same expression**, so the result is identical whichever path is taken.

### 3.3 Shader compilation macros

The shaders are compiled with constant macros injected by the C++ (`core/shader.cpp:136-262`):

- `PIXELOFFSET`: either a one-row LUT offset (`stagex[0] = 0.5/log2x`), or a one-texel shift (`1.0/x2n`) — see §5.
- `NORM`: normalization factor compiled into the shader (`BitReverseButterFlyBH` RADIX2LUT, `Combine` STOCKHAM). Defaults to `1.0` if absent (`fft3dgpu.hlsl:1148-1150`).
- `LOADOFFSET`: texture offset (`float2` macro) for reading the second operand of the STOCKHAM butterfly (`fft2ps.cpp:9,13,55,59-61,116,120,161,165-167`).
- `STAGE0`: forces the `sincos` computation in the pixel shader (STOCKHAM stage 0).
- `MULTIPASS`/`PASS0/1/2`: single-output variants.

### 3.4 Basic HLSL structures

```hlsl
// fft3dgpu.hlsl:44-55
sampler Src : register(s0);// : s0;
sampler Src2 : register(s2);
sampler I : register(s3);
sampler J : register(s4);
sampler Factor : register(s1);
sampler FactorM : register(s5);
sampler LoadMap : register(s6);

float2 _beta;
float2 _sigma;
float _stage : c0;
```

```hlsl
// fft3dgpu.hlsl:77-93
struct PS_OUTPUT
{
#ifndef MULTIPASS
	float4 o[2] : COLOR;
#else
	float4 o : COLOR;
#endif
};

struct PS_OUTPUT2
{
#ifndef MULTIPASS
	float4 o[3] : COLOR;
#else
	float4 o : COLOR;
#endif
};
```

---

## 4. Data encoding

### 4.1 One RGBA texel = two complex FFTs in parallel

Verbatim comment from the shader:

```hlsl
// fft3dgpu.hlsl:1103
//Does 2 fft in parallel. Data is stored in a float4 texture with Realfft1,Realfft2,Imagfft1,Imagfft2 in xyzw
```

A `float4 = (Re1, Re2, Im1, Im2)` texel carries **two complex numbers**: complex no. 1 = (x, z) = (Re1, Im1) and complex no. 2 = (y, w) = (Re2, Im2). The two components (x,z) and (y,w) are transformed independently, which gives two simultaneous FFTs.

### 4.2 Real FFT trick: two real signals in one half-size complex FFT

Documented verbatim in the constructor's documentation:

```cpp
// FFT.cpp:396-397
 *     the real arrayes a and b must be packed like this in one float 4  a[0] b[0] a[1] b[1]
 *     and the complex result like this a_real b_real a_imag b_imag
```

The real signal `a` of width `_x` is stored as a complex sequence of width `x = _x/2`: `c[k] = a[2k] + i·a[2k+1]` (even samples in Re, odd samples in Im). The signal `b` occupies the second complex component of the same texel. So we run a complex FFT of length `_x/2`, then the **R2C** shader (forward) separates and extends the spectrum to obtain the full real FFT of length `_x`, keeping only the useful half: `xV = _x/2 + 1` columns (`FFT.cpp:59`).

In the inverse direction, the **C2R** shader closes the real spectrum back into a half-size complex sequence before the inverse FFT.

> ⚠ Misleading naming: in the **forward** direction, it is the `R2C` shader (`fft3dgpu.hlsl:314`) that converts the result of the half-size complex FFT into a full-size real spectrum; in the **inverse** direction, it is the `C2R` shader (`fft3dgpu.hlsl:294`) that performs the opposite operation (`FFT.cpp:465` and `FFT.cpp:490-492`).

### 4.3 Dimensions

With `x = _x/2` (complex width), `xnum` blocks across the width, `y` block height, `ynum` blocks down the height, `bt` temporal width:

- Input image texture: `xn × yn` = `(x·xnum) × y` (float4, 2 signals).
- Intermediate horizontal spectrum (full-size complex): `xn × yn`.
- Final spectrum (kept half): `xVn × yn` = `xnum·(x+1) × y` (`FFT.cpp:140-141`).

---

## 5. Algorithm 1 — RADIX2LUT (`FFT2dRR`)

Class: `FFT2dRR` (`FFT.h`, `FFT.cpp`). Shaders: `fft3dgpu.hlsl:202-454`. Wrappers: `FFTPS.cpp`.

### 5.1 Principle

Radix-2 FFT **decimated in time** (bit-reversal at the input), with **2-in-1 fused** butterflies: each pixel shader computes one full butterfly and writes the two outputs into **two textures** `I` and `J` (MRT) or two passes. The texture pairs are reduced back to a single one at the end of the axis by the `Collect*` shader guided by `SaveLUT`. All twiddles and all read indices are read from **LUT textures** (`Factor`, `LoadMap`).

### 5.2 Textures and sizes (`FFT.cpp:130-149`)

| Texture | Size | Role |
|---|---|---|
| `TempF[0]` | `xn × yn` | full-size complex horizontal spectrum (before R2C / after C2R) |
| `TempI[0]`, `TempJ[0]`, `TempIpong[0]`, `TempJpong[0]` | `x2n × yn` (`x2n = xn/2`) | horizontal ping-pong (butterflies, 2 textures per buffer) |
| `TempF[1]` | `xVn × yn` | final spectrum (half the columns) |
| `TempI[1]`, `TempJ[1]`, `TempIpong[1]`, `TempJpong[1]` | `xVn × y2n` (`y2n = yn/2`) | vertical ping-pong |
| `TwiddleLUTforward/backward[i]` | `n2n[i] × log2n[i]`, `_float2` | twiddles per stage (i=0 horizontal, i=1 vertical) |
| `LoadLUT2[i]` | `n2n[i] × log2n[i]`, `_float2` | read coordinates + I/J selector |
| `SaveLUT[i]` | `nn[i] × log2n[i]`, `_float2` | write coordinates + I/J selector |
| `R2CLUT` | `xVn × 1`, `_float4` | forward complex→real conversion table |
| `C2RLUT` | `xn × 1`, `_float4` | inverse real→complex conversion table |

`_float4/_float2/_float1` formats: `FLOAT4/FLOAT2/FLOAT` if `precision > FLOAT16`, otherwise `HALF4/HALF2/HALF` (`FFT.cpp:100-111`). Output pool `FreeStreamPool`: `HALF4` unless `precision == FLOAT32_ALL` (`FFT.cpp:152`).

Stage constants (LUT row at the texel center):

```cpp
// FFT.cpp:206-215
  for (unsigned int i = 0, nn = 1; i < log2x; nn <<= 1, i++)
  {
    nnx[i] = (float)nn / xn;
    stagex[i] = (i + 0.5) / log2x;
  }
  for (unsigned int i = 0, nn = 1; i < log2y; nn <<= 1, i++)
  {
    nny[i] = (float)nn / yn;
    stagey[i] = (i + 0.5) / log2y;
  }
```

(`nnx`/`nny` are computed but **never used** — dead code, removed in the destructor `FFT.cpp:288-289`.)

### 5.3 Exact sequence — forward (`FFT.cpp:399-431`, `448-496`, `512-546`)

`CalcFFT(src, dst, forward=true)`: `Img = src`; `TempF[1]` = dst (or a texture from the pool); then **horizontal then vertical**; `dst = TempF[1]`.

**Horizontal** (`CalcFFTHori`, forward):

| # | Step | HLSL shader | Inputs | Outputs | Params |
|---|---|---|---|---|---|
| 1 | `BitReverseButterFlyH->Apply(Img, LoadLUT2[0], TempI[0], TempJ[0])` (`FFT.cpp:460`) | `BitReverseButterFlyH` (`fft3dgpu.hlsl:202-229`) | `Src`=Img, `LoadMap`=LoadLUT2[0] (row 0) | `TempI[0]` (o[0]), `TempJ[0]` (o[1]) | `PIXELOFFSET = stagex[0] = 0.5/log2x` |
| 2 | for `i = 1 … log2x-1`: `ButterflyCollectH->Apply(TempI[0], TempJ[0], TwiddleLUTforward[0], LoadLUT2[0], stagex[i], TempIpong[0], TempJpong[0])` then ping-pong swap (`FFT.cpp:475-485`) | `ButterflyCollectH` (`fft3dgpu.hlsl:338-376`) | `I`,`J`, `Factor`=TwiddleLUT, `LoadMap`=LoadLUT2, `_stage`=stagex[i] | pong I/J | `PIXELOFFSET = 1.0/x2n` (`FFT.cpp:240`) |
| 3 | `CollectH->Apply(TempI[0], TempJ[0], SaveLUT[0], stagex[log2x-1], TempF[0])` (`FFT.cpp:489`) | `CollectH` (`fft3dgpu.hlsl:421-437`) | `I`,`J`, `LoadMap`=SaveLUT[0], `_stage` | `TempF[0]` | — |
| 4 | `R2C->Apply(TempF[0], R2CLUT, Temp)` (`FFT.cpp:492`) | `R2C` (`fft3dgpu.hlsl:314-334`) | `Src`=TempF[0], `Factor`=R2CLUT | `Temp` (= old `TempF[1]`) | — |

**Vertical** (`CalcFFTVert`, forward): same steps on the y axis, `LoadLUT2[1]`, `TwiddleLUTforward[1]`, `SaveLUT[1]`, ping-pong `TempI[1]/TempJ[1]` (height `y2n`), input `Temp`, final output `TempF[1]` (`FFT.cpp:520-543`).

### 5.4 Exact sequence — inverse (`FFT.cpp:422-430`, `463-469`, `524-527`)

`CalcFFT(src, dst, forward=false)`: `Img = src` (spectrum `xVn × yn`); `TempF[0] = dst`; **vertical first, then horizontal**; `dst = TempF[0]`.

| # | Step | Detail |
|---|---|---|
| 1 | `BitReverseButterFlyV->Apply(Img, LoadLUT2[1], TempI[1], TempJ[1])` (`FFT.cpp:526`) | vertical FFT (length y, full complex) over the columns of the real spectrum, twiddles `TwiddleLUTbackward[1]`, stages `stagey[i]`, `CollectV → TempF[1]` (`FFT.cpp:543`) |
| 2 | `C2R->Apply(TempF[1], C2RLUT, Temp)` (`FFT.cpp:465`) | closes the horizontal real spectrum back into a half-size complex sequence; `Temp` = old `TempF[0]` (`xn × yn`) |
| 3 | `BitReverseButterFlyBH->Apply(Temp, LoadLUT2[0], TempI[0], TempJ[0])` (`FFT.cpp:469`) | bit-reversal + first butterfly **with the 1/N normalization** (see §5.5) |
| 4 | `ButterflyCollectH` stages with `TwiddleLUTbackward[0]` (`FFT.cpp:475-485`) | same as forward but with conjugate twiddles |
| 5 | `CollectH → TempF[0]` = dst (`FFT.cpp:489`) | output `xn × yn` |

### 5.5 Where the 1/N normalization lives (RADIX2LUT)

Only in the `BitReverseButterFlyBH` shader, applied **at the very start of the first inverse horizontal butterfly**:

```cpp
// FFT.cpp:60 et 95
  , norm(1.0 / (_x*_y*_bt)), ...
  norm = 1.0 / (x*y*bt);
```

Note: on line 60, `x`/`y` are the already-initialized members (`x = _x/2`), so the effective value is `1/((_x/2)·_y·bt)` — the full normalization is spread out: the missing factor of 2 comes from the fact that the horizontal FFT is half-size and that the a+b pair is reconstructed without a factor (to be checked on the filter side if needed; the value compiled into the shader is `norm` as computed on line 95, copied into the `NORM` macro in `FFTPS.cpp:163`).

```hlsl
// fft3dgpu.hlsl:233-259
PS_OUTPUT BitReverseButterFlyBH(PS_INPUT In)
{
PS_OUTPUT dst;
float2 Index;
float2 LM;
float4 i;
float4 j;
Index=In.texCoord;
Index.y=PIXELOFFSET;
LM=tex2D(LoadMap,Index).xy;
Index.y=In.texCoord.y;
Index.x=LM.x;
i=tex2D(Src,Index)*NORM;
Index.x=LM.y;
j=tex2D(Src,Index)*NORM;
#ifndef MULTIPASS
dst.o[1]=i-j;//J
dst.o[0]=i+j;//I
#else
#ifdef PASS1
dst.o=i+j;//I
#else
dst.o=i-j;//J
#endif
#endif
return dst;
}
```

The temporal factor `1/bt` is therefore **folded into the spatial normalization** — there is no separate normalization pass for the temporal dimension.

### 5.6 RADIX2LUT shaders — verbatim formulas

**BitReverseButterFlyH** (`fft3dgpu.hlsl:202-229`) — bit-reversal + first horizontal butterfly:

```hlsl
PS_OUTPUT BitReverseButterFlyH(PS_INPUT In)
{
PS_OUTPUT dst;
float2 Index;
float2 LM;
float4 i;
float4 j;
Index=In.texCoord;
Index.y=PIXELOFFSET;
LM=tex2D(LoadMap,Index).xy;
Index.y=In.texCoord.y;
Index.x=LM.x;
i=tex2D(Src,Index);
Index.x=LM.y;
j=tex2D(Src,Index);
#ifndef MULTIPASS
dst.o[1]=i-j;//J
dst.o[0]=i+j;//I
#else
#ifdef PASS1
dst.o=i+j;//I
#else
dst.o=i-j;//J
#endif
#endif
return dst;
}
```

`BitReverseButterFlyV` (`fft3dgpu.hlsl:264-290`) is identical with the axes swapped (`Index.x=In.texCoord.y` for the LUT index, `Index.y=LM.x/y` for the read).

**ButterflyCollectH** (`fft3dgpu.hlsl:338-376`) — butterfly for stages 1+ with twiddle:

```hlsl
PS_OUTPUT ButterflyCollectH(PS_INPUT In)
{
PS_OUTPUT dst;
float2 Index;
float2 W;
float2 LM;
float4 i;
float4 j;
float2 Offset;
float4 temp;
Index.x=In.texCoord.x;
Index.y=_stage;
W=tex2D(Factor,Index).xy;
LM=tex2D(LoadMap,Index).xy;
Index.y=In.texCoord.y;
Index.x=LM.x;
Offset.x=PIXELOFFSET;
Offset.y=0;//1 pixel offset;
if(LM.y==0){
	i=tex2D(I,Index);
	j=tex2D(I,Index+Offset);}
else{
	i=tex2D(J,Index);
	j=tex2D(J,Index+Offset);}

temp.xy=W.xx*j.xy-W.yy*j.zw;
temp.zw=W.xx*j.zw+W.yy*j.xy;
#ifndef MULTIPASS
dst.o[1]=i-temp;
dst.o[0]=i+temp;
#else
#ifdef PASS1
	dst.o=i+temp;//I
#else
	dst.o=i-temp;//J
#endif
#endif
return dst;
}
```

Reading the algebra: `W = (Wr, Wi)`; `temp = j·W` (complex multiplication); outputs `i + temp` and `i − temp`. The `LM.y` selector (0 or 1) says which texture (I or J) holds the butterfly's partner. `PIXELOFFSET = 1/x2n` = one texel.

`ButterflyCollectV` (`fft3dgpu.hlsl:380-418`): same, axes swapped, `Offset.y=PIXELOFFSET` (`PIXELOFFSET = 1.0/y2n`, `FFT.cpp:242`).

**CollectH** (`fft3dgpu.hlsl:421-437`) — merges I/J into a single texture:

```hlsl
float4 CollectH( PS_INPUT In ) : COLOR
{
float4 dst;
//float2 Offset;
float2 Index1;
float2 Index2;
Index1=In.texCoord;
Index1.y=_stage;
Index2=tex2D(LoadMap,Index1).xy;
Index1.x=Index2.x;
Index1.y=In.texCoord.y;
if(Index2.y==0)
	dst=tex2D(I,Index1);
else
	dst=tex2D(J,Index1);
return dst;
}
```

`CollectV` (`fft3dgpu.hlsl:439-454`): same, vertical.

**R2C** (forward, half-size complex → real spectrum) (`fft3dgpu.hlsl:314-334`):

```hlsl
float4 R2C(PS_INPUT In) : COLOR
{
	float4 dst;
	float4 M=tex2D(Factor,In.texCoord);
	float2 Index=In.texCoord;
	float4 F;
	float4 N;
	float4 FpN;
	float4 FmN;
	Index.x=M.x;
	F=tex2D(Src,Index);
	Index.x=M.y;
	N=tex2D(Src,Index);
	FpN=0.5*(F+N);
	FmN=0.5*(F-N);
	dst.xy=FpN.zw*M.zz+FmN.xy*M.ww+FpN.xy;
	dst.zw=-FmN.xy*M.zz+FpN.zw*M.ww+FmN.zw;
	return dst;
}
```

`M = (posF, posN, cos, sin)` read from `R2CLUT`; `F` = F(k), `N` = F(N−k) (mirror positions). Equivalent mathematical formula (component 1): `Re' = cos·Im(F−N)… ` — see §10 for the derivation; copy the swizzle verbatim.

**C2R** (inverse, real spectrum → half-size complex) (`fft3dgpu.hlsl:294-312`):

```hlsl
float4 C2R(PS_INPUT In) : COLOR
{
	float4 dst;
	float4 M=tex2D(Factor,In.texCoord);
	float2 Index=In.texCoord;
	float4 F;
	float4 N;
	float4 FpN;
	float4 FmN;
	Index.x=M.x;
	F=tex2D(Src,Index);
	Index.x=M.y;
	N=tex2D(Src,Index);
	FpN=0.5*(F+N);
	FmN=0.5*(F-N);
	dst.xy=-FpN.zw*M.zz+FmN.xy*M.ww+FpN.xy;
	dst.zw=FmN.xy*M.zz+FpN.zw*M.ww+FmN.zw;
	return dst;
}
```

C2R and R2C are **conjugates**: the signs of the `M.zz` (cos) terms are opposite.

### 5.7 RADIX2LUT CPU tables — verbatim formulas

**Twiddles — Numerical Recipes recurrence** (`FFT.cpp:743-768`):

```cpp
void FFT2dRR::CreateTwiddleMap(float *Map, unsigned int n, bool forward, unsigned int rep) {
  //WR,WI0 WR,WI1 WR,WI2 ... ... WR,WIn
  int isign = forward * 1 - 1 * !forward;

  double theta, wtemp, wpr, wpi, wr, wi;
  for (unsigned int m = 2, y = 0; m < n * 2; m <<= 1) {
    for (unsigned int r = 0; r < rep; r++, y += n) {
      theta = (2 * pi / m * isign);
      wtemp = sin(0.5*theta);
      wpr = -2 * wtemp*wtemp;
      wpi = sin(theta);
      wr = 1;
      wi = 0;
      for (unsigned int o = 0, x = 0; o < m; o += 2) {
        for (unsigned int i = o; i < n; i += m) {
          Map[y + x] = wr;
          Map[y + x + 1] = wi;
          x += 2;
        }
        wr = (wtemp = wr)*wpr - wi * wpi + wr;
        wi = wi * wpr + wtemp * wpi + wi;
      }
    }
  }

}
```

⚠ The twiddles are **not** direct `cos/sin` values: they are the values produced by the **Numerical Recipes recurrence** (error accumulated along the stage). `isign = +1` in the forward direction, `−1` in the inverse direction. To reproduce them bit-for-bit, this recurrence must be reproduced in `double` exactly.

**Bit-reversal** (`FFT.cpp:714-723`):

```cpp
void FFT2dRR::CreateBitReverseMap(float *Map, unsigned int n, unsigned int rep) {
  unsigned int logn = 0;
  float div = n * rep;
  for (unsigned int i = (n - 1); i > 0; i /= 2, logn++);
  for (unsigned int r = 0, y = 0, roff = 0; r < rep; r++, roff += n, y += n) {
    for (unsigned int i = 0; i < n; i++) {
      Map[y + i] = (ReverseBit(i, logn) + roff + 0.5) / div;
    }
  }
}
```

`ReverseBit`: x86 assembly version (`ror`/`rcl`, `FFT.cpp:678-696`), portable C version `ReverseBit_c` (`FFT.cpp:664-676`) used on x64. Identical result (bit permutation).

**SaveMap** (`FFT.cpp:566-582`):

```cpp
void FFT2dRR::CreateSaveMap(float *Map, unsigned int n, unsigned int rep) {
  float div = n / 2 * rep;//n=x|y I=n/2 OK?
  for (unsigned int m = 2, y = 0; m < n * 2; m <<= 1) {
    for (unsigned int r = 0, roff = 0; r < rep; r++, roff += n / 2, y += 2 * n) {
      for (unsigned int o = 0, x = 0; o < m; o += 2) {
        for (unsigned int i = o; i < n * 2; i += m * 2) {
          unsigned int j = i + m;
          Map[y + i] = (x + roff + 0.5) / div;
          Map[y + i + 1] = 0; //Texture I
          Map[y + j] = (x + roff + 0.5) / div;
          Map[y + j + 1] = 1; //Texture J
          x++;
        }
      }
    }
  }
}
```

**LoadMap** (`FFT.cpp:633-646`):

```cpp
void FFT2dRR::CreateLoadMap(float *Map, unsigned int n, unsigned int rep) {
  for (unsigned int m = 1, y = 0; m < n; m <<= 1) {
    for (unsigned int r = 0, roff = 0; r < rep; r++, roff += n, y += n / 2) {
      for (unsigned int o = 0, x = 0; o < m; o += 1) {
        for (unsigned int i = o; i < n; i += m * 2) {

          Map[y + x] = i + roff;

          x++;
        }
      }
    }
  }
}
```

**LoadMap2** (bit-reversal + load fused per stage) (`FFT.cpp:605-615`):

```cpp
void FFT2dRR::CreateLoadMap2(float *Map, float* BitReverseMap, float* LoadMap, float* SaveMap, int width, int height) {

  for (int x = 0; x < width; x++)
    Map[x] = BitReverseMap[x];
  for (int y = 1, offset = width / 2, offset1 = 0; y < height; y++, offset += width / 2, offset1 += 2 * width)
    for (int x = 0; x < width / 2; x++) {
      int index = LoadMap[x + offset] * 2;
      Map[2 * x + offset * 2] = SaveMap[index + offset1];
      Map[2 * x + 1 + offset * 2] = SaveMap[index + 1 + offset1];
    }
}
```

**C2R/R2C table** (`FFT.cpp:788-820`) — careful, here `forward=true` means "complex→real conversion", so it is used for the **R2CLUT of the forward direction** (`CreateC2RMap(R2CMap, x, true, xnum)`, `FFT.cpp:120`) and `forward=false` for the inverse `C2RLUT` (`FFT.cpp:124`):

```cpp
void FFT2dRR::CreateC2RMap(float *Map, unsigned int n, bool forward, unsigned int rep) {
  unsigned int N = 2 * n;
  float div = (n + !forward)*rep;
  //float sign;
  for (unsigned int r = 0, y = 0, roff = 0; r < rep; r++, roff += n + !forward, y += (n + forward) * 4) {
    if (forward) {
      //sign=1.0;
      Map[0 + y] = (roff + 0.5) / div;
      Map[1 + y] = (roff + 0.5) / div;
      Map[2 + y] = 1;
      Map[3 + y] = 0;
    }
    else {
      //sign=-1.0;
      Map[0 + y] = (roff + 0.5) / div;
      Map[1 + y] = (n + roff + 0.5) / div;
      Map[2 + y] = 1;
      Map[3 + y] = 0;
    }
    for (unsigned int i = 1, o = 4; i < n; i++, o += 4) {
      Map[o + y] = (i + roff + 0.5) / div;
      Map[o + 1 + y] = (n - i + roff + 0.5) / div;
      Map[o + 2 + y] = cos(2 * pi*i / N);
      Map[o + 3 + y] = sin(2 * pi*i / N);
    }
    if (forward) {
      Map[n * 4 + y] = (roff + 0.5) / div;
      Map[n * 4 + 1 + y] = (roff + 0.5) / div;
      Map[n * 4 + 2 + y] = cos(2 * pi*n / N);
      Map[n * 4 + 3 + y] = sin(2 * pi*n / N);
    }
  }
}
```

Here the conversion twiddles are **direct `cos`/`sin` values in double** (no recurrence), stored as float. The first two fields are the texture coordinates of the mirror pairs `(i, N−i)`.

---

## 6. Algorithm 2 — STOCKHAM (`FFT2dRR2`)

Class: `FFT2dRR2` (`fft2.h`, `fft2.cpp`). Shaders: `fft3dgpu.hlsl:1068-1204`. Wrappers: `fft2ps.cpp`. Geometries: `core/vertexbuffer.cpp` (`FFTQuad0`, `FFTQuad2`, `C2RQuad`, `R2CFFTQuad`, `CollectQuad`).

### 6.1 Principle

**Stockham** FFT (no bit-reversal, automatic ping-pong). The twiddles are **not** stored in textures: they are encoded in the **vertex texture coordinates** (interpolated by the rasterizer), one quad per butterfly. Each stage draws two groups of quads (`DrawOdd`/`DrawEven`) to distribute the two butterfly outputs into two destination textures.

Dimensions (`fft2.cpp:23-34`): `cx = x/2` (complex width), `rx = cx+1`, `_log2cx = log2(x/2)`, `_log2y = log2(y)`.

### 6.2 Textures (`fft2.cpp:36-45`)

| Texture | Size | Role |
|---|---|---|
| `TempHorizontal1..4` | `srcwidth/2 × height` = `cx·xrep/2 × y·yrep` | horizontal ping-pong (2 textures per buffer) |
| `TempVertical1..4` | `width × height/2` = `(cx+1)·xrep × y·yrep/2` | vertical ping-pong |
| Output pool `FreeStreamPool` | `width × height` | `HALF4` except `FLOAT32_ALL` (`fft2.cpp:65`) |

Internal format: `precision > FLOAT16 ? FLOAT4 : HALF4` (`fft2.cpp:36`).

### 6.3 Exact sequence — forward (`fft2.cpp:96-123`)

```cpp
// fft2.cpp:100-122 (extrait)
    ping1 = TempHorizontal1; ping2 = TempHorizontal2; pong1 = TempHorizontal3; pong2 = TempHorizontal4;
    ButterFlyH->Apply(src, pong1, pong2, true);                 // stage 0
    std::swap(ping1, pong1); std::swap(ping2, pong2);

    for (unsigned int i = 1; i < _log2cx; i++)
    {
      ButterFlyH->Apply(ping1, ping2, pong1, pong2, i, true);   // stages 1..log2cx-1
      std::swap(ping1, pong1); std::swap(ping2, pong2);
    }
    //convert complex halfsize fft to real fullsize fft
    C2Rfft->Apply(ping1, ping2, dst);                           // C2Rfft (4 draws)
    //Vertical 1d fft
    ping1 = TempVertical1; ping2 = TempVertical2; pong1 = TempVertical3; pong2 = TempVertical4;
    ButterFlyV->Apply(dst, pong1, pong2, true);                 // stage 0 vertical
    std::swap(ping1, pong1); std::swap(ping2, pong2);
    for (unsigned int i = 1; i < _log2y; i++)
    {
      ButterFlyV->Apply(ping1, ping2, pong1, pong2, i, true);   // stages 1..log2y-1
      std::swap(ping1, pong1); std::swap(ping2, pong2);
    }
    //combine result in ping1 and ping2
    CombineV->Apply(ping1, ping2, dst);                         // NORM=1
```

### 6.4 Exact sequence — inverse (`fft2.cpp:124-150`)

```cpp
    //vertical
    ButterFlyV->Apply(src, pong1, pong2, false);                // stage 0
    for (i = 1 … _log2y-1) ButterFlyV->Apply(ping1, ping2, pong1, pong2, i, false);
    //convert from real fft to complex
    R2Cfft->Apply(ping1, ping2, dst);                           // R2Cfft (2 draws)
    //horizontal
    ButterFlyH->Apply(dst, pong1, pong2, false);                // stage 0
    for (i = 1 … _log2cx-1) ButterFlyH->Apply(ping1, ping2, pong1, pong2, i, false);
    //combine
    CombineH->Apply(ping1, ping2, dst);                         // NORM = 2.0/(x*y*z)
```

### 6.5 Normalization (STOCKHAM)

```cpp
// fft2.cpp:25
  , norm(2.0 / (x*y*z))
// fft2.cpp:61-62
  CombineV = NEW psCombine(pDevice, xrep, rx, yrep, y, false, 1);
  CombineH = NEW psCombine(pDevice, xrep, cx, yrep, y, true, norm);
```

- Forward: `CombineV` with `NORM=1` (no normalization).
- Inverse: `CombineH` with `NORM = 2.0/(x·y·z)` (x,y = block size, z = bt). The factor **2** compensates for the `0.5` applied twice in the `R2Cfft` shader (see §6.8): the `0.5` in R2Cfft divides by 2, the `2.0` in Combine restores it.

Combine shader (`fft3dgpu.hlsl:1152-1154`):

```hlsl
float4 Combine(VS_OUTPUT1 In):COLOR{
return tex2D(Src0,In.texCoord)*NORM;
}
```

`NORM` is a compile-time macro (`fft2ps.cpp:257`). Default `1.0` if not defined (`fft3dgpu.hlsl:1148-1150`).

### 6.6 STOCKHAM butterfly — vertex shaders and twiddles

**Vertex shaders** (`fft3dgpu.hlsl:1068-1080`):

```hlsl
VS_OUTPUT4 fft2(VS_OUTPUT4 In)
{
	VS_OUTPUT4 Out=In;
	Out.texCoord[1].y=dir*Out.texCoord[1].y;
	return Out;
}

VS_OUTPUT4 fft0(VS_OUTPUT4 In)
{
	VS_OUTPUT4 Out=In;
	Out.texCoord[1].x=dir*Out.texCoord[1].x;
	return Out;
}
```

- `fft2` (stages ≥ 1): the vertex carries `texCoord[1] = (cos, sin)` already computed on the CPU; the VS **flips the sign of the sin component** according to `dir` (`dir = forward ? 1 : -1`, `fft2ps.cpp:49-50` and `103-104`).
- `fft0` (stage 0): the vertex carries an **angle** in `texCoord[1].x`; the VS flips the sign of the **angle** before the GPU `sincos`.

**Twiddles for stages ≥ 1** — computed on the CPU in double, stored as float32 in the vertex buffer (`core/vertexbuffer.cpp:2774`, repeated at 2782/2790/2798):

```cpp
v[nvertex].tex[1].u = cos(-pi * m / n); v[nvertex].tex[1].v = sin(-pi * m / n);
```

where `n` = complex block size, `m` = butterfly index within the stage (loop `for (int l = 0, m = 0; l < parts; l++, offset += 2 * size, srcoffset += size, m += 2 * size)`, `core/vertexbuffer.cpp:2764`). Forward angle = `−π·m/n`.

**Stage 0** (`FFTQuad0`, `core/vertexbuffer.cpp:2871-2958`): the angle is encoded per quad corner and interpolated linearly across the quad: `tex[1].u ∈ {0, −π}` (`core/vertexbuffer.cpp:2902, 2909, 2911, 2920, 2929`). The pixel shader does `sincos(I.x, T.y, T.x)` → `T = (cos, sin)` of the interpolated angle.

**Stage 0 — important detail**: `FFTQuad0::Execute`/`psButterfly*::Apply(stage 0)` does pass `dir` (`fft2ps.cpp:49-50`), but since the angle is 0 or −π, `e^{−iπ} = e^{+iπ} = −1`: the direction sign has no numerical effect at stage 0.

**Odd/even split**: `FFTQuad2` pre-builds the geometry of all stages (`core/vertexbuffer.cpp:2756-2805`) and stage `s` is drawn via `stageLUT[endstage - s]` (`core/vertexbuffer.cpp:2846-2856`):

```cpp
HRESULT FFTQuad2::DrawOdd(int stage)
{
  int i = endstage - stage;
  return _pDevice->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, stageLUT[i].minvertexodd, stageLUT[i].nvertex, stageLUT[i].startindexodd, stageLUT[i].nvertex / 2);
}
```

Each stage first draws the "odd" quads (output `a+b` → dst1), then the "even" quads (output `(a−b)·W` → dst2) — see the PS below. `LOADOFFSET` (a `float2` macro) gives the read offset of the second term: `1/(xrep·2)` in u for the horizontal pass (`fft2ps.cpp:9`), `1/(yrep·2)` in v for the vertical pass (`fft2ps.cpp:116`), i.e. half a block size in texels.

### 6.7 STOCKHAM butterfly — pixel shader verbatim (`fft3dgpu.hlsl:1112-1144`)

```hlsl
#ifdef LOADOFFSET
//LOADOFFSET = 0.5 for single fft
PS_OUTPUT Butterfly(VS_OUTPUT4 In)
{
	PS_OUTPUT Out;
	float4 a;
	float4 b;
	float2 T;
	float2 I=In.texCoord[1];
	#ifdef STAGE0
	sincos(I.x,T.y,T.x);
	#else
	T=I;
	#endif
	a=tex2D(Src0,In.texCoord[0]);
	b=tex2D(Src0,In.texCoord[0]+LOADOFFSET);
	#ifdef MULTIPASS
		#ifdef PASS0
	Out.o=a+b;
		#else
	float4 temp=a-b;
	float4 T1=float4(T.x,T.x,-T.y,T.y);
	Out.o=temp.zwxy*T1.zzww+temp.xyzw*T1.xyyy;
		#endif
	#else
	Out.o[0]=a+b;
	float4 temp=a-b;
	float4 T1=float4(T.x,T.x,-T.y,T.y);
	Out.o[1]=temp.zwxy*T1.zzww+temp.xyzw*T1.xyyy;
	#endif
	return Out;
}
#endif	
```

Sequence of the algebra (to reproduce in OpenCL): `T = (c, s)`; `T1 = (c, c, −s, s)`; `temp = a − b = (tr1, tr2, ti1, ti2)`;

```
Out = (ti1·(−s) + tr1·c,  ti2·(−s) + tr2·c,  tr1·s + ti1·c,  tr2·s + ti2·c)
    = (Re1', Re2', Im1', Im2')   avec  (Re'+i·Im') = (tr+i·ti)·(c+i·s)
```

that is exactly `(a−b)·W` complex component by complex component, **plus** the `a+b` output. The two outputs are the two RTs (MRT) or two passes (`PASS0` = `a+b`, the other = the product).

### 6.8 C2Rfft / R2Cfft STOCKHAM — verbatim

**R2Cfft** (inverse, real→complex, `fft3dgpu.hlsl:1156-1171`):

```hlsl
float4 R2Cfft(VS_OUTPUT3 In):COLOR
{
float4 a,b;
float4 Out;
float2 T;
a=tex2D(Src0,In.texCoord[0]);
b=tex2D(Src0,In.texCoord[1]);
//calc twiddle
sincos(In.texCoord[2].x,T.y,T.x);
//Out.xy=((a.zw+b.zw)*T.x-(a.xy-b.xy)*T.y+(a.xy+b.xy))*0.5;
//Out.zw=(-(a.xy-b.xy)*T.x-(a.zw+b.zw)*T.y+(a.zw-b.zw))*0.5;
Out.xy=(-(a.zw+b.zw)*T.x+(b.xy-a.xy)*T.y+a.xy+b.xy)*0.5;
Out.zw=((a.xy-b.xy)*T.x-(a.zw+b.zw)*T.y+a.zw-b.zw)*0.5;
//Out=float4(In.texCoord[2].x,In.texCoord[0].x*W,In.texCoord[1].x*W,Out.x+Out.z);
return Out;
}
```

The angle comes from the vertex (`texCoord[2].x`), interpolated from **0 to π** across the width of the block (`R2CFFTQuad`, `core/vertexbuffer.cpp:2631, 2638, 2645, 2652`: `v[nvertex].tex[2].u = 0` on the left, `= pi` on the right) → effective angle `π·i/x`. The `sincos` is done **on the GPU**. The two reads `a` (column i) and `b` (mirror column x−i) come from the **same** texture (`Src0`), quads `DrawOdd`/`DrawEven` (`fft2ps.cpp:289-302`).

**C2Rfft_ps** (forward, complex→real, `fft3dgpu.hlsl:1177-1204`):

```hlsl
bool t00;
bool t11;

float4 C2Rfft_ps(VS_OUTPUT3 In):COLOR0
{
float4 Out;
float4 a,b;
float2 T;
if(t00)
{
a=tex2D(Src0,In.texCoord[0]);
Out.xy=a.xy+In.texCoord[2].x*a.zw;
Out.zw=float2(0,0);
}
if(t11)
{
a=tex2D(Src0,In.texCoord[0]);
Out.xy=a.xy;
Out.zw=-a.zw;
}
else if(!t00)
{
a=tex2D(Src0,In.texCoord[0]);
b=tex2D(Src1,In.texCoord[1]);
sincos(In.texCoord[2].x,T.y,T.x);
Out.xy=((a.zw+b.zw)*T.x-(a.xy-b.xy)*T.y+(a.xy+b.xy))*0.5;
Out.zw=(-(a.xy-b.xy)*T.x-(a.zw+b.zw)*T.y+(a.zw-b.zw))*0.5;
}
//Out=float4(Out.x+Out.z,W*In.texCoord[0].x,W*In.texCoord[1].x,In.texCoord[2].x);
return Out;
}
```

The `C2RQuad` geometry (`core/vertexbuffer.cpp:2249-2476`) draws **5 quads per block** covering the `x+1` columns of the real spectrum, in 4 groups with constant flags (`fft2ps.cpp:232-253`):

```cpp
// core/vertexbuffer.cpp:2253-2263 (commentaire verbatim)
   //5 quads per fft organized like this:
   //quad 1: from 0 to 1
   //quad 2: from 1 to x/2-1
   //quad 3: from x/2-1 to x/2
   //quad 4: from x/2 to x-1
   //quad 5:from x-1 to x
   //total size=x+1
   // quad 1 and 5 belongs to 00 as they only read from texture 0
   // quad 2 to 01 as it read from texture 0 and 1
   // quad 4 to 10 as it read from texture 1 and 0
   // quad 5 to 11 as it only read from texture 1
```

- **Draw00** (columns 0 and x, DC/Nyquist case): `t00=true`; `texCoord[2].x = ±1` encoded per vertex (`core/vertexbuffer.cpp:2289`: `v[nvertex].tex[2].u = (j == 0 ? 1 : -1)`) → `Out.xy = a.xy ± a.zw`, imaginary = 0.
- **Draw01 / Draw10** (general case): formula with GPU `sincos`; corner angles `2 * pi * 1 / (2 * x)` and `2 * pi*(x / 2) / (2 * x)` (`core/vertexbuffer.cpp:2325, 2332`) and `2 * pi*(1 + x / 2) / (2 * x)`, `2 * pi*(x) / (2 * x)` (`core/vertexbuffer.cpp:2396, 2403`) — interpolated linearly, so exact angle `2π·i/(2x)` per pixel.
- **Draw11** (center column): `t11=true`; `Out.xy = a.xy ; Out.zw = -a.zw`.

Compare with the R2Cfft: the sign of the `T.x` term is **conjugated** between forward (C2Rfft, `+(a.zw+b.zw)*T.x`) and inverse (R2Cfft, `-(a.zw+b.zw)*T.x`), and the `T.y` term goes from `-(a.xy-b.xy)` to `+(b.xy-a.xy)`.

### 6.9 Combine (STOCKHAM)

`psCombine` (`fft2ps.cpp:256-278`) uses `CollectQuad` (`core/vertexbuffer.cpp:2478-2594`): two draws (odd/even) that gather the two ping-pong half-textures into the final texture, with multiplication by `NORM` (macro).

---

## 7. 1D temporal FFT (2 and 3 points)

When `bt = 2` or `3`, the temporal dimension is handled by a direct mini-FFT on 2 or 3 **textures** (one per image of the temporal buffer), without twiddle.

Shaders (`fft3dgpu.hlsl:456-535`):

```hlsl
PS_OUTPUT FFT2p( PS_INPUT In)
{
PS_OUTPUT dst;
float4 src1=tex2D(I,In.texCoord);
float4 src2=tex2D(J,In.texCoord); 
#ifndef MULTIPASS
dst.o[0]=src1+src2;
dst.o[1]=src1-src2;
#else
#ifdef PASS1
	dst.o=src1+src2;
#else
	dst.o=src1-src2;
#endif
#endif
return dst;
}

float4 iFFT2p( PS_INPUT In) : COLOR
{
float4 src1=tex2D(I,In.texCoord);
float4 src2=tex2D(J,In.texCoord); 
return src1 - src2;
}

PS_OUTPUT2 FFT3p( PS_INPUT In)
{
PS_OUTPUT2 dst;
float4 src1=tex2D(Src,In.texCoord);
float4 src2=tex2D(I,In.texCoord);
float4 src3=tex2D(J,In.texCoord);
float4 t1=src2+src3;
float4 y1=src1-0.5*t1;
//-0.8660254=sin(pi*4/3)
float4 y2;
y2.zwxy=float4(-0.8660254,-0.8660254,0.8660254,0.8660254)*(src2-src3);
#ifndef MULTIPASS
dst.o[0]=src1+t1;
dst.o[1]=y1+y2;
//dst.o[1].xy=y1.xy-y2.zw;
//dst.o[1].zw=y1.zw+y2.xy;
dst.o[2]=y1-y2;
//dst.o[2].xy=y1.xy+y2.zw;
//dst.o[2].zw=y1.zw-y2.xy;
#else
...
#endif
return dst;
}

float4 iFFT3p( PS_INPUT In) : COLOR
{
float4 src1=tex2D(Src,In.texCoord);
float4 src2=tex2D(I,In.texCoord);
float4 src3=tex2D(J,In.texCoord);
float4 dst;
float4 t1=src2+src3;
float4 y1=src1-0.5*t1;
float4 y2;//=0.8660254*(src2-src3);
y2.zwxy=float4(-0.8660254,-0.8660254,0.8660254,0.8660254)*(src2-src3);
//dst.xy=(y1.xy-y2.zw);
//dst.zw=(y1.zw+y2.xy);
dst=y1-y2;
return dst;
}
```

- `FFT3p`: 3-point FFT; `−0.8660254 = sin(π·4/3) = −√3/2` (constant **truncated to 7 decimal places** in the shader - to be copied as is for bit-for-bit compatibility). The ±√3/2 rotation is done by the `zwxy` swizzle (Re/Im swap with sign).
- Inverse: **only one output** is kept (`iFFT2p` returns `src1 − src2`, `iFFT3p` returns `y1 − y2`) - the filters only need the current image.
- Orchestration: `FFT2dRR::CalcFFT1d` (`FFT.cpp:312-331` for simple textures, `FFT.cpp:350-379` for texture pairs "mode 1"). In inverse, the result is written to `dst[1]` only.
- The `1/bt` normalization is **already folded into the spatial `norm`** (§5.5), there is no temporal pass.

---

## 8. The `precision` parameter (0/1/2) and the texture formats

```cpp
// FFT.h:39-43
enum { FLOAT16, FLOAT32_FFT, FLOAT32_ALL };
```

- `0 = FLOAT16`: the whole FFT pipeline (internal textures **and** LUTs) in half float.
- `1 = FLOAT32_FFT`: FFT in float32, **output** in half.
- `2 = FLOAT32_ALL`: everything in float32, output included.

Format mapping (`core/texture.cpp:42-178`, `GPUTYPES`):

| Logical type | D3D9 format |
|---|---|
| `HALF4` | `D3DFMT_A16B16G16R16F` |
| `FLOAT4` | `D3DFMT_A32B32G32R32F` |
| `HALF2` | `D3DFMT_G16R16F` |
| `FLOAT2` | `D3DFMT_G32R32F` |

Format choices:

| Object | RADIX2LUT | STOCKHAM |
|---|---|---|
| Working textures | `FFT.cpp:100-111`: `precision>0` → FLOAT*, otherwise HALF* | `fft2.cpp:36`: `precision > FLOAT16 ? FLOAT4 : HALF4` |
| LUTs (twiddle, load, save, C2R, R2C) | same rule (`FFT.cpp:165-172`) — therefore **in half when precision=0** | n/a (twiddles in the vertices, always float32) |
| Output pool | `FFT.cpp:152`: `precision != FLOAT32_ALL ? HALF4 : FLOAT4` | `fft2.cpp:65`: same |

Conversion on upload: `UploadToTexture` converts the CPU `float` values into `D3DXFLOAT16` when the texture is half (`core/texture.cpp:891-903`).

**Key point: there is no precision macro anywhere in the HLSL.** All pixel shader computation is done in `float32` (HLSL SM3.0), whatever the value of `precision`. The parameter changes **only the storage** (working textures, LUTs, output). With `precision=0`, the twiddles themselves are rounded to fp16 (about 3 decimal digits of precision) even before entering the shader.

Override: if `bits_per_pixel > 8`, the plugin forces `FLOAT32_ALL` (`FFT3dGPU.cpp:244-246`).

---

## 9. MEASURE mode

`FFT3dGPU.cpp:103-127`: `MeasureFFT` creates both implementations (`FFT2dRR` and `FFT2dRR2`), runs a number of `CalcFFT` calls on test textures, times them, and keeps the fastest one; the choice is exposed through the `fftnode` argument (`MEASURE`/`RADIX2LUT`/`STOCKHAM`, dispatch `FFT3dGPU.cpp:363-387`). This is a **pure GPU benchmark** — there is no CPU fallback implementation.

---

## 10. Numerical reproducibility

Exhaustive list of the factors that determine the bits of the result. For a "bit-exact" OpenCL port, each one must be reproduced or consciously set aside.

### 10.1 Order of additions

- **RADIX2LUT**: decimated-in-time radix-2, stage order `m = 2, 4, 8, …` (`CreateTwiddleMap` `FFT.cpp:748`), butterflies `i ± j·W` in the order of the pairs given by `LoadMap`/`SaveMap`. Each output is the sum of only **two** terms at each stage — the order is entirely fixed by the LUTs.
- **STOCKHAM**: stages `1 … log2n-1` with stride `size = n/2^stage` (`core/vertexbuffer.cpp:2756`), output `a+b` then `(a−b)·W`.
- The two algorithms **do not produce the same bits** as each other (different accumulation order) — the `fftnode` choice changes the numeric result, not just the speed.

### 10.2 Origin of the twiddles (the largest source of divergence)

| Case | Origin | Precision |
|---|---|---|
| RADIX2LUT stages | CPU **NR recurrence** (`FFT.cpp:750-763`), error accumulated along the stage | double → float32 (or **half** when precision=0) |
| RADIX2LUT C2R/R2C | direct CPU `cos(2πi/N)`, `sin(2πi/N)` (`FFT.cpp:810-811, 816-817`) | double → float32 (or half) |
| STOCKHAM stages ≥ 1 | CPU `cos(-pi*m/n)`, `sin(-pi*m/n)` (`core/vertexbuffer.cpp:2774`) | double → float32 (vertex buffer) |
| STOCKHAM stage 0 | GPU **`sincos`** on an interpolated angle 0→−π (`fft3dgpu.hlsl:1122`) | hardware-dependent |
| STOCKHAM C2Rfft/R2Cfft | GPU **`sincos`** on an interpolated angle (`fft3dgpu.hlsl:1164, 1198`) | hardware-dependent |

⚠ HLSL `sincos` is compiled into hardware instructions (often MUFU `sin/cos` on NVIDIA, accuracy better than 1 ulp not guaranteed); OpenCL `sincosf`/`sin`/`cos` is **not** the same. For cross-vendor reproducibility, replace it with CPU pre-computed double-precision tables (as STOCKHAM stages ≥ 1 and all RADIX2LUT LUTs already do).

⚠ The NR recurrence (`FFT.cpp:762-763`) must be copied **exactly**, including the variable reuse `(wtemp = wr)` that is part of the formula.

### 10.3 Storage precision

- `precision=0`: **all** LUTs (twiddles, load, save, C2R, R2C) are stored as `A16B16G16R16F`/`G16R16F` — the twiddles are rounded to fp16 before use. This is the historical default mode (`bits_per_pixel ≤ 8`).
- STOCKHAM vertices are always float32 (the vertex buffer is never half).
- Output: half except in `FLOAT32_ALL`.

### 10.4 Frozen constants and factors

- `−0.8660254` (truncated sin(4π/3)) in `FFT3p`/`iFFT3p` (`fft3dgpu.hlsl:491, 530`).
- the `0.5` of the C2R/R2C formulas and the `0.5` of STOCKHAM R2Cfft (`fft3dgpu.hlsl:1167-1168`), compensated by the `2.0/(x·y·z)` of `CombineH` (`fft2.cpp:25, 62`).
- `norm = 1.0/(x*y*bt)` RADIX2LUT (`FFT.cpp:95`), compiled into the `BitReverseButterFlyBH` shader.
- STOCKHAM angles use a double `pi` on the CPU side (`const float pi(acos(-1.0))`, `fft2ps.cpp:3`; global `pi` in vertexbuffer.cpp).

### 10.5 Sampling and addressing

- **POINT** sampler everywhere in the FFTs (coordinates are aligned to the center of the texels, §3.1). An OpenCL port must use equivalent integer indices; any rounding difference in the coordinate computation (`(i+0.5)/W` in float) can shift by one texel — replace it with integer arithmetic.
- **MIRROR** addressing at runtime (`FFT3dGPU.cpp:816-820`): only relevant for out-of-range reads (should not happen with correct LUTs; to be checked in the port).

### 10.6 Forward/inverse sign

- RADIX2LUT: two separate twiddle tables (`isign = ±1`, `FFT.cpp:745`).
- STOCKHAM: same geometry, `dir` constant (`fft2ps.cpp:49-50, 103-104`) that flips either the sin value (VS `fft2`) or the angle (VS `fft0`).

---

## Appendix A — HLSL shader summary

| Shader (`fft3dgpu.hlsl`) | Line | Algo | Role |
|---|---|---|---|
| `Img2toFloat4` / `_2` | 113-135 / 165-183 | input | image packing → float4 (outside the FFT) |
| `Float4toImg2` / `_2` | 137-161 / 185-198 | output | unpacking (outside the FFT) |
| `BitReverseButterFlyH` | 202-229 | RADIX2LUT fwd H | bitrev + butterfly 1 |
| `BitReverseButterFlyBH` | 233-259 | RADIX2LUT inv H | same + ×NORM |
| `BitReverseButterFlyV` | 264-290 | RADIX2LUT H+V | same, vertical |
| `C2R` | 294-312 | RADIX2LUT inv | real → complex |
| `R2C` | 314-334 | RADIX2LUT fwd | complex → real |
| `ButterflyCollectH` | 338-376 | RADIX2LUT | butterfly stages 1+ (H) |
| `ButterflyCollectV` | 380-418 | RADIX2LUT | butterfly stages 1+ (V) |
| `CollectH` / `CollectV` | 421-437 / 439-454 | RADIX2LUT | merge I/J → 1 texture |
| `FFT2p` / `iFFT2p` | 456-472 / 474-479 | temporal | 2-point FFT |
| `FFT3p` / `iFFT3p` | 481-519 / 521-535 | temporal | 3-point FFT |
| `fft2` (VS) | 1068-1073 | STOCKHAM | sin sign flip (dir) |
| `fft0` (VS) | 1075-1080 | STOCKHAM | angle sign flip (dir) |
| `passthough3` / `passthough` | 1082-1085 / 1087-1090 | STOCKHAM | identity VS |
| `Butterfly` | 1114-1143 | STOCKHAM | butterfly (STAGE0 = GPU sincos) |
| `Combine` | 1152-1154 | STOCKHAM | ×NORM |
| `R2Cfft` | 1156-1171 | STOCKHAM inv | real → complex |
| `C2Rfft_ps` | 1177-1204 | STOCKHAM fwd | complex → real (t00/t11) |

## Appendix B — Vertex geometry summary

| Class (`core/vertexbuffer.cpp`) | Line | Algo | Encoding |
|---|---|---|---|
| `NQuad` | 40-169 | RADIX2LUT | full-screen quad, texCoord = (i+0.5)/W |
| `C2RQuad` | 2249-2476 | STOCKHAM fwd | 5 quads/block, `2πi/(2x)` angles per corner, ±1 for DC/Nyquist |
| `CollectQuad` | 2478-2594 | STOCKHAM | odd/even gathering for `Combine` |
| `R2CFFTQuad` | 2598-2715 | STOCKHAM inv | interpolated angles 0→π, mirror pairs |
| `FFTQuad2` | 2723-2867 | STOCKHAM | quads per stage, `cos/sin(-πm/n)` twiddles per vertex, stageLUT |
| `FFTQuad0` | 2871-2983 | STOCKHAM | stage 0, 0/−π angles per corner |

## Appendix C — DX9 → OpenCL mapping

| DX9 concept | Suggested OpenCL equivalent |
|---|---|
| Pixel shader + full-screen quad | 2D kernel, one work-item per output texel |
| LUT texture + `tex2D` (POINT, +0.5) | `__global`/`__constant` array, integer index |
| MRT (2 outputs) | the work-item writes the 2 outputs (or 2 identical kernels) |
| Vertex buffer (interpolated twiddles) | twiddle table in memory, computed index (note: linear interpolation of the angles across the C2RQuad/R2CFFTQuad/FFTQuad0 quads = exact value at the centers, so a direct `i` → angle table) |
| HLSL `sincos` | CPU double→float table (reproducibility) or `sincosf` (fast, not reproducible) |
| `D3DFMT_A16B16G16R16F` | `half` (cl_khr_fp16 extension) or float + explicit half rounding |
| MIRROR sampler | not needed if indices are validated |
| HLSL runtime compilation | pre-compiled kernels + specialization via `#define` (NORM, PIXELOFFSET become parameters) |

**Porting recommendations**:

1. Port both algorithms (RADIX2LUT and STOCKHAM): they are not numerically equivalent, and the historical filter uses one or the other depending on `fftnode`.
2. Copy the twiddle NR recurrence verbatim (`FFT.cpp:750-763`) if the goal is bit-exactness with the original plugin in RADIX2LUT.
3. Remove the GPU `sincos` calls (STOCKHAM stage 0, C2Rfft, R2Cfft) by pre-computing on the CPU the same values as the `cos/sin(double)` of the original vertex buffer — this is already the case for STOCKHAM stages ≥ 1.
4. Keep the order of the stages and of the pairs (LUTs) exactly as it is; do not "optimize" the factorization.
5. Freeze the constants: `−0.8660254`, `0.5`, `2.0/(x·y·z)`, `1/(x·y·bt)`.
6. Support the fp16 mode (precision=0): round the LUTs **and** the working textures to half at storage time, not during computation.
