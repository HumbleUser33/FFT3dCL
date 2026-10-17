# FFT3dGPU — Exhaustive technical documentation of the CPU classes driving the HLSL pixel shaders (OpenCL port)

Sources analysed: `FFT3dGPU/FFT3dGPU/filtersps.h` (418 lines), `filtersps.cpp` (2100 lines), `filters.h` (126 lines), `filters.cpp` (1001 lines), `fft3dgpu.hlsl` (1205 lines), `FFT3dGPU.cpp/.h`, `core/shader.{h,cpp}`, `core/texture.{h,cpp}`, `core/vertexbuffer.{h,cpp}`, `core/ManageDirect3DWindow.cpp`.
All references are `file:line`.

---

## 0. Cross-cutting mechanisms (essential for the port)

### 0.1 Shader compilation
- Single HLSL file: `#define SRC_SHADER "fft3dgpu.hlsl"` (`core/shader.h:28`), resolved to an absolute path next to the DLL (`core/shader.cpp:54-68`), compiled at runtime by `D3DXCompileShaderFromFile` with the macro table `defs` (`core/shader.cpp:68`).
- Each class passes different **preprocessor defines** → the same file produces variants. For OpenCL: one source, several `-D`.
- Macro helpers (`core/shader.cpp`):
  - `VecToMacroArray(vec,name)` → `#define name float2(x,y)` (`shader.cpp:136-146`, `std::fixed` format)
  - `FloatToMacroArray(f,name)` → `#define name f` (`shader.cpp:194-204`, `std::fixed`)
  - `IntToMacroArray(i,name)` → `#define name i` (`shader.cpp:165-175`)
  - `SetMacroArray(name)` → `#define name` (empty) (`shader.cpp:221-227`)
  - `CreateMacroArray(size)` → array of `size+1` entries initialized to 0 (`shader.cpp:253-262`)
- Trap warning: when a define is not wanted, the code sometimes passes `" "` as the macro **name** (`filtersps.cpp:88`, `238`) — a define named space, harmless in HLSL.

### 0.2 HLSL sampler registers (fixed)
`fft3dgpu.hlsl:44-50`:
```
sampler Src : register(s0);
sampler Src2 : register(s2);
sampler I : register(s3);
sampler J : register(s4);
sampler Factor : register(s1);
sampler FactorM : register(s5);
sampler LoadMap : register(s6);
```
Wienner (`hlsl:636-641`): `prev2`, `prev1` unregistered (assigned by the compiler), `current : s3`, `next : s4`, `degrid : s1`, `sigma` unregistered.
Kalman (`hlsl:782-785`): `Last : s1`, `Covar : s2`, `CovarProcess : s3`, `CovarNoise : s4`.
The actual stage index is read dynamically via `GetSamplerIndex` (`core/shader.cpp:106-117`).

### 0.3 Sampler state (to reproduce in OpenCL)
- At device creation, stages 0..5: `ADDRESSU=ADDRESSV=D3DTADDRESS_BORDER`, `BORDERCOLOR=0x00000000` (`core/ManageDirect3DWindow.cpp:221-225`).
- **At the start of each frame**, stages 0..5 are reset to `ADDRESSU=ADDRESSV=D3DTADDRESS_MIRROR` (`FFT3dGPU.cpp:816-820`, same in the constructor for the degrid phase: `FFT3dGPU.cpp:533-537`).
- Filtering: no `MINFILTER/MAGFILTER` is ever set in production code (D3D9 default = **POINT**). Only the commented-out `psMeanSD` code touched `MINFILTER` (`filtersps.cpp:858,869`). Therefore **all tex2D use point filtering**, unless stated otherwise.
- Temporary exceptions in the Img2ToFloat4/Float4ToImg2 "ohalf" classes: the `Factor` stage switches to `ADDRESSU=WRAP` during the draw then returns to `MIRROR` (`filtersps.cpp:1239/1246`, `1332/1360`, `1414/1427`, `1516/1549`, `1911/1918`, `2076/2094`). ADDRESSV stays MIRROR.
- OpenCL equivalent: `CLK_ADDRESS_MIRRORED_REPEAT` + `CLK_FILTER_NEAREST` everywhere, with `CLK_ADDRESS_CLAMP_TO_EDGE` + border 0 at device startup, and `CLK_ADDRESS_REPEAT` on the U axis of the factor LUT for the "ohalf" variants.

### 0.4 Texture formats (`GPUTYPES`, `core/texture.cpp:42-193`)
| Logical type | D3D format (RenderTarget / Managed) | Elements |
|---|---|---|
| FLOAT4 | `D3DFMT_A32B32G32R32F` (`texture.cpp:51-54,118-121`) | 4×float32 |
| FLOAT2 | `D3DFMT_G32R32F` (`texture.cpp:56-59,123-131`) | 2×float32 |
| FLOAT1 | `D3DFMT_R32F` (`texture.cpp:61-64,129-131`) | float32 |
| HALF4 | `D3DFMT_A16B16G16R16F` (`texture.cpp:66-69,133-136`) | 4×float16 |
| HALF2 | `D3DFMT_G16R16F` (`texture.cpp:71-74,138-146`) | 2×float16 |
| HALF1 | `D3DFMT_R16F` (`texture.cpp:76-79,143-146`) | float16 |
| FIXED4 (8 bit) | `D3DFMT_A8R8G8B8` (`texture.cpp:82-85,149-152`) | 4×uint8 normalized 0..1 |
| FIXED2 (8 bit) | `D3DFMT_A8L8` (`texture.cpp:88-91,154-157`) | 2×uint8 |
| FIXED1 (8 bit) | `D3DFMT_L8` (`texture.cpp:93-96,159-162`) | uint8 |
| FIXED4_16 (10-16 bit) | `D3DFMT_A16B16G16R16` (`texture.cpp:99-102,175-178`) | 4×uint16 |
| FIXED2_16 | `D3DFMT_G16R16` (`texture.cpp:107-110,180-183`) | 2×uint16 |
| FIXED1_16 | `D3DFMT_L16` (`texture.cpp:112-115,185-188`) | uint16 |
Fallbacks when the format is not supported as a RT (e.g. A8L8 RT rejected, see comment `texture.cpp:87`): it falls back to the 4-component format. `types` carries `elem_size` (number of logical components) and `real_elem_size` (physical components) (`core/texture.h:40-58`).

Selection according to bit depth (`FFT3dGPU.cpp:292-345`, `filters.cpp:65-74`): `bits_per_pixel==8` → FIXED2/FIXED4; `<=16` → FIXED2_16/FIXED4_16; otherwise (float) → FLOAT2/FLOAT4.
FFT working textures (`FFT3dGPU.cpp:348-358`): `img`/`imgp` as **HALF4** if `useHalf` (precision==FLOAT16) otherwise **FLOAT4**, size `totw/2 × toth` with `totw=nx*bw`, `toth=ny*bh` (`FFT3dGPU.cpp:284-285`).
`precision`: enum `FLOAT16=0, FLOAT32_FFT=1, FLOAT32_ALL=2` (`FFT.h:40-42`); **if bits_per_pixel>8, precision is forced to FLOAT32_ALL** (`FFT3dGPU.cpp:245-246`).

### 0.5 Coordinate mapping (NQuad) — critical point of the port
`NQuad(pDevice, RECT* Rect, int size, int RectOffset=0, bool IndependentOffset=false)` (`core/vertexbuffer.h:179`, implementation `core/vertexbuffer.cpp:61-110`):
- `Rect[0]` = rectangle of the **render target** (the quad is drawn 1:1 in screen pixels over this rectangle).
- For each stage `i` from 0 to `size-1`, the texture coordinates come from `Rect[i+RectOffset]`:
```
W  = Rect[i+RectOffset].right - left ; H = ...
etaW = 0.5/W + left/W          // correction demi-texel
etaH = 0.5/H + top/H
u: etaW → etaW + W0/W          (W0,H0 = taille du quad, ou W,H si IndependentOffset)
v: etaH → etaH + H0/H
```
(`vertexbuffer.cpp:88-99`). This is what encodes the offsets from a source pixel to another texture. A pixel `offset` is achieved by translating the source RECT (`rect[2].left += offset.x` etc.).
- `pTextureRTpair` = pair of `TextureRT*` (`first`,`last`) with a reference counter (`core/texture.h:209-244`); used in mode 1 (2 block textures instead of one).

### 0.6 HLSL macros common to the conversions
- `SW1` (choice of the components read from a 2-component texture) `hlsl:95-104`:
```
#if defined(G16R16) || defined(G32R32F)  → SW1 = rg
#elif defined(A8L8)                      → SW1 = xw
#else                                     SW1 = xy
```
- CPU macro `GETPIXEL2TYPEMACRO` (`filtersps.cpp:335`):
```
(src->GetType()->format==D3DFMT_A8L8 || ==D3DFMT_A8R8G8B8 ? "A8L8"
 : ==D3DFMT_G16R16 ? "G16R16" : ==D3DFMT_G32R32F ? "G32R32F" : "A8L8")
```
Set only if `real_elem_size==2 || (real_elem_size==4 && elem_size==2)` (texture with 2 logical components).
- `BIT_DEPTH` (int) and 10-14 bit normalization (`hlsl:35-39`):
```
#if BIT_DEPTH>8 && BIT_DEPTH<16
BITDEPTHNORM     = float4(1<<(16-BIT_DEPTH), ...)      // multiplication à l'entrée
INV_BITDEPTHNORM = float4(1.0/(1<<(16-BIT_DEPTH)),...) // division à la sortie
MAX_PIXEL_VALUE  = float4((1<<BIT_DEPTH)-1, ...)       // clamp min(dst, MAX) à la sortie
```
- `CHROMA_BIT_DEPTH`/`CHROMANORM` (`hlsl:21-33`): YUY2 chroma offset correction; `CHROMA_CORR = RANGE_HALF/RANGE_FULL` = `32768/65535` if bit depth ≤16, `0` otherwise; `CHROMANORM = float4(c,c,c,c)`. Enabled on the CPU side by the boolean `chroma` = `!isLumaPlane && !isRGB` (`FFT3dGPU.cpp:462`).

---

## 1. psMinimize (the degrid "selectDC" class)

**Declaration** `filtersps.h:24-30` — **Implementation** `filtersps.cpp:27-49`.

1. **Shader**: HLSL function `Minimize` (`hlsl:617-620`), **no defines**:
```hlsl
float4 Minimize( PS_INPUT In) :COLOR
{
	return tex2D(Src,In.texCoord).xyxy;
}
```
2. **Inputs**: 1 texture `Src` (stage via `GetConstantByName("Src")`, `filtersps.cpp:29`). This is the full FFT texture `FImg2d[i]`/`ping`/`FImgDegridp.first`: FLOAT4 or HALF4, `totw/2 × toth`. Sampler: MIRROR/MIRROR, point.
3. **Output**: `mintex` = `TextureRT` **FLOAT4**, size **nx × ny** (`FFT3dGPU.cpp:516`) — one pixel per block, containing the block's DC value (the .xy components of the FFT texture at each block's corner, duplicated as xyxy).
4. **Constants**: none.
5. **Conditions**: created only if `degrid != 0` (`FFT3dGPU.cpp:514-520`, named `selectDC`).
6. **Call order**: before `calcgridcorrection` on each frame (`FFT3dGPU.cpp:547, 856, 896, 898, 947, 1001, 1003`); in the constructor, once on frame 0 (`FFT3dGPU.cpp:547`).
7. **Geometry**: `NQuad(rec[2]={dst->GetRect(), src->GetRect()}, 1, 1, true)` (`filtersps.cpp:31-32`): 1 stage, RectOffset=1, IndependentOffset=true → the nx×ny quad samples the FFT texture at block→pixel scale.

---

## 2. psGridCorrection (degrid)

**Declaration** `filtersps.h:32-40` — **Implementation** `filtersps.cpp:51-83`.

1. **Shader**: HLSL function `GridCorrection` (`hlsl:622-632`):
```hlsl
float4 GridCorrection( PS_INPUT In) :COLOR
{
	float4 src=tex2D(Src,In.texCoord);
	float4 factor=tex2D(Factor,In.texCoord);
	#ifdef DGRID
	float4 dst=src/factor.xyxy*DGRID;
	#else
	float4 dst=src*factor;
	#endif
	return dst;
}
```
Two compiled variants:
- without `DGRID`: the `calcgridcorrection` instance (`FFT3dGPU.cpp:518`, ctor `filtersps.cpp:51-57`);
- with `DGRID` = the script's float `degrid` (`FloatToMacroArray(degrid,"DGRID",...)`, `filtersps.cpp:59-66`): the `setupgridcorrection` instance, used only once in the constructor (`FFT3dGPU.cpp:519, 548, 557, 570`).
2. **Inputs**:
   - `Src`: either the FFT texture (setup: `FImgDegrid` = FFT of frame 0; runtime: `mintex` nx×ny FLOAT4),
   - `Factor`: either `mintex` (nx×ny FLOAT4) at setup, or `gridsample` = `TextureM` **FLOAT4** `temp->GetWidth()×GetHeight()` downloaded/uploaded on the CPU in the constructor (`FFT3dGPU.cpp:560-565`).
3. **Output**: `TextureRT` from the pool (FLOAT4/HALF4 `totw/2×toth`) → `FImgDegrid` / `FImgDegridp.first|last`: **grid pattern to subtract** before filtering.
4. **Constants**: `DGRID` = the script parameter `degrid` (default 1.0 if mode==1 otherwise 0.0, `FFT3dGPU.cpp:1267`).
5. **Conditions**: only `degrid != 0`.
6. **Order**: `selectDC` (Minimize) → `calcgridcorrection` → then the result `FImgDegrid` is passed to the Wienner and the Sharpen (`FFT3dGPU.cpp:855-859, 894-903, 945-954, 1000-1007`).
7. **Geometry**: `NQuad(&target->GetRect(), 1, 0, false)` (`filtersps.cpp:56,65`) — same RECT for the target and the source (RectOffset=0).

---

## 3. psWiennerFilter (Wienner2d / Wienner3d2 / Wienner3d3 / Wienner3d4)

**Declaration** `filtersps.h:43-55` — **Implementation** `filtersps.cpp:86-233`.

### 3.1 Variant selection and defines
Constructor (`filtersps.cpp:86-110`):
```cpp
Pixelshader(pDevice, SRC_SHADER, n == 1 ? "Wienner2d" : n == 2 ? "Wienner3d2" : n == 3 ? "Wienner3d3" : "Wienner3d4",
  D3DXGetPixelShaderProfile(pDevice),
  SetMacroArray(degrid ? "DEGRID" : " ",
    VecToMacroArray(sigma,  pattern ? "" : "SIGMA",
      VecToMacroArray(beta, "BETA", CreateMacroArray(3)), 1), 2))
```
Defines: `BETA=float2(beta2)`, `SIGMA=float2(sigma2)` (absent if `pattern`), `DEGRID` (if degrid). The `#ifdef BETA` block encloses all the Wienner HLSL code (`hlsl:634-778`).

### 3.2 CPU constants (exact formulas)
`FFT3dGPU.cpp:172`: `sigma = _sigma / 255` (the script parameter is scaled to 0..1).
`FFT3dGPU.cpp:396-398`:
```cpp
float s = sigma * sigma*bt*bw*bh;
sigma2 = D3DXVECTOR2(beta*s, s);                 // SIGMA.x = beta*s ; SIGMA.y = s
beta2  = D3DXVECTOR2((beta - 1.0) / beta, beta); // BETA.x = (beta-1)/beta ; BETA.y = beta
```
Created if `bm > 0` (bt∈{1..4}): `wiennerfilter = NEW psWiennerFilter(pDevice, pFreeFImgdPool->top(), beta2, sigma2, bt, degrid != 0, usePattern)` (`FFT3dGPU.cpp:406-407`).
`usePattern = _sigma != _sigma2 || _sigma3 != _sigma4 || _sigma != _sigma3` (`FFT3dGPU.cpp:175`).

### 3.3 The "sigma" texture (noise pattern) when usePattern
`SigmaToPatternTexture` (`FFT3dGPU.cpp:588-631`), CPU floats:
```cpp
_sigma2 *= _sigma2 * bt*bw*bh / (255*255);   // idem sigma3, sigma4  (FFT3dGPU.cpp:401-403)
float ft2 = sqrt(0.5f) / 2;   // fréquence de coupure sigma2
float ft3 = sqrt(0.5f) / 4;   // fréquence de coupure sigma3
fy = (bh - 2.0f*abs(h - bh/2.0f)) / bh;      // h = 0..bh-1
fx = (w*1.0f) / fbw;  fbw = bw/2+1;
f  = sqrt((fx*fx + fy*fy)*0.5f);
f < ft3 : pattern = sigma4 + (sigma3-sigma4)*f/ft3
f < ft2 : pattern = sigma3 + (sigma2-sigma3)*(f-ft3)/(ft2-ft3)
sinon   : pattern = sigma  + (sigma2-sigma )*(1-f)/(1-ft2)
```
Then tiled `nx×ny` and uploaded into `Pattern` = `TextureM` **FLOAT1** (`gtype->FLOAT()`), size `fbw*nx × bh*ny` (`FFT3dGPU.cpp:597`). Passed as the `sigma` sampler (unregistered).

### 3.4 HLSL functions and formulas
Shared helper `Wienner(src,texCoord)` (`hlsl:643-664`):
```hlsl
#ifdef SIGMA
float4 MulFac=float4(BETA.x,BETA.x,BETA.x,BETA.x);
float2 PSDInv=1/(float2(src.x*src.x+src.z*src.z,src.y*src.y+src.w*src.w)+0.0000000000000000000000000000000000001);
if(1/SIGMA.x>PSDInv.x)
	MulFac.xz=1.0-SIGMA.y*PSDInv.x;
if(1/SIGMA.x>PSDInv.y)
	MulFac.yw=1.0-SIGMA.y*PSDInv.y;
return MulFac*src;
#else
float4 MulFac=float4(BETA.x,BETA.x,BETA.x,BETA.x);
float2 Sigma=float2(tex2D(sigma,texCoord).x*BETA.y,tex2D(sigma,texCoord).x);
float2 PSDInv=1/(float2(src.x*src.x+src.z*src.z,src.y*src.y+src.w*src.w)+0.0000000000000000000000000000000000001);
if(1/Sigma.x>PSDInv.x)
	MulFac.xz=1.0-Sigma.y*PSDInv.x;
if(1/Sigma.x>PSDInv.y)
	MulFac.yw=1.0-Sigma.y*PSDInv.y;
return MulFac*src;
#endif
```
**Wienner2d** (`hlsl:666-686`): `src=tex2D(current)`; if DEGRID `correction=tex2D(degrid)`, `src-=correction`; `retval=Wienner(src)`; if DEGRID `retval+=correction`.

**Wienner3d2** (`hlsl:690-710`) — 2-image temporal Hadamard:
```hlsl
float4 src1=tex2D(prev1,In.texCoord);
float4 src2=tex2D(current,In.texCoord);
dst[0]=src2-src1;
dst[1]=src2+src1;
#ifdef DEGRID
float4 factor=tex2D(degrid,In.texCoord)*2;
dst[1]=dst[1]-factor;
#endif
src1=Wienner(dst[0],In.texCoord);
src2=Wienner(dst[1],In.texCoord);
retval=src2 + src1;
#ifdef DEGRID
retval=retval+factor;
#endif
```
**Wienner3d3** (`hlsl:712-748`) — 3-point DFT:
```hlsl
float4 p=tex2D(prev1); float4 c=tex2D(current); float4 n=tex2D(next);
#ifdef DEGRID
float4 factor=tex2D(degrid)*3;
#endif
float4 t1=p+n;
float4 y1=c-0.5*t1;
//-0.8660254=sin(pi*4/3)
float4 y2;
y2.zwxy=float4(-0.8660254,-0.8660254,0.8660254,0.8660254)*(n-p);
dst[0]=y1+y2;
dst[1]=c+t1;
dst[2]=y1-y2;
#ifdef DEGRID
dst[1]=dst[1]-factor;
#endif
p=Wienner(dst[0]); c=Wienner(dst[1]); n=Wienner(dst[2]);
retval=p+c+n;
#ifdef DEGRID
retval=retval+factor;
#endif
```
**Wienner3d4** (`hlsl:751-777`, "modified from Fizick fft3dfilter") — 4-point DFT:
```hlsl
float4 p2=tex2D(prev2); float4 p1=tex2D(prev1); float4 c=tex2D(current); float4 n=tex2D(next);
#ifdef DEGRID
float4 factor=tex2D(degrid)*4;
#endif
float4 fp1 = -p2 + float4(1.0,1.0,-1.0,-1.0)*p1.zwxy + c + float4(-1.0,-1.0,1.0,1.0)*n.zwxy;
float4 fc  = p2 + p1 + c + n;
float4 fn  = -p2 + float4(-1.0,-1.0,1.0,1.0)*p1.zwxy + c + float4(1.0,1.0,-1.0,-1.0)*n.zwxy;
float4 fp2 = p2 - p1 + c - n;
#ifdef DEGRID
fp1=fp1-factor;
#endif
fp2=Wienner(fp2); fp1=Wienner(fp1); fc=Wienner(fc); fn=Wienner(fn);
retval=fp2+fp1+fc+fn;
#ifdef DEGRID
retval=retval+factor;
#endif
```
(Do not forget: the inverse transform has **no 1/N factor** — the normalization is absorbed elsewhere.)

### 3.5 Textures, samplers, mapping
- CPU handles (`filtersps.cpp:91-108`, beware of the switch *fall-through*):
  - n=4: `sSrc[3]="prev2"`, `sSrc[2]="prev1"`, `sSrc[1]="current"`, `sSrc[0]="next"`;
  - n=3: `sSrc[2]="prev1"`, `sSrc[1]="current"`, `sSrc[0]="next"`;
  - n=2: `sSrc[1]="prev1"`, `sSrc[0]="current"`; n=1: `sSrc[0]="current"`.
  - `sFactor="degrid"` if degrid; `sSigma="sigma"` if pattern.
- All input/output textures are pool `TextureRT` **FLOAT4/HALF4 `totw/2×toth`**; `degrid` likewise; `sigma` = `Pattern` FLOAT1.
- Vector→samplers mapping (`filtersps.cpp:144-148, 170-174, 187-191`): `for (i=0, j=size-1; j>=0; i++,j--) src[i] → sSrc[j]` — the vector is ordered **oldest→newest** (`FImg2d[i]` = frame `n-current+i`, `FFT3dGPU.cpp:829-853`), so `src[0]` (the oldest) goes to `sSrc[size-1]` = `prev2`/`prev1`, and the newest to `next`. `current = bm==1?0 : bm==4?2 : 1` (`FFT3dGPU.cpp:176`).
- 4 `Apply` overloads:
  - `Apply(Texture* src, TextureRT* dst, degrid, sigma)` — bt=1, mode 0/2 (`filtersps.cpp:114-134`);
  - `Apply(vector<TextureRT*>, TextureRT* dst, ...)` — bt≥2, mode 0/2 (`136-159`);
  - `Apply(vector<pTextureRTpair*>, pTextureRTpair* dst, ...)` — bt≥2, mode 1: **two draws** (on `dst->first` then `dst->last`) (`162-200`);
  - `Apply(pTextureRTpair*, pTextureRTpair*, ...)` — bt=1, mode 1: two draws (`202-233`).
- Geometry: `NQuad(&_Rendertarget->GetRect(), 1)` (`filtersps.cpp:109`) — RectOffset=0, source=target.

### 3.6 Position in the pipeline (GetFrame)
- bt≥2: FFT of the bt frames (`GPUCache` cache) → optional degrid → `wiennerfilter->Apply(FImg2d…, FImg1d[current], FImgDegrid, Pattern)` → optional sharpen → iFFT (`FFT3dGPU.cpp:823-914`).
- bt=1 (bm==1): FFT → degrid → `wiennerfilter->Apply(ping, pong, FImgDegrid, Pattern)` → swap → sharpen → iFFT (`FFT3dGPU.cpp:952-971`, mode 1: `1010-1031`).

---

## 4. psKalman (abstract), psKalmanSP, psKalmanMP

**Declarations** `filtersps.h:58-89` — **Implementation** `filtersps.cpp:236-332`.

### 4.1 psKalmanSP (single-pass, MRT 3 outputs)
Constructor (`filtersps.cpp:236-247`): HLSL function **`Kalman`**, defines:
```cpp
FloatToMacroArray(covarNoise, pattern ? " " : "COVARNOISE",
  FloatToMacroArray(kratiosquared, "KRATIOSQUARED", CreateMacroArray(2)), 1)
```
→ `KRATIOSQUARED=float` always; `COVARNOISE=float` only **without** a pattern (otherwise it is a sampler).

HLSL (`hlsl:780-833`), guard `#ifdef KRATIOSQUARED`; samplers `Last:s1, Covar:s2, CovarProcess:s3, CovarNoise:s4`; if `COVARNOISE` is not defined: `#define COVARNOISE tex2D(CovarNoise,In.texCoord).x` (`hlsl:794-796`). Output structure `PS_RETVAL {COLOR0 dst, COLOR1 covar, COLOR2 covarprocess}` (`hlsl:787-792`). Exact formula:
```hlsl
PS_RETVAL Kalman(PS_INPUT In)
{
	PS_RETVAL retval;
	PS_RETVAL r_1;
	float4 covar=tex2D(Covar,In.texCoord);
	float4 covarprocess=tex2D(CovarProcess,In.texCoord);
	float4 src=tex2D(Src,In.texCoord);
	float4 last=tex2D(Last,In.texCoord);
	float4 mask=(src-last)*(src-last);
	mask.x=max(mask.x,mask.z);
	mask.y=max(mask.y,mask.w);

		r_1.dst=src;
		r_1.covar=COVARNOISE;
		r_1.covarprocess=COVARNOISE;

		float4 s=covar+covarprocess;
		float4 gain=s/(s+COVARNOISE);
		retval.covarprocess=gain*gain*COVARNOISE;
		retval.covar=(1-gain)*s;
		retval.dst=gain*src+(1-gain)*last;
	if(mask.x>KRATIOSQUARED*COVARNOISE){
		retval.dst.xz=r_1.dst.xz;
		retval.covar.xz=r_1.covar.xz;
		retval.covarprocess.xz=r_1.covarprocess.xz;
	 }
	if(mask.y>KRATIOSQUARED*COVARNOISE)
	{
		retval.dst.yw=r_1.dst.yw;
		retval.covar.yw=r_1.covar.yw;
		retval.covarprocess.yw=r_1.covarprocess.yw;
	 }
	return retval;
}
```
`Apply` (`filtersps.cpp:249-277`): 3 simultaneous render targets — `dst` (RT0), `covar` (RT1), `covarprocess` (RT2); textures `Src`, `Last`, `CovarProcess`(←`covarprocesslast`), `Covar`(←`covarlast`); with a pattern, `CovarNoise`←`pattern` texture. After the draw: `SetRenderTarget(1,0); SetRenderTarget(2,0)` (`filtersps.cpp:274-275`).

### 4.2 psKalmanMP (multi-pass, graphics cards with a single RT — old Riva128)
Constructor (`filtersps.cpp:279-294`): 3 instances of the **`KalmanMP`** function:
- `pass1`: define `PASS1_K` + `COVARNOISE`(if !pattern) + `KRATIOSQUARED`;
- `pass2`: define `PASS2_K` + same;
- `pass3`: no PASS define (just `COVARNOISE`/`KRATIOSQUARED`).

HLSL (`hlsl:839-877`) — comment `//PASS1=dst //PASS2=covar //PASS3=covarprocess`:
```hlsl
float4 KalmanMP(PS_INPUT In):COLOR
{
	float4 retval;
	float4 r_1;
	float4 covar=tex2D(Covar,In.texCoord);
	float4 covarprocess=tex2D(CovarProcess,In.texCoord);
	float4 src=tex2D(Src,In.texCoord);
	float4 last=tex2D(Last,In.texCoord);
	float4 mask=(src-last)*(src-last);
	mask.x=max(mask.x,mask.z);
	mask.y=max(mask.y,mask.w);

		#ifdef PASS1_K
		r_1=src;
		#else
		r_1=COVARNOISE;
		#endif

		float4 s=covar+covarprocess;
		float4 gain=s/(s+COVARNOISE);

		#ifdef PASS1_K
		retval=gain*src+(1-gain)*last;
		#elif defined(PASS2_K)
		retval=(1-gain)*s;
		#else
		retval=gain*gain*COVARNOISE;
		#endif
	if(mask.x>KRATIOSQUARED*COVARNOISE){ retval.xz=r_1.xz; }
	if(mask.y>KRATIOSQUARED*COVARNOISE){ retval.yw=r_1.yw; }
	return retval;
}
```
`Apply` (`filtersps.cpp:296-332`): draw1 → RT=`dst` (result), draw2 → RT=`covar` with `pass2`, draw3 → RT=`covarprocess` with `pass3`. The bound textures do not change between the 3 draws.

### 4.3 CPU constants
- `ps = NEW psKalmanSP(pDevice, StreamPoolPointer->top(), sigmaSquaredNoiseNormed2D, kratio*kratio, false)` if `Cap.NumSimultaneousRTs > 1`, otherwise `psKalmanMP` (`filters.cpp:734-746`).
- Without a pattern: `sigmaSquaredNoiseNormed2D = sigma*sigma*bw*bh` with `sigma=_sigma/255` (`FFT3dGPU.cpp:446-450`) — **careful: no bt factor** (unlike the Wienner).
- With a pattern: covarNoise hard-coded to 0, the `COVARNOISE` comes from the `Pattern` texture (`FFT3dGPU.cpp:433-441`, ctor `filters.cpp:749-762`).
- `kratio` script default = 2.0 (`FFT3dGPU.cpp:1290`).

### 4.4 KalmanFilter (CPU wrapper)
**Declaration** `filters.h:85-110` — **Implementation** `filters.cpp:734-911`.
- `Init(useTexturepair)` (`filters.cpp:765-813`): allocates `covar1/covar2` (+`covarprocess1/2`) from the pool or the `covar1d…` pairs; fills `covar1` and `covarprocess1` with `f[i] = _sigmaSquaredNoiseNormed2D` (float), or with the pattern: each texel of `_pattern` (FLOAT1) is replicated into the 4 components: `f[j]=f[j+1]=f[j+2]=f[j+3]=f1[i+x]` (`filters.cpp:774-776`).
- `Filter(Texture* src, TextureRT* dst)` (`filters.cpp:828-837`): on the first call `last=src`; `ps->Apply(src, last, covarprocess1, covar1, covarprocess2, covar2, dst, _pattern)`; then `swap(covar1,covar2)`, `swap(covarprocess1,covarprocess2)`, `last=dst` — **persistent ping-pong temporal state**.
- Pair version (`filters.cpp:839-856`): 2 `ps->Apply` calls (first then last).
- `Restore()` (`filters.cpp:858-911`): reinitializes covar/covarprocess (lost device) — same formulas as Init.
- Conditions: `bm == 0` (bt=0) only (`FFT3dGPU.cpp:431-453`); called as `kalman->Filter(ping, pong)` (`FFT3dGPU.cpp:959, 1017`).

---

## 5. psSharpen + Sharpen

**psSharpen**: `filtersps.h:389-397`, `filtersps.cpp:742-770`. **Sharpen**: `filters.h:112-125`, `filters.cpp:935-1001`.

### 5.1 Shader and defines
Constructor (`filtersps.cpp:742-751`): HLSL function **`Sharpen`**, defines:
```cpp
SetMacroArray(degrid ? "DEGRID" : " ",
  FloatToMacroArray(sigmaSquaredSharpenMax, "SSMAX",
    FloatToMacroArray(sigmaSquaredSharpenMin, "SSMIN", CreateMacroArray(3)), 1), 2)
```
HLSL (`hlsl:587-603`, guard `#ifdef SSMAX`):
```hlsl
float4 Sharpen( PS_INPUT In) :COLOR
{
float4 src=tex2D(Src,In.texCoord);
float factor=tex2D(Factor,In.texCoord).x;
#ifdef DEGRID
float4 degrid=tex2D(FactorM,In.texCoord);
src=src-degrid;
#endif
float4 psd=(float2(src.x*src.x+src.z*src.z,src.y*src.y+src.w*src.w)+0.0000000000000000000000000000000000001).xyxy;
float4 dst=src*(float4(1,1,1,1)+factor*sqrt( psd*SSMAX/((psd + SSMIN)*(psd + SSMAX))));
#ifdef DEGRID
dst=dst+degrid;
#endif
return dst;
}
```
Samplers: `Src` (s0), `Factor` (s1), `FactorM` (s5, if DEGRID).

### 5.2 CPU constants
`FFT3dGPU.cpp:187-188`: `smin /= 255 ; smax /= 255` (script defaults: smin=4.0, smax=20.0, scutoff=0.3, svr=1.0 — `FFT3dGPU.cpp:1286-1289`).
`FFT3dGPU.cpp:413`:
```cpp
sharp = NEW Sharpen(sharpen, svr, scutoff, smin*smin*bw*bh*bt, smax*smax*bw*bh*bt,
                    bw, nx, bh, ny, _degrid != 0, pFreeFImgdPool, hr, pDevice, gtype);
```
→ `SSMIN = smin²·bw·bh·bt`, `SSMAX = smax²·bw·bh·bt` (the psSharpen ctor receives (max, min) in that order, `filtersps.cpp:742`).

### 5.3 Factor LUT (CPU computation, float)
`Sharpen::Sharpen` (`filters.cpp:935-976`), comment "from Fizick's code fft3dfilter 1.8":
```cpp
int bh = _y; int bw = _x;
unsigned int x = (_x / 2 + 1);      // demi-largeur FFT + 1
int width = x * _xnum;              // largeur LUT
unsigned int y = _y;
int height = y * _ynum;
for (j = 0..y-1) {
  int dj = j; if (j >= y/2) dj = y - j;
  float d2v = float(dj*dj)*(svr*svr) / ((bh/2)*(bh/2)); // v1.7
  for (i = 0..x-1) {
    float d2 = d2v + float(i*i) / ((bw/2)*(bw/2));      // distance_2 - v1.7
    Factor[i + offset] = strength * (1 - exp(-d2 / (2 * scutoff*scutoff)));
  }
}
```
Then tiling `FactorMap[offset++] = Factor[i + j*x]` over `_ynum×_xnum` (`filters.cpp:963-968`) and upload into `FilterLUT` = `TextureM` **FLOAT1** (`gtype->FLOAT()`), size `width × height` (`filters.cpp:970-972`).
Note: `strength` = the script `sharpen` parameter (default 0.0; negative = blur, see doc `filters.cpp:920`).

### 5.4 Textures / call
- Inputs: `Src` = filtered FFT texture (FLOAT4/HALF4 `totw/2×toth`), `Factor` = FilterLUT FLOAT1, `FactorM` = `FImgDegrid` (if degrid). Output: `TextureRT` from the pool.
- `Apply` (`filtersps.cpp:753-770`): 1 draw, `NQuad(&_Rendertarget->GetRect(), 1)` (`filtersps.cpp:750`, the ctor's "rendertarget" is actually FilterLUT — same RECT as the FFT textures).
- Simple `Sharpen::Filter` (`filters.cpp:984-987`); pair version: 2 draws, degrid first/last (`filters.cpp:989-1001`).
- Conditions: `sharpen != 0` (`FFT3dGPU.cpp:173, 412`). Order: **after** the Wienner/Kalman filter, **before** the iFFT; with degrid, the pattern is removed and then put back inside the shader (`FFT3dGPU.cpp:860-865, 905-913, 964-969, 1023-1028`).

---

## 6. ImgStream (modes 0 and 2; legacy mode 1) + psImg2toFloat4*/psFloat4toImg2* classes

**ImgStream**: `filters.h:28-61`, `filters.cpp:56-341`. Created when `mode != 1` (`FFT3dGPU.cpp:466`). Constructor doc: `filters.cpp:29-53` — mode 0 = 1:1 overlap, modified cosine window; mode 1 = 2:1 overlap, half-cosine; mode 2 = 1:1 overlap + added border.

### 6.1 Allocated textures (`filters.cpp:56-134`)
- `Img` `TextureM`: `((width+1)>>1) × height`, FIXED2 (8 bit) / FIXED2_16 (≤16 bit) / FLOAT2 (float) (`filters.cpp:65`). **2 source pixels per texel** (column-pair packing).
- `Imgd` `TextureRT`: same size/format (`filters.cpp:71`) — float intermediate buffer.
- `Imgr` `TextureRT`: `((width+3)>>2) × height`, FIXED4/FIXED4_16/FLOAT4 (`filters.cpp:74`) — used by the final conversion shader `conv` (4 pixels/texel). Destroyed at the end of the ctor (`filters.cpp:132`) but `conv` keeps the RECT (`filtersps.cpp:345-346`).
- `FactorLUT` `TextureM`: mode≠2: `xn2 × yn` (`xn=x*bw…` actually `xn=x*xnum`, `x=bw`) as **HALF2** if useHalf otherwise **FLOAT2** (`filters.cpp:95-98`); mode 2: `bw/2 × bh` (`filters.cpp:118-121`).
- CPU offset: `D3DXVECTOR2 offset = D3DXVECTOR2(0.25*(bw), 0.5*bh)` (`filters.cpp:80`); src RECT shifted by `-0.25*bw, -0.5*bh` (`filters.cpp:86-90`); rendertarget `(0,0,xn/2,yn)` (`filters.cpp:81-85`).

### 6.2 psImg2toFloat4m0 (mode 0)
`filtersps.h:103-113`, `filtersps.cpp:368-417`. HLSL function **`Img2toFloat4`** (`hlsl:113-135`); defines: `BIT_DEPTH=int` + pixel type macro (`GETPIXEL2TYPEMACRO`) for a 2-component texture (`filtersps.cpp:369-372`).
```hlsl
float4 Img2toFloat4( PS_INPUT4 In) :COLOR
{
	float4 src; float4 dst; float2 factor;
	float2 img=tex2D(Src,In.texCoord[1]).SW1;
	float2 img_shifted=tex2D(Src,In.texCoord[2]).SW1;
	factor.xy=tex2D(Factor,In.texCoord[0]).xy;
	src.xy=img;
	src.zw=img_shifted;
	dst.xzyw=src*factor.xyxy;
#ifdef BITDEPTHNORM
  dst = dst * BITDEPTHNORM;
#endif
	return dst;
}
```
Geometry: `RECT rect[3]={Rendertarget, Src, Src+offset}` then `NQuad(rect, 3)` (`filtersps.cpp:376-382`) — stage0=Factor (target RECT), stage1=Src, stage2=Src shifted by `offset` (the pixel shifted by bw/4 in x and bh/2 in y). Output: `img` FLOAT4/HALF4 `totw/2×toth`. 1 draw.

### 6.3 psFloat4toImg2m0 (mode 0, inverse)
`filtersps.h:134-145`, `filtersps.cpp:473-539`. Function **`Float4toImg2`** (`hlsl:137-161`), **no define** (`#else` branch, not MODE2):
```hlsl
dst=tex2D(Src,In.texCoord[1]).xz*tex2D(Factor,In.texCoord[1]).xy
   +tex2D(Src,In.texCoord[0]).yw*tex2D(Factor,In.texCoord[0]).xy;
return dst.xyyy;
```
Geometry: `rect[3]={Imgd->GetRect(), FactorMap->GetRect(), FactorMap->GetRect()+offset}`, `NQuad(rect, 2, 1)` (`filtersps.cpp:478-483`). Samples the same Src texture at 2 addresses (the shift = the sliding window) and sums the 2 window contributions. Output: `Imgd` (FIXED2/FIXED2_16/FLOAT2). `pback` = backbuffer restored after the draw (`filtersps.cpp:484, 531`).

### 6.4 psImg2toFloat4m2 / psFloat4toImg2m2 (mode 2, with border)
`filtersps.cpp:419-471` and `541-610`. Same HLSL functions, but:
- `psFloat4toImg2m2` compiles with define **`MODE2`** (`filtersps.cpp:542-543`) and sampler constant **`FactorM`** (`filtersps.cpp:545`). MODE2 branch of the HLSL (`hlsl:146-156`):
```hlsl
float2 src_shifted=tex2D(Src,In.texCoord[1]).xz;
float2 factor_shifted=tex2D(FactorM,In.texCoord[3]).xy;
float2 src=tex2D(Src,In.texCoord[0]).yw;
float2 factor=tex2D(FactorM,In.texCoord[2]).xy;
dst=src_shifted*factor_shifted+src*factor;
dst=src_shifted+src;   // ⚠ écrase la ligne précédente : en MODE2 la fenêtre n'est PAS appliquée à la synthèse
```
- Geometry: `MQuad::CreateVB/CreateVBi(srcW, srcH, bw/2, bh, xnum, ynum, border)` (`filtersps.cpp:427-429, 549-551`) — per-block vertex buffer with border (mode 2).
- Mode 2 CPU window: `CreateFactorMapBorder` (see 6.7).

### 6.5 psImg2toFloat4_2 / psFloat4toImg2_2 (ImgStream's mode 1, legacy)
`filtersps.cpp:613-662` and `665-738`. Functions **`Img2toFloat4_2`** (`hlsl:165-183`, identical to Img2toFloat4 but with texCoord[1]/[2] swapped) and **`Float4toImg2_2`** (`hlsl:185-198`):
```hlsl
dst=tex2D(Src,In.texCoord[1]).xz*tex2D(Factor,In.texCoord[1]).xy+tex2D(Src,In.texCoord[0]).yw*tex2D(Factor,In.texCoord[0]).xy
+tex2D(I,In.texCoord[2]).xz*tex2D(Factor,In.texCoord[2]).xy+tex2D(I,In.texCoord[3]).yw*tex2D(I,In.texCoord[3]).xy;
```
(4 terms: the 2 halves of the 2:1 overlap, texture `Src`=first pair, `I`=second pair). `psFloat4toImg2_2::Apply(src1, src2, FactorLUT, dst)` binds `Src=s1, I=s2` (`filtersps.cpp:713-738`). Used by `ImgStream::ImgToStream(pTextureRTpair*)` / `StreamToImg(pair)` (`filters.cpp:286-301, 335-341`) — this path is no longer taken by the filter (mode 1 → `ImgStream2`), but `ImgStream` still instantiates it when `mode==1` (`filters.cpp:111-114`).

### 6.6 psImg2toImg4 (final 2→4 component conversion)
`filtersps.h:188-194`, `filtersps.cpp:337-366`. Function **`Img2toImg4`** (`hlsl:980-1003`), defines: `OFFSET=float2(1.0/src->GetWidth(), 0)` (`filtersps.cpp:339`), `BIT_DEPTH`, type macro:
```hlsl
float4 Img2toImg4( PS_INPUT In ) :COLOR
{
float4 dst=float4(0,0,0,0);
#if defined(G16R16) || defined(G32R32F)
dst.xy=tex2D(Src,In.texCoord).SW1;
dst.zw=tex2D(Src,In.texCoord+OFFSET).SW1;
#else
dst.zy=tex2D(Src,In.texCoord).SW1;
dst.xw=tex2D(Src,In.texCoord+OFFSET).SW1;
#endif
#if defined(INV_BITDEPTHNORM)
  dst = dst * INV_BITDEPTHNORM;   // retour 10-14 bit
  dst = min(dst, MAX_PIXEL_VALUE);
#endif
return dst;
}
```
Geometry: `RECT rect[2]={Dst->GetRect(), src->GetRect()}`, `NQuad(rect, 2, 1, true)` (`filtersps.cpp:345-346`). Called by `ImgStream::StreamToImg`: `Float4toImg2->Apply(src,FactorLUT,Imgd); conv->Apply(Imgd,dst)` (`filters.cpp:322-332`) — `dst` = `DownloadImg[plane]` FIXED4/FIXED4_16/FLOAT4 `((width+3)>>2)×height`.

### 6.7 ImgStream CPU windows (double precision)
`const double pi(acos(-1.0));` (`filters.cpp:27`).
**CreateFactorMap** (`filters.cpp:157-185`) — 2D window `x×y` (bw×bh), tiled `xnum×ynum`:
```cpp
for (double n1 = -y1/2.0 + 0.5; n1 < y1/2.0; n1++) {
  cosy = cos(n1*pi / (y1));
  for (double n2 = -x1/2.0 + 0.5; n2 < x1/2.0; n2++) {
    cosx = cos(n2*pi / (x1));
    if (full)  Factormap[offset++] = cosy * cosx;                       // mode 1 : demi-cosinus séparable
    else       Factormap[offset++] = sqrt(0.5*(cosx*cosx + cosy * cosy)); // mode 0 : cosinus modifié
  }
}
```
**CreateFactorMapBorder** (`filters.cpp:205-234`) — mode 2, `borderx=bordery=border*2`:
```cpp
cosy = cos(n1*pi / (y1 - 2*bordery));
int signy = cosy / abs(cosy);
cosx = cos(n2*pi / (x1 - 2*borderx));
int signx = cosx / abs(cosx);
double res = (0.5*(signx*cosx*cosx + signy * cosy*cosy));
Map[offset++] = res < 0 ? 0 : res;
```
(⚠ `abs()` on a double = integer truncation depending on the includes — reproduce it, or consciously fix it, when porting.)

### 6.8 ImgStream flow summary
- `ImgToStream(Texture* Img, TextureRT* dst)`: `Img2toFloat4->Apply(Img, FactorLUT, dst)` (`filters.cpp:265-283`).
- `StreamToImg(Texture* src, TextureRT* dst)`: `Float4toImg2->Apply(src, FactorLUT, Imgd)` then `conv->Apply(Imgd, dst)` (`filters.cpp:322-332`).
- CPU→GPU upload of the frame: `UploadToTexture(upload, src->GetReadPtr(plane), pitch/pixelsize)` (`FFT3dGPU.cpp:839, 931`) into `UploadImg[plane]` (FIXED2/FIXED2_16/FLOAT2, `((width+1)>>1)×height`).

---

## 7. ImgStream2 (mode 1 — modern main path) + psImg2ToFloat4* / psFloat4ToImg2* families

**ImgStream2**: `filters.h:63-83`, `filters.cpp:366-707`. Created when `mode == 1` (`FFT3dGPU.cpp:461-464`):
```cpp
convert2 = NEW ImgStream2(bw, bh, ow, oh, isLumaPlane?UploadImg[0]:UploadImg[1],
    imgp->first, isLumaPlane?DownloadImg[0]:DownloadImg[1],
    useChromaDisplacementMacro /* = !isLumaPlane && !isRGB */, wintype, interlaced, pDevice, gtype, useHalf, bits_per_pixel, hr);
```
Important note: the shader classes receive **`bw/2` as the block width** (the 2-columns-per-texel packing is already applied) and `ow/2` (`filters.cpp:425-444`).

### 7.1 Textures and class selection (`filters.cpp:366-478`)
- `Imgd` `TextureRT` `width×height` FIXED2/FIXED2_16/FLOAT2 (`filters.cpp:371`); `conv = psImg2toImg4(Imgd → dst)` (`filters.cpp:372`).
- `FactorLUTana` / `FactorLUTsyn` `TextureM` **FLOAT4** (`gtype->FLOAT4()`):
  - if `Cap.NumSimultaneousRTs > 1 || ow*2 != bw || oh*2 != bh`: size **`bw/2 × bh`** (small LUT, WRAP addressing) (`filters.cpp:376-380`);
  - otherwise (1 RT + half overlap): size **`fdst->GetWidth() × fdst->GetHeight()`** (full-size LUT) (`filters.cpp:381-385`).
- Selection of `i2f` (psImg2ToFloat4) and `f2i` (psFloat4ToImg2) (`filters.cpp:386-447`):

| Case | MRT>1 | MRT==1 |
|---|---|---|
| half overlap (`ow*2==bw && oh*2==bh`), progressive | `psImg2ToFloat4ohalfSP` + `psFloat4ToImg2ohalf(fullsizefactor=false)` | `psImg2ToFloat4ohalfMP` + `psFloat4ToImg2ohalf(fullsizefactor=true)` |
| half, interlaced | `psImg2ToFloat4ohalfInterlacedSP` + `psFloat4ToImg2ohalfInterlaced` | `psImg2ToFloat4ohalfInterlacedMP` + `psFloat4ToImg2ohalfInterlaced` |
| arbitrary overlap, progressive | `psImg2ToFloat4oSP` + `psFloat4ToImg2o` | `psImg2ToFloat4oMP` + `psFloat4ToImg2o` |
| arbitrary, interlaced | `psImg2ToFloat4oInterlacedSP` + `psFloat4ToImg2oInterlaced` | `psImg2ToFloat4oInterlacedMP` + `psFloat4ToImg2oInterlaced` |

- `ImgToTexture(src, dstpair)`: `i2f->Apply(src, FactorLUTana, dst->first, dst->last)` — **one pass, 2 RTs** (or 2 MP passes) (`filters.cpp:685-690`).
- `TextureToImg(srcpair, dst)`: `f2i->Apply(src->first, src->last, FactorLUTsyn, Imgd); conv->Apply(Imgd, dst)` (`filters.cpp:692-697`).
- `TextureToSrc`: same but `DownloadFromTexture(Imgd, dst, pitch)` (`filters.cpp:699-707`).

### 7.2 ImgStream2 CPU windows — `CreateFactorMap` (`filters.cpp:501-670`)
Doc: "based on the window function code from fft3dfilter 1.8.5". 1D windows over the overlap region `ow`/`oh` (double):
- **wintype 0** (default): analysis = synthesis = half-cosine:
```cpp
wanx[i] = cosf(pi*(i - ow + 0.5f) / (ow * 2));   wsynx[i] = wanx[i];
wany[i] = cosf(pi*(i - oh + 0.5f) / (oh * 2));   wsyny[i] = wany[i];
```
- **wintype 1**: analysis = √(half-cos), synthesis = (half-cos)^1.5:
```cpp
wanx[i] = sqrt(cosf(pi*(i - ow + 0.5f) / (ow * 2)));
wsynx[i] = wanx[i] * wanx[i] * wanx[i];   // = cos^1.5
```
- **wintype 2**: analysis = 1, synthesis = cos² (raised cosine):
```cpp
wanx[i] = 1.0;
wsynx[i] = cosf(pi*(i - ow + 0.5f) / (ow * 2)); wsynx[i] *= wsynx[i];
```
2D assembly (`filters.cpp:555-664`): the window is **separable and symmetric**; each LUT pixel is a `float4` carrying **2 consecutive horizontal samples** (x and x+1) for the left part, and the mirrors `wanx[(ow-1)-x]`, `wanx[ow-x-2]` for the right part; 3 vertical bands: `oh` rows at the top (product `wanx*wany`), `bh-2*oh` rows in the middle (x only, or 1), `oh` rows at the bottom in reverse order (`y = oh-1 → 0`). The center equals `1` (analysis and synthesis) (`filters.cpp:606-616`).
If `MRT==1 && half overlap`: the `bw*2×bh` LUT is tiled into `xnum×ynum` to cover the whole image (`filters.cpp:452-471`). Upload: `UploadToTexture(FactorLUTana, facana)` / `(FactorLUTsyn, facsyn)` (`filters.cpp:473-474`).

### 7.3 Common HLSL: Img2ToFloat4 / Img2ToFloat4MP (`hlsl:936-977`, guard `#ifdef OFFSET`)
```hlsl
PS_OUTPUT Img2ToFloat4(PS_INPUT4 In){
PS_OUTPUT dst;
float4 factor=tex2D(Factor,In.texCoord[2]).xxyy;
dst.o[0].xz=tex2D(Src,In.texCoord[0]).SW1;
dst.o[0].yw=tex2D(Src,In.texCoord[1]).SW1;
dst.o[1].xz=tex2D(Src,In.texCoord[0]+OFFSET).SW1;
dst.o[1].yw=tex2D(Src,In.texCoord[1]+OFFSET).SW1;
#ifdef BITDEPTHNORM
  dst.o[0] = dst.o[0] * BITDEPTHNORM;
  dst.o[1] = dst.o[1] * BITDEPTHNORM;
#endif
dst.o[0]=(dst.o[0]-CHROMANORM)*factor;
dst.o[1]=(dst.o[1]-CHROMANORM)*factor;
return dst;
}
float4 Img2ToFloat4MP(PS_INPUT4 In):COLOR{
float4 dst;
float4 factor=tex2D(Factor,In.texCoord[2]).xxyy;
#ifdef PASS1
dst.xz=tex2D(Src,In.texCoord[0]).SW1;
dst.yw=tex2D(Src,In.texCoord[1]).SW1;
#else
dst.xz=tex2D(Src,In.texCoord[0]+OFFSET).SW1;
dst.yw=tex2D(Src,In.texCoord[1]+OFFSET).SW1;
#endif
#ifdef BITDEPTHNORM
  dst = dst * BITDEPTHNORM;
#endif
dst=(dst-CHROMANORM);
return dst*factor;
}
```
Bit-depth macros via `DEFINE_A8L8_AND_CHROMA_MACROARRAY_MAX3/4/5` (`filtersps.cpp:874-876`): set `A8L8|G16R16|G32R32F`, `BIT_DEPTH`, and `CHROMA_BIT_DEPTH` if `chroma`.

### 7.4 psImg2ToFloat4oSP (arbitrary overlap, MRT, 1 pass)
`filtersps.cpp:878-915`. Defines: `OFFSET=float2(0, (bh-oh)/(float)src->GetHeight())` (`filtersps.cpp:880`), `BIT_DEPTH`, chroma. Dedicated vertex buffer `OQuad::CreateVB(srcW, srcH, bw, bh, ow, oh)` (`filtersps.cpp:886-888`) — one quad per block with distinct xy/zw coordinates. `Apply`: MRT `dst1=RT0` (first image of the overlap), `dst2=RT1` (`filtersps.cpp:896-915`). Outputs: `imgp->first/last` FLOAT4/HALF4.

### 7.5 psImg2ToFloat4oMP (1 RT, 2 passes)
`filtersps.cpp:918-965`. Pass1: `OFFSET=float2(0, (bh/2)/srcH)` + define `PASS1` (`filtersps.cpp:919-924`) → writes `dst1`; Pass2: `OFFSET=float2(0, (bh-oh)/srcH)` without PASS1 (`filtersps.cpp:925-929`) → writes `dst2`. Same `OQuad`.

### 7.6 psImg2ToFloat4oInterlacedSP / MP (interlaced)
`filtersps.cpp:968-1076` / `1078-1194`. `OFFSET=float2(0, 2*(bh-oh)/(float)src->GetHeight())` (`filtersps.cpp:970, 1080, 1086`). Vertex buffer `OIQuad::CreateVB` (row pairs). After the draw(s), **4×2 `StretchRect`** to swap the top/bottom halves of the fields in dst1 and dst2 (`filtersps.cpp:1057-1073`, rectangles `r[0..11]` computed in the ctor `filtersps.cpp:986-1004`):
```cpp
int ny = ((src->GetHeight() + oh - 1) / (bh - oh) + 1 + 1) / 2;
bottomheight = ny / 2 * (bh - oh) - src->GetHeight() / 2;
```
(swap r0↔r2, r4↔r6 in dst1; r8↔r10 in dst2, via `temptex` copy of `dst`). MP: pass1→dst1, pass2→dst2 then the same swaps (`filtersps.cpp:1154-1191`).

### 7.7 psImg2ToFloat4ohalfSP (exact 50% overlap, MRT)
`filtersps.cpp:1196-1249`. Defines: `OFFSET=float2(0, (bh/2)/(float)src->GetHeight())` (`filtersps.cpp:1198`). Geometry `NQuad(rec[4], 3, 1, false)` (`filtersps.cpp:1205-1225`):
```
rec[0] = destination  dstwidth=((srcW + bw*1.5 - 1)/bw)*bw, dstheight=((srcH + bh*1.5 -1)/bh)*bh
rec[1] = src décalé xy : left=-bw/2, right=srcW-bw/2, top=-bh/2, bottom=srcH-bh/2
rec[2] = src décalé zw : top=-bh/2, bottom=srcH-bh/2 (x non décalé)
rec[3] = factor : (0,0,bw,bh)  → LUT petit format, d'où ADDRESSU=WRAP
```
`Apply`: Factor sampler `ADDRESSU=WRAP` during the draw, reset to `MIRROR` afterwards (`filtersps.cpp:1239-1247`); MRT dst1/dst2.

### 7.8 psImg2ToFloat4ohalfInterlacedSP (interlaced, MRT)
`filtersps.cpp:1251-1364`. `OFFSET=float2(0, bh/(float)srcH)` (`filtersps.cpp:1253`). Two quads: `quad` (even field, `qrec={0,0,dstW,dstH/2}`) and `bottomquad` (odd field, source RECTs **+1 pixel** in y, `qrec` offset by dstH/2) (`filtersps.cpp:1262-1288`). `Apply`: `Clear(0x00880000)` (`filtersps.cpp:1328`), draw `bottomquad` then `quad` (`1338-1342`), Factor WRAP→MIRROR, then StretchRect swaps (`1345-1358`) with:
```cpp
int bottomheight = (dstheight - bh - src->GetHeight()) / 2;   // (filtersps.cpp:1290)
```

### 7.9 psImg2ToFloat4ohalfMP / ohalfInterlacedMP (1 RT)
`filtersps.cpp:1367-1429` / `1432-1551`. 2 shaders `Img2ToFloat4MP`: pass1 with define `PASS1` and `OFFSET=float2(0,(bh/2)/srcH)` (`1368-1373`), pass2 without PASS1, same OFFSET (`1374-1378`) — here OFFSET pass1==pass2 because the offset is carried by the quad's RECTs. Draw pass1→dst1, pass2→dst2 (`1405-1429`). Interlaced version: `OFFSET=float2(0, bh/srcH)` for both passes (`1433-1443`), Clear `0x00880000`, draws bottomquad then quad for each pass (`1504-1532`), StretchRect swaps (`1534-1547`).

### 7.10 psFloat4ToImg2o (synthesis arbitrary overlap, progressive)
`filtersps.h:348-363`, `filtersps.cpp:1556-1636`. **6 shaders** over the same pair of FFT textures (`Src`=even, `Src2`=odd) + `Factor` (synthesis LUT FLOAT4):
- `Float4ToImg2Corner` (no define, `hlsl:880-895`) — block corners, cross-mixing of the 4 textures:
```hlsl
float4 factortex0=tex2D(Factor,In.texCoord[2]);
float4 factortex1=tex2D(Factor,In.texCoord[3]);
tex0.xy=tex2D(Src,In.texCoord[0]).xz;
tex1.zw=tex2D(Src2,In.texCoord[1]).yw;
tex0.zw=tex2D(Src,float2(In.texCoord[1].x,In.texCoord[0].y)).yw;
tex1.xy=tex2D(Src2,float2(In.texCoord[0].x,In.texCoord[1].y)).xz;
float4 tmp=tex0*factortex0+tex1*factortex1;
dst=tmp.xyzw+tmp.zwxy;
return dst+CHROMANORM;
```
- `Float4ToImg2Nooverlap` (`hlsl:905-910`) with define **`XY`** for the xy variant (otherwise zw):
```hlsl
#ifdef XY
#define SW0 xzxz
#define SWF xyxy
#else
#define SW0 ywyw
#define SWF zwzw
#endif
float4 Float4ToImg2Nooverlap(PS_INPUT4 In):COLOR{
float4 dst=tex2D(Src,In.texCoord[0]).SW0;
return dst+CHROMANORM;
}
```
- `Float4ToImg2Horizontal` (`hlsl:912-922`): `dst=src1*factortex0+src2*factortex1` (SW0/SWF) `+CHROMANORM`; define `XY` for xy.
- `Float4ToImg2Vertical` (`hlsl:924-933`):
```hlsl
float4 factor=tex2D(Factor,In.texCoord[2]);
float4 src1=tex2D(Src,In.texCoord[0]).xzxz;
float4 src2=tex2D(Src,In.texCoord[1]).ywyw;
dst=src1*factor.xyxy+src2*factor.zwzw;
return dst+CHROMANORM;
```
Exact order of the draws in `Apply` (`filtersps.cpp:1579-1630`), after `Clear(0xFFFFFFFF)` (`1586`):
1. `DrawCorners` (Corner shader); 2. `DrawHorizontalxy` (XY); 3. `DrawHorizontalzw`; 4. `DrawNonoverlaptex1xy` (XY); 5. bind `Src2→sSrc1` then `DrawNonoverlaptex2xy`; 6. `DrawNonoverlaptex2zw`; 7. bind `Src1→sSrc1` then `DrawNonoverlaptex1zw`; 8. `DrawVerticaltex1`; 9. bind `Src2` then `DrawVerticaltex2`.
Bit-depth defines: `CHROMA_BIT_DEPTH` if chroma (`filtersps.cpp:1557-1573`).

### 7.11 psFloat4ToImg2oInterlaced (interlaced synthesis — z-buffer trick)
`filtersps.h:366-386`, `filtersps.cpp:1638-1865`. Same 6 shaders. In the constructor (`1658-1696`):
- `ztex` = `TextureRT` `width×height` **FIXED1** (`gtype->FIXED()`) filled on the CPU: row i = `(i & 1) * 255` (`1667-1671`);
- `zbuffer` = `CreateDepthStencilSurface(width, height, D3DFMT_D24X8, ...)` (`1665`);
- shader **`CopyTexToZBuffer`** (`hlsl:1012-1018`, `dst.d=tex2D(Src,In).x`) copies ztex into the z-buffer with `ZWRITEENABLE=TRUE, ZFUNC=ALWAYS` (`1676-1688`).
`Apply` (`1747-1855`): `Clear(0x00880000)`; `ZENABLE=TRUE, ZWRITE=FALSE, ZFUNC=LESS` → draws of the **even** quad (`oevenquad`); then `ZFUNC=GREATER` → draws of the **odd** quad (`ooddquad`) — the depth test filters the row pairs. The 2 quads come from `OIQuad::CreateVBi(width,height,bw,bh,ow,oh,srcwidth,srcheight, true/false)` (`1692-1693`).

### 7.12 psFloat4ToImg2ohalf (synthesis 50% overlap, 1 draw)
`filtersps.h:321-329`, `filtersps.cpp:1869-1920`. **`Float4ToImg2Corner`** shader only, `NQuad(rec[5], 5, 1, false)` (`1876-1897`):
```
rec[0] = dst (width×height)
rec[1] = tex0xy : left=bw/2, right=srcwidth+bw/2, top=bh/2, bottom=srcheight+bh/2
rec[2] = tex1zw : (0,0,srcwidth,srcheight)
rec[3] = factortex0 : left=bw/2, top=bh/2, right=(fullsizefactor?srcwidth:bw)+bw/2, bottom=(fullsizefactor?srcheight:bh)+bh/2
rec[4] = factortex1 : left=bw/2, top=0, right=(fullsizefactor?srcwidth:bw)+bw/2, bottom=(fullsizefactor?srcheight:bh)
```
Factor sampler `ADDRESSU=WRAP`→`MIRROR` (`1911-1918`). Output: `Imgd` FIXED2/FIXED2_16/FLOAT2.

### 7.13 psFloat4ToImg2ohalfInterlaced (interlaced synthesis 50%)
`filtersps.h:332-345`, `filtersps.cpp:1923-2100`. Same z-buffer principle as 7.11 (ztex FIXED1 + zbuffer D24X8 + `CopyTexToZBuffer`, `1937-1963`). Two 5-RECT quads: `quad` (first half, `qrec={0,0,width,height}` but rec[0].bottom=height/2) and `evenfieldquad` (source RECTs `+srcheight/2` in y, `qrec.top++/bottom++` — comment `//inc to get correct eta`, `1990-2000`). `Apply` (`2063-2100`): Clear `0x00880000`, Factor WRAP, `ZFUNC=GREATER` draw `quad`, then `ZFUNC=LESS` draw `evenfieldquad`, reset to neutral state.

### 7.14 Summary of the ImgStream2 flow (mode 1)
`UploadToTexture(UploadImg[plane])` → `convert2->ImgToTexture(upload, imgp)` (1 MRT draw or 2 passes) → FFT×2 (`imgp->first`, `imgp->last`) → filter (Wienner/Kalman in pairs) → sharpen → iFFT×2 → `convert2->TextureToImg(imgp, download)` (synthesis + `psImg2toImg4`) → `DownloadFromTexture(DownloadImg[plane])` (`FFT3dGPU.cpp:873-1042, 1140-1166`).

---

## 8. psFFTtoFixed (debug/visualization — not used in production)

`filtersps.h:400-409`, `filtersps.cpp:778-840`. **`FFTtoFixed`** function (`hlsl:606-615`), optional define **`FFT`** (`filtersps.cpp:779`):
```hlsl
float4 FFTtoFixed( PS_INPUT In) : COLOR
{
float4 src=tex2D(Src,In.texCoord);
#ifdef FFT
float4 dst=0.1*log(1.0f+length(src.xz));
#else
float4 dst=src.ywxz;
#endif
return dst;
}
```
Output to the backbuffer or an RT (`filtersps.cpp:818-840`). The only instantiation: **commented out** (`FFT3dGPU.cpp:509`). `psMeanSD` is entirely commented out (`filtersps.h:410-418`, `filtersps.cpp:842-872`) — the equivalent CPU logic is `FFT3dGPU::CalcMeanSD` (`FFT3dGPU.cpp:647-667`, not called).

---

## 9. Complete order of passes in GetFrame (reminder for the port)

Per plane (luma then chroma U,V — two FFT3dGPU instances if plane=4, `FFT3dGPU.cpp:1260-1341`):
1. Reset render/sampler state (MIRROR on stages 0..5) — `FFT3dGPU.cpp:813-820`.
2. Upload frame → `UploadImg[plane]` (FIXED2/16/FLOAT2) — `UploadToTexture` / `UploadInterleavedToFixedTexture` (YUY2) (`FFT3dGPU.cpp:837-842`).
3. `ImgToStream`/`ImgToTexture` (classes from §6/§7).
4. 2D FFT (`FFT2d*`, outside the scope of this document) on `img` or `imgp->first/last`.
5. bt≥2: FFT of the bt frames via `GPUCache` (`FFT3dGPU.cpp:829-853`).
6. degrid: `psMinimize` then `psGridCorrection` → `FImgDegrid` (`855-858`).
7. Filter: `psWiennerFilter` (bt≥1) or `KalmanFilter` (bt=0) (`859, 954-960`).
8. `Sharpen` if `sharpen!=0` (`860-863`).
9. iFFT.
10. `StreamToImg`/`TextureToImg`: synthesis (`psFloat4ToImg2*`) + `psImg2toImg4`.
11. `DownloadFromTexture(DownloadImg[plane])` (`FFT3dGPU.cpp:1140-1166`).
12. Early upload of the next frame (`UploadImg2` then swap, `FFT3dGPU.cpp:1093-1136`).

## 10. Points of attention for the OpenCL port
- **Half-texel**: all the mapping is in the NQuad RECTs (`eta = 0.5/W + left/W`, `vertexbuffer.cpp:90-91`); in OpenCL, work with integer texel indices rather than normalized coordinates.
- **Precision**: the float defines are formatted with `std::fixed` to 6 decimals (`shader.cpp:139,197`) → the compiled constant may differ from the CPU float; reproduce this if bitwise exactness is desired, or pass the constants as kernel arguments (preferable).
- **Multiple writes**: KalmanSP (3 RT), Img2ToFloat4 (2 RT) → 3/2 kernels or multiple `write_image`.
- **Sampling**: NEAREST + MIRRORED_REPEAT everywhere (except the "ohalf" factor LUT: REPEAT on U; initial device state: CLAMP border 0).
- **Formats**: HALF4↔`A16B16G16R16F` (fp16 RGBA), FIXED2↔`A8L8` (careful: L=R channel? mapping `SW1=xw` for A8L8: L→x, A→w; for G16R16: `SW1=rg`: R→x? — verify the component order in the port), packing 2 px/texel (FIXED2) and 4 px/texel (FIXED4).
- **10-14 bit normalization**: `BITDEPTHNORM = 1<<(16-BIT_DEPTH)` at the input, `INV_BITDEPTHNORM` + clamp `MAX_PIXEL_VALUE` at the output (`hlsl:35-39`, `hlsl:995-1000`).
- **CHROMANORM**: `32768/65535` offset added/removed for YUY2 chroma (`hlsl:21-33`, `hlsl:951-952`, `hlsl:894`).
- **Known bug to reproduce or fix deliberately**: MODE2 branch of `Float4toImg2` where the synthesis window is overwritten by `dst=src_shifted+src` (`hlsl:155-156`); integer `abs()` on a double in `CreateFactorMapBorder` (`filters.cpp:220,227`).
