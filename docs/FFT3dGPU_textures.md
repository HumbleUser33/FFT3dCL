# Exhaustive technical documentation — FFT3dGPU (AviSynth / DirectX 9) for an OpenCL port

Full analysis of the files: `core/texture.h/.cpp`, `GPUCache.h/.cpp`, `TexturePool.h/.cpp`,
`core/ManageDirect3DWindow.h/.cpp`, `core/pixelshader.h/.cpp`, `core/vertexshader.h/.cpp`,
`core/shader.h/.cpp`, `core/vertexbuffer.h/.cpp`, `Dxinput.h/.cpp`, `core/getdxver.cpp`,
`FFT3dGPU.cpp` (constructor, upload/download, CalcMeanSD, SigmaToPatternTexture),
`filters.h/.cpp`, `filtersps.cpp` and `fft3dgpu.hlsl` (for the conversion formulas).

All references are `file:line`. The formulas are copied verbatim.

---

## 0. Component map

| Role | Files |
|---|---|
| AviSynth filter, validation, CPU↔GPU upload/download | `FFT3dGPU.cpp`, `FFT3dGPU.h` |
| D3D9 texture abstraction + GPU formats | `core/texture.h`, `core/texture.cpp` |
| Temporal FFT cache | `GPUCache.h/.cpp` |
| Texture pool | `TexturePool.h/.cpp` |
| D3D9 device (hidden window, reset, caps) | `core/ManageDirect3DWindow.h/.cpp`, `core/getdxver.cpp` |
| Shaders (runtime HLSL compilation) | `core/shader.h/.cpp`, `core/pixelshader.*`, `core/vertexshader.*`, `fft3dgpu.hlsl` |
| Image↔frequency-domain conversion, window functions | `filters.h/.cpp` (ImgStream/ImgStream2), `filtersps.h/.cpp` (pixel shader wrappers) |
| Quads/vertex buffers (coordinates) | `core/vertexbuffer.h/.cpp` |
| NVPerfHUD debug only | `Dxinput.h/.cpp` |

---

## 1. D3D texture formats used

### 1.1 Base types (`core/texture.h`)

```cpp
enum gpu_types {
  _FLOAT = 4,
  _HALF = 2,
  _FIXED = 1,
  _FIXED16 = 3
};
```
(`core/texture.h:32-37`)

`types` = `{elem_size (logical channels), real_elem_size (real channels of the chosen D3D format), format (D3DFORMAT), type, PixelSizeInByte = real_elem_size*GetByteSize(gt)}` (`core/texture.h:40-58`, sizes: `_FIXED`=1 byte, `_FIXED16`=2, `_HALF`=2, `_FLOAT`=4, `core/texture.h:41-49`).

`Types` = pair `{RenderTarget, Managed}` (`core/texture.h:60-63`): **two different formats** are probed for the same logical type, depending on whether the texture is a render target (`TextureRT`) or a managed texture (`TextureM`).

### 1.2 Format probe (`GPUTYPES`, `core/texture.cpp:42-193`)

For each type, `CheckDeviceFormat(..., D3DUSAGE_RENDERTARGET, D3DRTYPE_TEXTURE, fmt)` for the RT pool, and the same call with usage `0` for the Managed pool. Fallback chain (if the 2-channel format is not supported as a RT, it falls back to the 4-channel one, `real_elem_size=4`):

| Logical type | Preferred RT/Managed format | Fallback |
|---|---|---|
| `_float4` | `D3DFMT_A32B32G32R32F` (l.51-54) | none (real_elem_size=0) |
| `_float2` | `D3DFMT_G32R32F` (l.56-59) | `_float4` |
| `_float1` | `D3DFMT_R32F` (l.61-64) | `_float2` |
| `_half4` | `D3DFMT_A16B16G16R16F` (l.66-69) | none |
| `_half2` | `D3DFMT_G16R16F` (l.71-74) | `_half4` |
| `_half1` | `D3DFMT_R16F` (l.76-79) | `_half2` |
| `_fixed4` | `D3DFMT_A8R8G8B8` (l.82-85) | none |
| `_fixed2` | `D3DFMT_A8L8` (l.88-91) — pinterf comment: "A8L8 render target is not allowed for me, can really fallback to fixed4" | `_fixed4` |
| `_fixed1` | `D3DFMT_L8` (l.93-96) | `_fixed2` |
| `_fixed4_16` | `D3DFMT_A16B16G16R16` (l.99-102) | none |
| `_fixed2_16` | `D3DFMT_G16R16` (l.107-110) | `_fixed4_16` |
| `_fixed1_16` | `D3DFMT_L16` (l.112-115) | `_fixed2_16` |

Important comment (l.104-106): "No luck with G13R16 render target… must be simulated". Unsigned 16-bit UNORM formats are therefore the key to the 10-16 bit path.

Accessors: `FLOAT()/FLOAT2()/FLOAT4()/FIXED()/FIXED2()/FIXED4()/FIXED_16()/FIXED2_16()/FIXED4_16()/HALF()/HALF2()/HALF4()` (`core/texture.h:71-82`).

### 1.3 Upload/download textures by bit depth (`FFT3dGPU.cpp:289-346`)

For the **luma** (`_plane<=0` instance), width `width = vi.width/_xRatio`, height `height = vi.height/_yRatio` (`FFT3dGPU.cpp:172`); `_xRatio/_yRatio` = 1 for luma, = UV subsampling for chroma (`FFT3dGPU.cpp:1269-1270`, `1300-1301`):

| bits/comp | UploadImg[0..2] (TextureM, Managed) | size | DownloadImg[0..2] (TextureRT) | size |
|---|---|---|---|---|
| 8 | `FIXED2()` (A8L8) | `((width+1)>>1, height)` (l.293-295) | `FIXED4()` (A8R8G8B8) | `((width+3)>>2, height)` (l.296) |
| 10-16 | `FIXED2_16()` (G16R16) | `((width+1)>>1, height)` (l.299-301) | `FIXED4_16()` (A16B16G16R16) | `((width+3)>>2, height)` (l.302) |
| 32 float | `FLOAT2()` (G32R32F) | `((width+1)>>1, height)` (l.306-308) | `FLOAT4()` (A32B32G32R32F) | `((width+3)>>2, height)` (l.309) |

Chroma (chroma instance, `width` already divided by the UV ratio): `UploadImg[1]` and `[2]` at `((width+1)/2, height)`, `DownloadImg[1]`/`[2]` at `((width+3)/4, height)` (l.316-344).

**Packing convention**: one `FIXED2` texel carries **2 samples** (pixel pairs), one `FIXED4` texel carries **4**. That is why the upload texture is `width/2` wide and the download one `width/4`.

Intermediate working textures (`FFT3dGPU.cpp:348-358`):
```cpp
if (useHalf)
  if (mode != 1)
    img = NEW TextureRT(pDevice, totw / 2, toth, gtype->HALF4(), hr);
  else
    imgp = NEW pTextureRTpair(NEW TextureRT(pDevice, totw / 2, toth, gtype->HALF4(), hr), ...);
else
  ... gtype->FLOAT4() ...
```
`totw = nx*bw ; toth = ny*bh` (`FFT3dGPU.cpp:284-285`).

### 1.4 Precision (`FFT.h:40-42`)

```cpp
FLOAT16 = 0,
FLOAT32_FFT = 1,
FLOAT32_ALL = 2
```
Validation (`FFT3dGPU.cpp:244-252`):
```cpp
// FLOAT16 precision is not enough for 10+ bits (s10e5), using FLOAT32_ALL for bits_per_pixel>8
if (bits_per_pixel > 8)
  precision = FLOAT32_ALL;

if (precision<FLOAT16 || precision>FLOAT32_ALL)
  precision = FLOAT32_FFT;

// now precision is final
useHalf = precision == FLOAT16;
```
So **in 10-16 bits, HALF is never used**; the working textures are FLOAT4. `useHalf` only drives the format of the FFT textures (`HALF4` vs `FLOAT4`) and of the FactorLUT (`HALF2` vs `FLOAT2`, `filters.cpp:95-98`).

### 1.5 TextureM vs TextureRT vs pairs

- **TextureM** (`core/texture.h:246-263`, `core/texture.cpp:769-786`): `CreateTexture(..., D3DPOOL_MANAGED, ...)` (l.780). Used for: UploadImg (3×), FactorLUT, Pattern, gridsample. Upload = direct `LockRect` on the managed texture (`core/texture.cpp:838-842`).
- **TextureRT** (`core/texture.h:128-206`, `core/texture.cpp:268-297`): double buffer — `_pShadowTexture` in **D3DPOOL_SYSTEMMEM** (l.283) + `_pVideoTexture` in **D3DPOOL_DEFAULT** with `D3DUSAGE_RENDERTARGET` (`CreateVideoTexture`, l.436). Upload: lock the shadow then `UpdateTexture(shadow, video)` (`SetDataEnd`, l.610-620). Download: `GetRenderTargetData(video, shadow)` then `LockRect(D3DLOCK_READONLY)` (`GetData`, l.582-595).
- **pTextureRTpair** (`core/texture.h:209-244`): pair of TextureRT with its own reference counter (`AddRef/Release`, auto-`delete` when RefCount≤0). Used in **mode 1** only (two block halves, `imgp`, `FImgdp`, etc., `FFT3dGPU.h:125,148-160`).

---

## 2. Converting 8-bit and 10-16-bit frames (upload/download)

### 2.1 What is done on the CPU side (host)

**No numeric conversion on the CPU side.** `UploadToTexture` (`core/texture.cpp:867-973`) does a byte/uint16/float-to-byte/uint16/float copy, with channel reordering when the real format differs from the logical format:

- `_FIXED` (8 bits): fast `memcpy` path if `dstInc==2 && srcInc==2 && dstpitch==srcpitch` (l.914-917). Otherwise "magic with alpha" (l.918-934) to package 2 bytes into A8R8G8B8 when A8L8 is not supported:
  ```cpp
  int alpha = srcInc == dstInc ? dstInc - 1 : dstInc - 1 - srcInc;
  ...
  for (int elem = 1; elem < srcInc; elem++)
    for (int x = 0, doffset = dstInc - 1 - elem, soffset = elem - 1; ...)
      *(dstp + doffset) = *(srcp + soffset);
  for (int x = 0, doffset = alpha, soffset = srcInc - 1; ...)
      *(dstp + doffset) = *(srcp + soffset);
  ```
- `_FIXED16` (10-16 bits): same in `uint16_t`, fast memcpy if `dstInc==2&&srcInc==2` (l.951-954).
- `_FLOAT`/`_HALF`: src is always `float*`, implicit float→D3DXFLOAT16 conversion by assignment (l.878-903).

The **pitch** is passed in *element units*, not in bytes: `src->GetPitch(currentPlane) / pixelsize` (`FFT3dGPU.cpp:839, 1111, 1146`), `pixelsize = vi.ComponentSize()` (`FFT3dGPU.cpp:775`).

**8-bit and 10-16-bit normalization: it is done by D3D texture sampling** (UNORM8 → value/255, UNORM16 → value/65535), not by the code. The bytes/uint16 are stored raw in A8L8/G16R16; any shader doing `tex2D` receives a normalized float.

### 2.2 HLSL conversion macros (`fft3dgpu.hlsl:21-39`)

```hlsl
#if CHROMA_BIT_DEPTH==8
#define RANGE_HALF (1 << (8 - 1))
#define RANGE_FULL (float)((1 << 8) - 1)
#define CHROMA_CORR RANGE_HALF/RANGE_FULL
#elif CHROMA_BIT_DEPTH<=8
// 10, 12 and 14 bits are normalized to 16 bits as well
#define RANGE_HALF (1 << (16 - 1))
#define RANGE_FULL (float)((1 << 16) - 1)
#define CHROMA_CORR RANGE_HALF/RANGE_FULL
#else
#define CHROMA_CORR 0
#endif
#define CHROMANORM float4(CHROMA_CORR, CHROMA_CORR, CHROMA_CORR, CHROMA_CORR)

#if BIT_DEPTH>8 && BIT_DEPTH<16
#define BITDEPTHNORM float4(1 << (16 - BIT_DEPTH), 1 << (16 - BIT_DEPTH),1 << (16 - BIT_DEPTH),1 << (16 - BIT_DEPTH))
#define INV_BITDEPTHNORM float4(1.0 / (1 << (16 - BIT_DEPTH)), 1.0 / (1 << (16 - BIT_DEPTH)),1.0 / (1 << (16 - BIT_DEPTH)),1.0 / (1 << (16 - BIT_DEPTH)))
#define MAX_PIXEL_VALUE float4((float)((1 << BIT_DEPTH) - 1), (float)((1 << BIT_DEPTH) - 1), (float)((1 << BIT_DEPTH) - 1), (float)((1 << BIT_DEPTH) - 1))
#endif
```

Effective values of `CHROMA_CORR` (the `CHROMA_BIT_DEPTH` macro is defined **only for non-RGB chroma instances**, `useChromaDisplacementMacro = !isLumaPlane && !isRGB`, `FFT3dGPU.cpp:462`; see `DEFINE_A8L8_AND_CHROMA_MACROARRAY_MAX4`, `filtersps.cpp:875`):
- **luma** (macro not defined → evaluated to 0 by the preprocessor → `#elif` branch): `CHROMA_CORR = 32768/65535 ≈ 0.5000076`;
- **8-bit chroma**: `CHROMA_CORR = 128/255 ≈ 0.5019608`;
- **10-16-bit and float chroma**: `CHROMA_CORR = 0`.

⚠️ Porting pitfall: the DC subtraction/reconstitution (`-CHROMANORM` / `+CHROMANORM`, see 2.4) applies **only to the mode 1 shaders** (`Img2ToFloat4*`, `Float4ToImg2*`). Mode 0 (`Img2toFloat4`) does not center. And the value of `CHROMANORM` differs between luma (0.5000076) and 8-bit chroma (0.5019608).

### 2.3 Upload — 10-14-bit scaling (`fft3dgpu.hlsl:106-135`)

Source comment (l.106-112):
```
// Converts input (already normalized float) texture buffer to a four-unit buffer.
// When input was originally less than 16 bits but still uint16_t (typically 10, 12 and 14 bits),
// the converted range does not match with 0..1, rather 0 - 1023/65535 in a 10 bit input case.
// We stretch it to match with full 16 bit range in order to filtering parameters act the same
// way for all bit depths.
```
Mode 0/2 analysis shader, `Img2toFloat4` (`fft3dgpu.hlsl:113-135`):
```hlsl
float2 img=tex2D(Src,In.texCoord[1]).SW1;
float2 img_shifted=tex2D(Src,In.texCoord[2]).SW1;
factor.xy=tex2D(Factor,In.texCoord[0]).xy;
src.xy=img;
src.zw=img_shifted;
dst.xzyw=src*factor.xyxy;
#ifdef BITDEPTHNORM
  dst = dst * BITDEPTHNORM;
#endif
```
So for 10 bits: `v_uint16/65535 × 64` → range 0..≈1. For 8 bits: `v/255` (no factor). For 16 bits: `v/65535` (no BITDEPTHNORM, `BIT_DEPTH<16` false). For float: `v` as-is.

Swizzle selector `SW1` (`fft3dgpu.hlsl:95-104`):
```hlsl
#if defined(G16R16) || defined(G32R32F)
#define SW1 rg        // rg is xy
#else
#ifdef A8L8
#define SW1 xw
#else
#define SW1 xy
#endif
#endif
```
The format macro (`A8L8`/`G16R16`/`G32R32F`) is injected from C++ by `GETPIXEL2TYPEMACRO` (`filtersps.cpp:335`):
```cpp
#define GETPIXEL2TYPEMACRO (src->GetType()->format==D3DFMT_A8L8 || src->GetType()->format==D3DFMT_A8R8G8B8 ? "A8L8" : src->GetType()->format==D3DFMT_G16R16 ? "G16R16" : src->GetType()->format==D3DFMT_G32R32F ? "G32R32F" : "A8L8")
```
With A8L8, the texture's "alpha" channel carries the plane's odd byte (the 2-samples-per-texel packing uses L=even, A=odd) — hence `xw`.

Mode 1 (`Img2ToFloat4`, `fft3dgpu.hlsl:938-956`) — centers the data around 0:
```hlsl
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
```
Multipass variant `Img2ToFloat4MP` (l.958-976): `dst=(dst-CHROMANORM); return dst*factor;`

### 2.4 Download — rounding, clamp, rescaling

Download chain (mode 0/2, `filters.cpp:322-332`):
```cpp
Float4toImg2->Apply(src, FactorLUT, Imgd);   // float4 -> Imgd (FIXED2/FIXED2_16/FLOAT2)
conv->Apply(Imgd, dst);                      // Img2toImg4: Imgd -> DownloadImg (FIXED4/...)
```
(mode 1: `f2i->Apply(...)` then `conv->Apply(...)` — `filters.cpp:692-697`).

Mode 1 synthesis shaders (`fft3dgpu.hlsl:880-933`) — re-add the DC:
```hlsl
// Float4ToImg2Corner:
float4 tmp=tex0*factortex0+tex1*factortex1;
dst=tmp.xyzw+tmp.zwxy;
return dst+CHROMANORM;
// Float4ToImg2Nooverlap:  return dst+CHROMANORM;
// Float4ToImg2Horizontal: dst=src1*factortex0+src2*factortex1; return dst+CHROMANORM;
// Float4ToImg2Vertical:   dst=src1*factor.xyxy+src2*factor.zwzw; return dst+CHROMANORM;
```

Final shader `Img2toImg4` (`fft3dgpu.hlsl:980-1003`), packaging 2 two-channel texels → 1 four-channel texel:
```hlsl
#ifdef OFFSET
float4 Img2toImg4( PS_INPUT In ) :COLOR
{
float4 dst=float4(0,0,0,0);
#if defined(G16R16) || defined(G32R32F)
/*0123 becomes 2103 order at 16 bits
             xyzw
             zyxw
*/
dst.xy=tex2D(Src,In.texCoord).SW1;
dst.zw=tex2D(Src,In.texCoord+OFFSET).SW1;
#else
dst.zy=tex2D(Src,In.texCoord).SW1;
dst.xw=tex2D(Src,In.texCoord+OFFSET).SW1;
#endif

#if defined(INV_BITDEPTHNORM)
  // scale back to 10-14 bit range
  dst = dst * INV_BITDEPTHNORM;
  // limit to valid range
  dst = min(dst, MAX_PIXEL_VALUE);;
#endif

return dst;
}
#endif
```
`OFFSET = (1.0/src->GetWidth(), 0)` (one Imgd texel) and `BIT_DEPTH = bits_per_pixel` (`psImg2toImg4`, `filtersps.cpp:337-347`).

**Rounding and clamp**: they are done by the write into the **UNORM** D3D9 render targets (round-to-nearest + clamp [0,1]), then by the raw copy:
- 8 bits: `Imgd` (A8L8) then `DownloadImg` (A8R8G8B8) → final byte = `round(v×255)` clamped to 0..255.
- 10-14 bits: double UNORM16 quantization. `Imgd` (G16R16) quantizes `round(v×65535)`; `Img2toImg4` does `v/64`, then the write into A16B16G16R16 re-quantizes → final integer ≈ `round(v×1023.98)`, so well within the 10-bit range. ⚠️ The `min(dst, MAX_PIXEL_VALUE)` with MAX=1023 can never trigger after `/64` (dst≤~1); an out-of-range value (ringing, v>1) is clamped by the RT to 65535, an out-of-10-bit-range value copied as-is into the frame. To reproduce (or fix) consciously in the port.
- 16 bits: no scaling, `round(v×65535)`.
- 32-bit float: `FLOAT4` (A32B32G32R32F), no quantization.

Final copy into the AviSynth frame (`DownloadFromTexture`, `core/texture.cpp:991-1109`) with `IgnoreAlphaChannel=true`: full memcpy path for the 4-element types:
```cpp
if (dstInc == 4 && srcInc == 4 && IgnoreAlphaChannel)
{
  for (int y = 0; y < height; y++, srcp += srcpitch, dstp += dstpitch)
    memcpy(dstp, srcp, srcpitch);          // _FIXED (l.1048-1053)
```
Source comment (l.1015-1017): "we are using four-pixel units for final download / At non-8 bits the original 0 1 2 3 is changed into 2 1 0 3 / but we changed the order in Img2toImg4 in order to allow memcpy here".

### 2.5 Analysis/synthesis windows (factors multiplied at the same places)

Mode 0/2 (`ImgStream::CreateFactorMap`, `filters.cpp:157-185`):
```cpp
for (double n1 = -y1 / 2.0 + 0.5; n1 < y1 / 2.0; n1++) {
  cosy = cos(n1*pi / (y1));
  for (double n2 = -x1 / 2.0 + 0.5; n2 < x1 / 2.0; n2++) {
    cosx = cos(n2*pi / (x1));
    if (full)
      Factormap[offset++] = cosy * cosx;
    else
      Factormap[offset++] = sqrt(0.5*(cosx*cosx + cosy * cosy));
```
Mode 2 with border (`CreateFactorMapBorder`, `filters.cpp:205-234`):
```cpp
cosy = cos(n1*pi / (y1 - 2 * bordery));
int signy = cosy / abs(cosy);
...
cosx = cos(n2*pi / (x1 - 2 * borderx));
int signx = cosx / abs(cosx);
double res = (0.5*(signx*cosx*cosx + signy * cosy*cosy));
Map[offset++] = res < 0 ? 0 : res;
```
Mode 1, analysis+synthesis windows (`ImgStream2::CreateFactorMap`, `filters.cpp:501-670`), 1D per side (`filters.cpp:509-553`):
```cpp
case 0: // half-cosine
  wanx[i] = cosf(pi*(i - ow + 0.5f) / (ow * 2));
  wsynx[i] = wanx[i];
  wany[i] = cosf(pi*(i - oh + 0.5f) / (oh * 2));
  wsyny[i] = wany[i];
case 1: // analysis: sqrt half-cosine, syntesis: (half cosine)^1.5
  wanx[i] = sqrt(cosf(pi*(i - ow + 0.5f) / (ow * 2)));
  wsynx[i] = wanx[i] * wanx[i] * wanx[i];
  wany[i] = sqrt(cosf(pi*(i - oh + 0.5f) / (oh * 2)));
  wsyny[i] = wany[i] * wany[i] * wany[i];
case 2: // analysis: 1, syntesis: raised cosine
  wanx[i] = 1.0;
  wsynx[i] = cosf(pi*(i - ow + 0.5f) / (ow * 2));
  wsynx[i] *= wsynx[i];
  wany[i] = 1.0;
  wsyny[i] = cosf(pi*(i - oh + 0.5f) / (oh * 2));
  wsyny[i] *= wsyny[i];
```
The 2D map is filled by bands (corners as the product `wanx*wany`, center as `1` or `wany` alone) with a particular channel order `i, i+1, i+2, i+3` where positions 2/3 receive `wanx[(ow-1)-x]` and `wanx[ow-x-2]` (mirror) — `filters.cpp:558-664`. Two separate LUTs: `FactorLUTana` (analysis) and `FactorLUTsyn` (synthesis), in `FLOAT4` (`filters.cpp:378-385`). If `NumSimultaneousRTs==1` and half-block overlap, the LUTs are repeated over the full `fdst` size (`filters.cpp:452-471`).

---

## 3. GPU memory management

### 3.1 TextureRT: GPU↔system exchange (the real "swap" mechanism)

Each `TextureRT` permanently keeps a **shadow texture in D3DPOOL_SYSTEMMEM** (`core/texture.cpp:283`) in addition to the GPU texture (`D3DPOOL_DEFAULT` + `D3DUSAGE_RENDERTARGET`, l.436).

- `SwapToMem()` (`core/texture.cpp:461-482`): `GetRenderTargetData(_pVideoSurface, _pShadowSurface)` (l.468) then releases the GPU texture as many times as `_RefCount`; `_SwapedToMem=true`; returns `_MemUsage = width*height*PixelSizeInByte` (`core/texture.cpp:241`).
- `SwapToGPU()` (l.497-520): recreates the texture, `UpdateTexture(_pShadowTexture, _pVideoTexture)` (l.517), re-adjusts the AddRefs, puts it back at the head of `LastUsedList`.
- `ReadyTexture()` (l.522-533): called before any use (rendertarget/texture/getdata); brings it back to the GPU if needed and updates the LRU.
- Release policy: `CreateVideoTexture` (l.432-446) — on `D3DERR_OUTOFVIDEOMEMORY`, loop `ReleaseMem(_MemUsage)` (l.535-545) which walks `LastUsedList` **from least recently used to most recent** and `SwapToMem`s until the requested size is freed; otherwise MessageBox: "fft3dgpu needs more memory than there are available on the graphics card… use usefloat16=true or mode 0 or 2" (l.444).
- Reference counting: `Release()` decrements and destroys the C++ object when `_RefCount<1` (l.377-404); `operator delete` only frees the memory if `_RefCount<=0` (l.359-374). The system shadow is only freed at destruction.

### 3.2 TexturePool (`TexturePool.h:24-40`, `TexturePool.cpp`)

Simple `std::stack<TextureRT*>` of textures **all of the same size/type** (created on demand, l.34-42, 55-65). `pop(pTextureRTpair&)` guarantees 2 textures (l.67-79). `push` sets the caller's pointer to 0 (l.49-53, 81-87). The destructor `delete`s everything (l.27-32). A single global pool per filter (`pFreeFImgdPool`, passed to the FFTs/filters, `FFT3dGPU.cpp:366-385, 463, 470-507`).

### 3.3 GPUCache (`GPUCache.h:24-67`, `GPUCache.cpp`)

**Simple LRU** cache (linked list, most recent at the head) of computed 2D FFTs, indexed by frame number `n`:
- `AddtoCache` (`GPUCache.cpp:75-97`): insertion at the head; if `_cachesize == _maxcachesize`, `deletelast` (the oldest node).
- Cache size = `bt` (number of temporal frames), `FFT3dGPU.cpp:419-430`; one cache per plane (`caches[0]` luma, `caches[1..2]` chroma), shared with `pFreeFImgdPool` via `StreamPoolPointer`.
- The node owns the textures; its destructor **does not destroy** them but **returns them to the pool**: `~GPUCacheNode(){ _StreamPool->push(_stream1); if (_stream2)_StreamPool->push(_stream2); }` (`GPUCache.h:33`).
- `GetStream` (l.157-191): linear search by `n`, AddRef of the textures found.
- `FlushCache`/`FlushAll` (l.195-213): `FlushAll` is called after a device reset (`FFT3dGPU.cpp:1400`).

The GPUCache does **not** do system swapping; it is TextureRT (§3.1) that does it, transparently, under VRAM pressure.

---

## 4. Creating the D3D9 device (`core/ManageDirect3DWindow.cpp`)

Architecture: a **hidden window** "FFT3dGPU" and a **dedicated message thread**; the device is created there and all `Reset`/`TestCooperativeLevel` operations are posted via `SendMessage` (the device must stay on a single thread).

- `GetDirectDirect3DDevice()` (l.112-121): first call → `Init()` (l.55-67) which creates the `MessageLoop` thread and waits for `einitdone`.
- `MessageLoop` (l.323-340): `InitDevice()` then the message loop; on exit, `DestroyDevice()` (l.94-109).
- DirectX version (`InitDevice`, l.153-171): `GetDXVersion` (Microsoft implementation in `core/getdxver.cpp:66+`, via dxdiag COM or file versions); requires `dwDirectXVersion >= 0x00090003` (9.0c) — MessageBox "FFT3dGPU need at least DirectX version 9.0c installed" (l.160-164).
- Window: `CreateWindow("FFT3dGPU", "FFT3dGPU Debug", WS_OVERLAPPEDWINDOW, 100,100, 1024,768, GetDesktopWindow(), ...)` (l.173-175). `WM_CLOSE` is intercepted → minimizes instead of closing (l.301-304).
- `d3dpp` (l.183-191):
```cpp
d3dpp.BackBufferHeight = 1024;
d3dpp.BackBufferWidth = 768;      // (largeur/hauteur inversées, volontaire ou bug historique)
d3dpp.BackBufferCount = 1;
d3dpp.Windowed = TRUE;
d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
d3dpp.BackBufferFormat = D3DFMT_UNKNOWN;
d3dpp.EnableAutoDepthStencil = FALSE;
d3dpp.AutoDepthStencilFormat = D3DFMT_D24S8;
```
- Adapter choice (l.193-207): `D3DADAPTER_DEFAULT` + `D3DDEVTYPE_HAL` (`DEVICETYPE`, l.30-34), **except** if an adapter with `Description=="NVIDIA NVPerfHUD"` is found → that adapter + `D3DDEVTYPE_REF`.
- `CreateDevice` (l.208-213) with flags:
```cpp
D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED | D3DCREATE_PUREDEVICE
```
(the vertices are pre-transformed `D3DFVF_XYZRHW`, so vertex processing is irrelevant in practice).
- Initial states (l.219-225): `ZWRITEENABLE=FALSE`, `ZFUNC=LESS`; samplers 0..5: `ADDRESSU/V = D3DTADDRESS_BORDER`, `BORDERCOLOR = 0x00000000`. (In `GetFrame`, the samplers are switched back to `MIRROR` — `FFT3dGPU.cpp:816-820`.)
- Caps checked:
  - `PixelShaderVersion` major ≥ 2: `if (((unsigned char*)&Cap.PixelShaderVersion)[1] < 2) env->ThrowError("Only pixelshader 2.0 or greater supported");` (`FFT3dGPU.cpp:205-211`).
  - Texture formats: probed by `GPUTYPES` with `CheckDeviceFormat` (§1.2).
  - `Cap.NumSimultaneousRTs > 1`: choice between "SP" (multi-RT) and "MP" (multi-pass) shaders — `filters.cpp:373-376`.
- Shaders: compiled **at runtime from the `fft3dgpu.hlsl` file** located next to the DLL (`shader.cpp:56-68`, `D3DXCompileShaderFromFile`, profile = `D3DXGetPixelShaderProfile(pDevice)`), with the parameters passed as preprocessor macros (`VecToMacroArray/IntToMacroArray/FloatToMacroArray/SetMacroArray`, `shader.cpp:136-227`).

### 4.1 Device lost

- Detection: `D3DDevice::DeviceLost()` (`FFT3dGPU.cpp:1397-1412`) calls `DeviceReady()` = `SendMessage(hWnd, TESTCOOPERATIVELEVEL)` → `pDevice->TestCooperativeLevel()` (`ManageDirect3DWindow.cpp:238-241, 310-314`).
- If lost: `ResetDevice()` (`ManageDirect3DWindow.cpp:243-287`): loop `while (hr != D3DERR_DEVICENOTRESET && hr != D3D_OK) Sleep(100)`; then `TextureRT::Reset(pDevice, true)` (destroys all DEFAULT textures via the LRU list `LastUsedList`, `core/texture.cpp:690-703, 723-766`), `Pixelshader::Reset(...,true)`, `pDevice->Reset(&d3dpp)` via the `RESETDEVICE` message (l.305-309), then recreation `TextureRT::Reset(...,false)` (re-creation + `UpdateTexture` from the shadow → **the content is preserved**), recreation of the query event, reset of the shaders.
- Then (`FFT3dGPU.cpp:1400-1406`): `GPUCache::FlushAll()` and marking `lost=true` on all `D3DDevice` instances.
- In `GetFrame` (`FFT3dGPU.cpp:1046-1064`): if `d3ddevice.DeviceLost()` → `kalmans[i]->Restore()` (the Kalman filters have persistent state across frames), return `dst` in all-plane mode, then **recursive restart** `return this->GetFrame(n, env)`.

---

## 5. AviSynth frame → texture mapping

### 5.1 Planar (YV12/YV16/YV24/Y8/YV411/planar RGB, 8/10-16/32 bits)

An `FFT3dGPU` filter processes either the luma (`plane=0`), or U+V (`plane=1`), or both via two instances chained by `FFT3dGPUallPlane` (`plane=4`, `FFT3dGPU.cpp:1260-1341`). Packed RGB is rejected: "FFT3dGPU: no packed RGB, only planar RGB is supported" (l.1243-1245). Greyscale + plane=1 rejected (l.1251-1253).

Each plane is uploaded **separately, one plane per texture** (`UploadImg[0]=Y/G, [1]=U/B, [2]=V/R`), never packed together. The `GetFrame` loop iterates `planeNo = 0` (luma) or `1→2` (chroma) (`FFT3dGPU.cpp:794-1071`), with `planes_y/planes_r` (l.789-791).

Upload (l.927-937):
```cpp
src = child->GetFrame(n, env);
if (!isYUY2)
  UploadToTexture(upload, src->GetReadPtr(currentPlane), src->GetPitch(currentPlane) / pixelsize);
```
→ **stride**: the AviSynth pitch (bytes) is divided by `pixelsize` to be expressed in elements (1/2/4 bytes); the copy handles a different D3D texture pitch (`LockedRect.Pitch`, `core/texture.cpp:885, 912`).

Optimized prefetch: after the download, the filter uploads the **next** frame (`nuploaded = lastn + delta`, l.1093-1104) into `UploadImg2` then `std::swap(UploadImg2, UploadImg1)` (l.1115, 1133-1135); if the requested frame is already on the GPU (`nuploaded == n`), it uses `upload1` without re-uploading (l.936-937, 992-993).

Download (l.1140-1166):
```cpp
DownloadFromTexture(DownloadImg[0], dst->GetWritePtr(currentPlane), dst->GetPitch(currentPlane) / pixelsize, true);
```
Untreated planes are copied as-is by `env->BitBlt` (l.1081-1091); in all-plane mode, `dst` is shared to avoid a BitBlt (`FFT3dGPUallPlane::GetDstFrame/ReturnDstFrame`, l.1216-1229).

### 5.2 YUY2 (packed)

`isYUY2 = vi.IsYUY2()` (`FFT3dGPU.cpp:169`). De-interleaved upload by byte offset:
```cpp
UploadInterleavedToFixedTexture(upload, src->GetReadPtr() + (planeNo == 0 ? 0 : planeNo == 1 ? 1 : 3), src->GetPitch(), planeNo == 0 ? 2 : 4);
```
(`FFT3dGPU.cpp:842, 882, 933, 988`; prefetch l.1114, 1130-1131). `srcoffset=2` for Y (all even bytes), `+1/+3` and `srcoffset=4` for U/V. The implementation (`core/texture.cpp:1129-1161`):
```cpp
for (int x = 0, doffset = dstInc - 1 - elem, soffset = (elem - 1)*srcoffset; x < width; x++, doffset += dstInc, soffset += srcInc * srcoffset)
```
→ texel x = (Y[2x], Y[2x+1]) in A8L8 (L=even, A=odd). Symmetric download: `DownloadFromFixedTextureInterleaved(DownloadImg[1], dst->GetWritePtr() + 1, dst->GetPitch(), true, 4)` (l.1163-1164, impl. `core/texture.cpp:1180-1233`). Note: in YUY2, `dst` is first pre-filled by a BitBlt of the source (l.782-786).

### 5.3 Packing summary

- 1 `FIXED2`/`FIXED2_16`/`FLOAT2` texel = 2 samples of the plane (consecutive pixel pairs, xy/rg or L/A channels).
- 1 `FIXED4`/`FIXED4_16`/`FLOAT4` texel (download) = 4 samples.
- The order of the 4 samples in the download texel depends on the format (comment `/*0123 becomes 2103 order at 16 bits*/`, `fft3dgpu.hlsl:984`), compensated for to allow the final memcpy.

---

## 6. Interlaced mode (`interlaced=true`, mode 1 only)

Shader selection: `ImgStream2` picks `psImg2ToFloat4ohalfInterlacedSP/MP` and `psFloat4ToImg2ohalfInterlaced` (half-block overlap) or the `oInterlaced` variants (`filters.cpp:392-395, 406-409, 423-426, 436-439`).

- **Number of blocks doubled in Y** (`FFT3dGPU.cpp:262-265, 272-275`):
```cpp
if (mode == 1 && interlaced)
  ny = 2 * (height / 2 + bh * 1.5 - 1) / bh;
...
if (mode == 1 && interlaced)
  ny = 2 * (((height / 2 + oh - 1) / (bh - oh) + 1 + 1) / 2);
```
- **Analysis**: the shader's `OFFSET` shift goes from `(0, (bh/2)/height)` (progressive, `filtersps.cpp:1198`) to **`(0, bh/height)`** (interlaced, `filtersps.cpp:1253`) — the second sample is taken one `bh` block lower, i.e. in the other field; the FFT blocks cover 2 field lines. Two quads are drawn: `quad` (upper half of the destination, `qrec={0,0,dstwidth,dstheight/2}`) and `bottomquad` with the src RECTs shifted by **+1 line** (`rec[1].top += 1; rec[2].top += 1;`, l.1282-1288) — one-field shift.
- **Rearrangement via StretchRect** (`Apply`, l.1345-1358): 12 RECTs `r[0..11]` (l.1293-1307) swap bands of `bh/2` lines between the top, the middle and the bottom of the two destination textures, via a temporary texture `temptex` (`TextureRT` copy constructor, l.1291) — re-phasing the fields before the FFT.
- **Synthesis**: field separation via the **Z-buffer** (`psFloat4ToImg2ohalfInterlaced`, `filtersps.cpp:1923-2100`). An alternating-line pattern is uploaded:
```cpp
for (int i = 0; i < height; i++)
  memset(t + i * width, (i & 1) * 255, width);
```
(l.1940-1941) then copied into the `D3DFMT_D24X8` depth-stencil by the `CopyTexToZBuffer` shader (`fft3dgpu.hlsl:1012-1018`, `dst.d=tex2D(Src,In).x;`) with `ZFUNC=D3DCMP_ALWAYS` (l.1945-1959). On download: first quad with `ZFUNC=D3DCMP_GREATER` (writes the even lines), `evenfieldquad` with `ZFUNC=D3DCMP_LESS` (odd lines), the second quad's src RECTs shifted by `srcheight/2` (l.1993-2000). The quad uses a constant `z=0.5` (`vertexbuffer.cpp:78`).
- The equivalent non-interlaced path (`psFloat4ToImg2ohalf`) draws both fields in a single quad (`filtersps.cpp:1869-1920`).

---

## 7. CalcMeanSD and SigmaToPatternTexture

### 7.1 `SigmaToPatternTexture` (`FFT3dGPU.cpp:588-631`)

Builds a 1D-frequency texture `Pattern` (TextureM `FLOAT`, size `fbw*nx × bh*ny`, `fbw = bw/2+1`) giving the **sigma per frequency** when sigma2≠sigma3≠sigma4 (`usePattern`, l.175). Called with `s = sigma*sigma*bt*bw*bh` (l.396) and sigmas 2-4 pre-multiplied:
```cpp
_sigma2 *= _sigma2 * bt*bw*bh / (255 * 255);
_sigma3 *= _sigma3 * bt*bw*bh / (255 * 255);
_sigma4 *= _sigma4 * bt*bw*bh / (255 * 255);
```
(l.401-403). Exact formulas (l.599-620):
```cpp
float ft2 = sqrt(0.5f) / 2; // frequency for sigma2
float ft3 = sqrt(0.5f) / 4; // frequency for sigma3
...
float fy = (bh - 2.0f*abs(h - bh / 2.0f)) / bh; // normalized to 1
float fx = (w*1.0f) / fbw;  // normalized to 1
float f = sqrt((fx*fx + fy * fy)*0.5f); // normalized to 1
if (f < ft3)
  pattern2d[offset + w] = sigma4 + (sigma3 - sigma4)*f / ft3;        // low
else if (f < ft2)
  pattern2d[offset + w] = sigma3 + (sigma2 - sigma3)*(f - ft3) / (ft2 - ft3); // middle
else
  pattern2d[offset + w] = sigma + (sigma2 - sigma)*(1 - f) / (1 - ft2);       // high
```
Then the pattern is repeated over `nx*ny` blocks (l.622-627) and `UploadToTexture(Pattern, pattern2drep)`. Note: constructor `sigma` = `_sigma / 255` (`FFT3dGPU.cpp:172`); `smin/=255; smax/=255` (l.187-188); everything is internalized in 0..1 ("Internally everything is normalized to the 0..1.0 range", l.392-394).

### 7.2 `CalcMeanSD` (`FFT3dGPU.cpp:647-667`)

Measures the average noise standard deviation from an FFT texture (PSD). **Not used** ("Not currently used because the window function screws this function somewhat up", l.645). Downloads the texture as float then:
```cpp
for (unsigned int i = 0; i < ny - 2; i++)
  for (unsigned int y = 0, o = 0; y < bh; y++, offset = 0, o += src->GetWidth()*src->GetType()->elem_size)
    for (unsigned int j = 0; j < (nx - 2); j++)
      for (unsigned int x = 0; x < bw / 2 + 1; x++) {
        if (y != 0 && x != 0) {
          mean += M[o + offset + 1] * M[o + offset + 1] + M[o + offset + 3] * M[o + offset + 3];
          count++;
        }
        offset += 4;
      }
mean /= count;
mean = sqrt(mean * 255 * 255 / bh / bw / bt);
```
Sum of `re²+im²` (the float4's .y and .w components), excluding the DC line `y==0` and the `x==0` column, and the first 2 border blocks (`ny-2`, `nx-2`); rescaling to 8 bits by `*255*255` and normalization `1/(bh*bw*bt)`.

---

## 8. Coordinate conversions — pitfalls for the port

1. **No explicit viewport**: no `SetViewport` in the code; the implicit D3D9 viewport = render target size. The quads are in **RHW** (`D3DFVF_XYZRHW`, `x,y` in RT pixels, `z=0.5f`, `rhw=1.0f`, `color=0x00808000` used as a weight in the MODE2 shaders) — `vertexbuffer.cpp:76-86`.
2. **Half-texel (texel centers)**: the quads' texture coordinates (`NQuad`, `vertexbuffer.cpp:87-99`):
```cpp
float etaW = 0.5 / W + (float)Rect[i + RectOffset].left / W;
float etaH = 0.5 / H + (float)Rect[i + RectOffset].top / H;
v[0].tex[i].u = etaW; v[0].tex[i].v = etaH;
v[1].tex[i].u = etaW + W0 / W; ...
```
The `0.5/W` places the sample at the center of the texel. In OpenCL, you must sample `src[x + dx]` with **integer** offsets and reproduce exactly the RECT offsets below, otherwise there is a half-pixel shift.
3. **Negative RECTs = block offsets**: the sampling offset is encoded in the source RECT (with `MIRROR` addressing of the sampler for the border). Mode 0/2 (`filters.cpp:80-90`):
```cpp
D3DXVECTOR2 offset = D3DXVECTOR2(0.25*(bw), 0.5*bh);
src.left -= 0.25*bw;  src.right -= 0.25*bw;
src.top -= 0.5*bh;    src.bottom -= 0.5*bh;
```
⚠️ units = **texels of the 2-in-1 packed texture** → `0.25*bw` texels = `bw/2` pixels (horizontal half-block shift), `0.5*bh` = one vertical half-block. Mode 1 SP (`filtersps.cpp:1212-1219`): `rec[1].left = -bw/2; rec[1].top = -bh/2;` with the `bw` parameter already divided by 2 at the call site (`filters.cpp:399` passes `bw / 2`).
4. **HLSL OFFSET in normalized texture coordinates**, computed on the C++ side: `Img2toImg4`: `(1.0/src->GetWidth(), 0)` = 1 texel = 2 pixels (`filtersps.cpp:339`); progressive `Img2ToFloat4`: `(0, (bh/2)/srcHeight)`, interlaced: `(0, bh/srcHeight)` (`filtersps.cpp:1198, 1253`); `Img2ToFloat4MP`: `(0, (bh/2)/srcHeight)` (`filtersps.cpp:1369`).
5. **Sampler addressing**: initialized to BORDER color 0 (`ManageDirect3DWindow.cpp:221-225`) but **systematically reset to MIRROR** at the start of each plane (`FFT3dGPU.cpp:816-820`); the `Factor` sampler temporarily switches to `WRAP` during mode 1 passes (`filtersps.cpp:1239, 1246`). Edge mirroring is therefore **an integral part of the algorithm** (block padding extends beyond the image via negative RECTs) — to reproduce explicitly in OpenCL (mirror `x → |x|` with edge, careful with the D3D "mirror" convention without repeating the edge texel).
6. **UNORM sampling**: the 8/16-bit normalization comes from the D3D sampler (value/255, /65535) and the final rounding/clamp from the write into the UNORM RTs (round-to-nearest, clamp [0,1]) — see §2.4. In OpenCL: `v/255.0f`, `v/65535.0f`, and manual `round` + clamp, including the 10-14 bit overflow behavior (§2.4).
7. **FFT in packed 2:1**: the FFT textures are `totw/2` wide (`FFT3dGPU.cpp:350-358`); the mode 1 `imgp->first/last` pairs contain the two half-offsets (`Img2toFloat4_2` with `texCoord[2]/[1]`, `fft3dgpu.hlsl:165-183`).
8. **Thread prefix**: `SetCacheHints → MT_SERIALIZED` (`FFT3dGPU.h:45-46, 74-77`) — the filter is never called in parallel; the D3D9 device is on a dedicated thread with `D3DCREATE_MULTITHREADED`. An OpenCL port must decide its own queue/context policy on its own.
9. **Special FFT quads** (`core/vertexbuffer.h:77-294`): `FFTQuad0/2`, `R2CFFTQuad`, `CollectQuad`, `C2RQuad` use vertex buffers indexed by butterfly stage (`viLUT`, `DrawEven/DrawOdd`) — in OpenCL, replace with computed indices; `MQuad` (mode 2) and `OQuad`/`OIQuad` (mode 1 non-half-block overlap, with `CreateVBi(..., bool even)` for interlaced) build per-block vertex buffers with `etaW = 0.5f/width` etc. (`vertexbuffer.cpp:379-382, 467-470`).

---

## 9. Miscellaneous useful for the port

- **Dxinput.h/.cpp**: DirectInput keyboard/mouse for NVPerfHUD debug only (`FFT3dGPU.cpp:180, 196-200, 1168-1173`) — to remove in the port.
- **getdxver.cpp**: DirectX version detection (Microsoft example, dxdiag COM + file versions), called only to require 9.0c — no OpenCL equivalent.
- **Constructor error messages** (`FFT3dGPU.cpp`): "Error Creating Direct3D Device" (195), "Only pixelshader 2.0 or greater supported" (211), "Valid modes for bt are -1,0,1,2,3,4" (233), "Failed Creating FFT3dGPU::Texture img" (360), "Faillure creating WiennerFilter" (410), "Faillure creating Sharp" (416), "Failed Creating ImgStream" (458), "Failed creating vertexbuffer" (524); VRAM MessageBox (`core/texture.cpp:444`) and DirectX (`ManageDirect3DWindow.cpp:162,168`).
- **Parameter validation**: bw/bh rounded down to the previous power of 2 via `(bw-1)` counted by divisions by 2, bounded to [4,512] (`FFT3dGPU.cpp:213-231`); `ow *= (float)bw/_bw; ow &= 0xFFFFFFFE;` then bounds `ow*2<=bw`, `oh*2<=bh` (236-242); `mode` outside 0..2 → 0 (234-235); `wintype` outside 0..2 → 0 (254-255); `bm=_bt` with `bt=max(_bt,1)` (173); `current = bm==1?0 : bm==4?2 : 1` (176); `nx/ny` depending on mode (258-282).
- **Kalman (bt=0)**: persistent state across frames (`kalmans[]`, `FFT3dGPU.cpp:431-453`), restored after device lost (`FFT3dGPU.cpp:1050-1056`) — to serialize in a port.
- **GPU synchronization**: `StartGPU()` = query event `Issue(D3DISSUE_END)` + `GetData(D3DGETDATA_FLUSH)`; `SleepWhileGPUwork()` loops `Sleep(1)` while `S_FALSE` (max 10 s) (`ManageDirect3DWindow.cpp:345-367`) — OpenCL equivalent: `clFinish`/event.
- **Filter signature**: `fft3dGPU(c[sigma]f[beta]f[bw]i[bh]i[bt]i[sharpen]f[plane]i[mode]i[bordersize]i[precision]i[NVPerf]b[degrid]f[scutoff]f[svr]f[smin]f[smax]f[kratio]f[ow]i[oh]i[wintype]i[interlaced]b[sigma2]f[sigma3]f[sigma4]f[oldfft]b)` (`FFT3dGPU.cpp:1359`), version "fft3dGPU ver 0.8.4" (l.1362).
