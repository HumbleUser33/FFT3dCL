# Documentation technique exhaustive — FFT3dGPU (AviSynth / DirectX 9) en vue d'un port OpenCL

Analyse complète des fichiers : `core/texture.h/.cpp`, `GPUCache.h/.cpp`, `TexturePool.h/.cpp`,
`core/ManageDirect3DWindow.h/.cpp`, `core/pixelshader.h/.cpp`, `core/vertexshader.h/.cpp`,
`core/shader.h/.cpp`, `core/vertexbuffer.h/.cpp`, `Dxinput.h/.cpp`, `core/getdxver.cpp`,
`FFT3dGPU.cpp` (constructeur, upload/download, CalcMeanSD, SigmaToPatternTexture),
`filters.h/.cpp`, `filtersps.cpp` et `fft3dgpu.hlsl` (pour les formules de conversion).

Toutes les références sont `fichier:ligne`. Les formules sont recopiées à l'identique.

---

## 0. Carte des composants

| Rôle | Fichiers |
|---|---|
| Filtre AviSynth, validation, upload/download CPU↔GPU | `FFT3dGPU.cpp`, `FFT3dGPU.h` |
| Abstraction textures D3D9 + formats GPU | `core/texture.h`, `core/texture.cpp` |
| Cache de FFT temporelles | `GPUCache.h/.cpp` |
| Pool de textures | `TexturePool.h/.cpp` |
| Device D3D9 (fenêtre cachée, reset, caps) | `core/ManageDirect3DWindow.h/.cpp`, `core/getdxver.cpp` |
| Shaders (compilation HLSL à chaud) | `core/shader.h/.cpp`, `core/pixelshader.*`, `core/vertexshader.*`, `fft3dgpu.hlsl` |
| Conversion image↔domaine fréquentiel, fenêtres | `filters.h/.cpp` (ImgStream/ImgStream2), `filtersps.h/.cpp` (wrappers de pixel shaders) |
| Quads/vertex buffers (coordonnées) | `core/vertexbuffer.h/.cpp` |
| Debug NVPerfHUD uniquement | `Dxinput.h/.cpp` |

---

## 1. Formats de textures D3D utilisés

### 1.1 Types de base (`core/texture.h`)

```cpp
enum gpu_types {
  _FLOAT = 4,
  _HALF = 2,
  _FIXED = 1,
  _FIXED16 = 3
};
```
(`core/texture.h:32-37`)

`types` = `{elem_size (canaux logiques), real_elem_size (canaux réels du format D3D choisi), format (D3DFORMAT), type, PixelSizeInByte = real_elem_size*GetByteSize(gt)}` (`core/texture.h:40-58`, tailles : `_FIXED`=1 octet, `_FIXED16`=2, `_HALF`=2, `_FLOAT`=4, `core/texture.h:41-49`).

`Types` = paire `{RenderTarget, Managed}` (`core/texture.h:60-63`) : **deux formats différents** sont sondés pour le même type logique, selon que la texture est une render target (`TextureRT`) ou une texture managée (`TextureM`).

### 1.2 Sonde de formats (`GPUTYPES`, `core/texture.cpp:42-193`)

Pour chaque type, `CheckDeviceFormat(..., D3DUSAGE_RENDERTARGET, D3DRTYPE_TEXTURE, fmt)` pour le pool RT, et le même appel avec usage `0` pour le pool Managed. Chaîne de fallback (si le format 2 canaux n'est pas supporté en RT, on retombe sur le 4 canaux, `real_elem_size=4`) :

| Type logique | Format RT/Managed préféré | Fallback |
|---|---|---|
| `_float4` | `D3DFMT_A32B32G32R32F` (l.51-54) | aucun (real_elem_size=0) |
| `_float2` | `D3DFMT_G32R32F` (l.56-59) | `_float4` |
| `_float1` | `D3DFMT_R32F` (l.61-64) | `_float2` |
| `_half4` | `D3DFMT_A16B16G16R16F` (l.66-69) | aucun |
| `_half2` | `D3DFMT_G16R16F` (l.71-74) | `_half4` |
| `_half1` | `D3DFMT_R16F` (l.76-79) | `_half2` |
| `_fixed4` | `D3DFMT_A8R8G8B8` (l.82-85) | aucun |
| `_fixed2` | `D3DFMT_A8L8` (l.88-91) — commentaire pinterf : « A8L8 render target is not allowed for me, can really fallback to fixed4 » | `_fixed4` |
| `_fixed1` | `D3DFMT_L8` (l.93-96) | `_fixed2` |
| `_fixed4_16` | `D3DFMT_A16B16G16R16` (l.99-102) | aucun |
| `_fixed2_16` | `D3DFMT_G16R16` (l.107-110) | `_fixed4_16` |
| `_fixed1_16` | `D3DFMT_L16` (l.112-115) | `_fixed2_16` |

Commentaire important (l.104-106) : « No luck with G13R16 render target… must be simulated ». Les formats 16 bits non signés UNORM sont donc la clé du chemin 10-16 bits.

Accesseurs : `FLOAT()/FLOAT2()/FLOAT4()/FIXED()/FIXED2()/FIXED4()/FIXED_16()/FIXED2_16()/FIXED4_16()/HALF()/HALF2()/HALF4()` (`core/texture.h:71-82`).

### 1.3 Textures d'upload/download selon profondeur (`FFT3dGPU.cpp:289-346`)

Pour la **luma** (instance `_plane<=0`), largeur `width = vi.width/_xRatio`, hauteur `height = vi.height/_yRatio` (`FFT3dGPU.cpp:172`) ; `_xRatio/_yRatio` = 1 pour la luma, = sous-échantillonnage UV pour la chroma (`FFT3dGPU.cpp:1269-1270`, `1300-1301`) :

| bits/comp | UploadImg[0..2] (TextureM, Managed) | taille | DownloadImg[0..2] (TextureRT) | taille |
|---|---|---|---|---|
| 8 | `FIXED2()` (A8L8) | `((width+1)>>1, height)` (l.293-295) | `FIXED4()` (A8R8G8B8) | `((width+3)>>2, height)` (l.296) |
| 10-16 | `FIXED2_16()` (G16R16) | `((width+1)>>1, height)` (l.299-301) | `FIXED4_16()` (A16B16G16R16) | `((width+3)>>2, height)` (l.302) |
| 32 float | `FLOAT2()` (G32R32F) | `((width+1)>>1, height)` (l.306-308) | `FLOAT4()` (A32B32G32R32F) | `((width+3)>>2, height)` (l.309) |

Chroma (instance chroma, `width` déjà divisé par le ratio UV) : `UploadImg[1]` et `[2]` en `((width+1)/2, height)`, `DownloadImg[1]`/`[2]` en `((width+3)/4, height)` (l.316-344).

**Convention de packing** : une texel `FIXED2` porte **2 échantillons** (paires de pixels), une texel `FIXED4` en porte **4**. C'est pour ça que la texture d'upload fait `width/2` et la download `width/4`.

Textures de travail intermédiaires (`FFT3dGPU.cpp:348-358`) :
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
Validation (`FFT3dGPU.cpp:244-252`) :
```cpp
// FLOAT16 precision is not enough for 10+ bits (s10e5), using FLOAT32_ALL for bits_per_pixel>8
if (bits_per_pixel > 8)
  precision = FLOAT32_ALL;

if (precision<FLOAT16 || precision>FLOAT32_ALL)
  precision = FLOAT32_FFT;

// now precision is final
useHalf = precision == FLOAT16;
```
Donc **en 10-16 bits, HALF n'est jamais utilisé** ; les textures de travail sont en FLOAT4. `useHalf` ne pilote que le format des textures FFT (`HALF4` vs `FLOAT4`) et des FactorLUT (`HALF2` vs `FLOAT2`, `filters.cpp:95-98`).

### 1.5 TextureM vs TextureRT vs paires

- **TextureM** (`core/texture.h:246-263`, `core/texture.cpp:769-786`) : `CreateTexture(..., D3DPOOL_MANAGED, ...)` (l.780). Utilisée pour : UploadImg (3×), FactorLUT, Pattern, gridsample. Upload = `LockRect` direct sur la texture managée (`core/texture.cpp:838-842`).
- **TextureRT** (`core/texture.h:128-206`, `core/texture.cpp:268-297`) : double buffer — `_pShadowTexture` en **D3DPOOL_SYSTEMMEM** (l.283) + `_pVideoTexture` en **D3DPOOL_DEFAULT** avec `D3DUSAGE_RENDERTARGET` (`CreateVideoTexture`, l.436). Upload : lock de la shadow puis `UpdateTexture(shadow, video)` (`SetDataEnd`, l.610-620). Download : `GetRenderTargetData(video, shadow)` puis `LockRect(D3DLOCK_READONLY)` (`GetData`, l.582-595).
- **pTextureRTpair** (`core/texture.h:209-244`) : paire de TextureRT avec compteur de références propre (`AddRef/Release`, auto-`delete` quand RefCount≤0). Utilisée en **mode 1** uniquement (deux moitiés de blocs, `imgp`, `FImgdp`, etc., `FFT3dGPU.h:125,148-160`).

---

## 2. Conversion des frames 8 bits et 10-16 bits (upload/download)

### 2.1 Ce qui est fait côté CPU (host)

**Aucune conversion numérique côté CPU.** `UploadToTexture` (`core/texture.cpp:867-973`) fait une copie octet/uint16/float par octet/uint16/float, avec réordonnancement de canaux si le format réel diffère du format logique :

- `_FIXED` (8 bits) : chemin rapide `memcpy` si `dstInc==2 && srcInc==2 && dstpitch==srcpitch` (l.914-917). Sinon « magic with alpha » (l.918-934) pour packager 2 octets dans A8R8G8B8 quand A8L8 n'est pas supporté :
  ```cpp
  int alpha = srcInc == dstInc ? dstInc - 1 : dstInc - 1 - srcInc;
  ...
  for (int elem = 1; elem < srcInc; elem++)
    for (int x = 0, doffset = dstInc - 1 - elem, soffset = elem - 1; ...)
      *(dstp + doffset) = *(srcp + soffset);
  for (int x = 0, doffset = alpha, soffset = srcInc - 1; ...)
      *(dstp + doffset) = *(srcp + soffset);
  ```
- `_FIXED16` (10-16 bits) : idem en `uint16_t`, memcpy rapide si `dstInc==2&&srcInc==2` (l.951-954).
- `_FLOAT`/`_HALF` : src est toujours `float*`, conversion float→D3DXFLOAT16 implicite par affectation (l.878-903).

Le **pitch** est passé en *unités d'éléments* et non en octets : `src->GetPitch(currentPlane) / pixelsize` (`FFT3dGPU.cpp:839, 1111, 1146`), `pixelsize = vi.ComponentSize()` (`FFT3dGPU.cpp:775`).

**Normalisation 8 bits et 10-16 bits : elle est faite par l'échantillonnage texture D3D** (UNORM8 → valeur/255, UNORM16 → valeur/65535), pas par le code. Les octets/uint16 sont stockés bruts dans A8L8/G16R16 ; tout shader qui fait `tex2D` reçoit un float normalisé.

### 2.2 Macros HLSL de conversion (`fft3dgpu.hlsl:21-39`)

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

Valeurs effectives de `CHROMA_CORR` (le macro `CHROMA_BIT_DEPTH` n'est défini **que pour les instances chroma non-RGB**, `useChromaDisplacementMacro = !isLumaPlane && !isRGB`, `FFT3dGPU.cpp:462` ; voir `DEFINE_A8L8_AND_CHROMA_MACROARRAY_MAX4`, `filtersps.cpp:875`) :
- **luma** (macro non définie → évaluée à 0 par le préprocesseur → branche `#elif`) : `CHROMA_CORR = 32768/65535 ≈ 0.5000076` ;
- **chroma 8 bits** : `CHROMA_CORR = 128/255 ≈ 0.5019608` ;
- **chroma 10-16 bits et float** : `CHROMA_CORR = 0`.

⚠️ Piège de port : la soustraction/reconstitution de DC (`-CHROMANORM` / `+CHROMANORM`, voir 2.4) ne vaut **que pour les shaders du mode 1** (`Img2ToFloat4*`, `Float4ToImg2*`). Le mode 0 (`Img2toFloat4`) ne centre pas. Et la valeur de `CHROMANORM` diffère entre luma (0.5000076) et chroma 8 bits (0.5019608).

### 2.3 Upload — mise à l'échelle 10-14 bits (`fft3dgpu.hlsl:106-135`)

Commentaire source (l.106-112) :
```
// Converts input (already normalized float) texture buffer to a four-unit buffer.
// When input was originally less than 16 bits but still uint16_t (typically 10, 12 and 14 bits),
// the converted range does not match with 0..1, rather 0 - 1023/65535 in a 10 bit input case.
// We stretch it to match with full 16 bit range in order to filtering parameters act the same
// way for all bit depths.
```
Shader d'analyse mode 0/2, `Img2toFloat4` (`fft3dgpu.hlsl:113-135`) :
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
Donc pour 10 bits : `v_uint16/65535 × 64` → plage 0..≈1. Pour 8 bits : `v/255` (aucun facteur). Pour 16 bits : `v/65535` (pas de BITDEPTHNORM, `BIT_DEPTH<16` faux). Pour float : `v` tel quel.

Sélecteur de swizzle `SW1` (`fft3dgpu.hlsl:95-104`) :
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
Le macro de format (`A8L8`/`G16R16`/`G32R32F`) est injecté depuis le C++ par `GETPIXEL2TYPEMACRO` (`filtersps.cpp:335`) :
```cpp
#define GETPIXEL2TYPEMACRO (src->GetType()->format==D3DFMT_A8L8 || src->GetType()->format==D3DFMT_A8R8G8B8 ? "A8L8" : src->GetType()->format==D3DFMT_G16R16 ? "G16R16" : src->GetType()->format==D3DFMT_G32R32F ? "G32R32F" : "A8L8")
```
Avec A8L8, le canal « alpha » de la texture porte l'octet impair du plan (le packing 2-échantillons/texel utilise L=pair, A=impair) — d'où `xw`.

Mode 1 (`Img2ToFloat4`, `fft3dgpu.hlsl:938-956`) — centre la donnée autour de 0 :
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
Variante multipass `Img2ToFloat4MP` (l.958-976) : `dst=(dst-CHROMANORM); return dst*factor;`

### 2.4 Download — arrondi, clamp, remise à l'échelle

Chaîne de download (mode 0/2, `filters.cpp:322-332`) :
```cpp
Float4toImg2->Apply(src, FactorLUT, Imgd);   // float4 -> Imgd (FIXED2/FIXED2_16/FLOAT2)
conv->Apply(Imgd, dst);                      // Img2toImg4: Imgd -> DownloadImg (FIXED4/...)
```
(mode 1 : `f2i->Apply(...)` puis `conv->Apply(...)` — `filters.cpp:692-697`).

Shaders de synthèse mode 1 (`fft3dgpu.hlsl:880-933`) — ré-ajoutent le DC :
```hlsl
// Float4ToImg2Corner:
float4 tmp=tex0*factortex0+tex1*factortex1;
dst=tmp.xyzw+tmp.zwxy;
return dst+CHROMANORM;
// Float4ToImg2Nooverlap:  return dst+CHROMANORM;
// Float4ToImg2Horizontal: dst=src1*factortex0+src2*factortex1; return dst+CHROMANORM;
// Float4ToImg2Vertical:   dst=src1*factor.xyxy+src2*factor.zwzw; return dst+CHROMANORM;
```

Shader final `Img2toImg4` (`fft3dgpu.hlsl:980-1003`), packager 2 texels 2-canaux → 1 texel 4-canaux :
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
`OFFSET = (1.0/src->GetWidth(), 0)` (un texel d'Imgd) et `BIT_DEPTH = bits_per_pixel` (`psImg2toImg4`, `filtersps.cpp:337-347`).

**Arrondi et clamp** : ils sont faits par l'écriture dans les render targets **UNORM** D3D9 (round-to-nearest + clamp [0,1]), puis par la copie brute :
- 8 bits : `Imgd` (A8L8) puis `DownloadImg` (A8R8G8B8) → l'octet final = `round(v×255)` clampé 0..255.
- 10-14 bits : double quantification UNORM16. `Imgd` (G16R16) quantifie `round(v×65535)` ; `Img2toImg4` fait `v/64` puis l'écriture dans A16B16G16R16 re-quantifie → entier final ≈ `round(v×1023.98)`, donc bien dans la plage 10 bits. ⚠️ Le `min(dst, MAX_PIXEL_VALUE)` avec MAX=1023 ne peut jamais se déclencher après `/64` (dst≤~1) ; un dépassement de plage (ringing, v>1) est clampé par le RT à 65535, valeur hors plage 10 bits recopiée telle quelle dans la frame. À reproduire (ou corriger) consciemment dans le port.
- 16 bits : aucune échelle, `round(v×65535)`.
- 32 bits float : `FLOAT4` (A32B32G32R32F), aucune quantification.

Copie finale vers la frame AviSynth (`DownloadFromTexture`, `core/texture.cpp:991-1109`) avec `IgnoreAlphaChannel=true` : chemin memcpy total pour les types 4-éléments :
```cpp
if (dstInc == 4 && srcInc == 4 && IgnoreAlphaChannel)
{
  for (int y = 0; y < height; y++, srcp += srcpitch, dstp += dstpitch)
    memcpy(dstp, srcp, srcpitch);          // _FIXED (l.1048-1053)
```
Commentaire source (l.1015-1017) : « we are using four-pixel units for final download / At non-8 bits the original 0 1 2 3 is changed into 2 1 0 3 / but we changed the order in Img2toImg4 in order to allow memcpy here ».

### 2.5 Fenêtres d'analyse/synthèse (facteurs multipliés aux mêmes endroits)

Mode 0/2 (`ImgStream::CreateFactorMap`, `filters.cpp:157-185`) :
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
Mode 2 avec border (`CreateFactorMapBorder`, `filters.cpp:205-234`) :
```cpp
cosy = cos(n1*pi / (y1 - 2 * bordery));
int signy = cosy / abs(cosy);
...
cosx = cos(n2*pi / (x1 - 2 * borderx));
int signx = cosx / abs(cosx);
double res = (0.5*(signx*cosx*cosx + signy * cosy*cosy));
Map[offset++] = res < 0 ? 0 : res;
```
Mode 1, fenêtres analyse+synthèse (`ImgStream2::CreateFactorMap`, `filters.cpp:501-670`), 1D par côté (`filters.cpp:509-553`) :
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
La 2D est remplie par bandes (coins en produit `wanx*wany`, centre en `1` ou `wany` seul) avec un ordre de canaux particulier `i, i+1, i+2, i+3` où les positions 2/3 reçoivent `wanx[(ow-1)-x]` et `wanx[ow-x-2]` (miroir) — `filters.cpp:558-664`. Deux LUT séparées : `FactorLUTana` (analyse) et `FactorLUTsyn` (synthèse), en `FLOAT4` (`filters.cpp:378-385`). Si `NumSimultaneousRTs==1` et recouvrement demi-bloc, les LUT sont répétées sur toute la taille `fdst` (`filters.cpp:452-471`).

---

## 3. Gestion mémoire GPU

### 3.1 TextureRT : échange GPU↔système (le vrai mécanisme de « swap »)

Chaque `TextureRT` garde en permanence une **shadow texture en D3DPOOL_SYSTEMMEM** (`core/texture.cpp:283`) en plus de la texture GPU (`D3DPOOL_DEFAULT` + `D3DUSAGE_RENDERTARGET`, l.436).

- `SwapToMem()` (`core/texture.cpp:461-482`) : `GetRenderTargetData(_pVideoSurface, _pShadowSurface)` (l.468) puis relâche la texture GPU autant de fois que `_RefCount` ; `_SwapedToMem=true` ; retourne `_MemUsage = width*height*PixelSizeInByte` (`core/texture.cpp:241`).
- `SwapToGPU()` (l.497-520) : recrée la texture, `UpdateTexture(_pShadowTexture, _pVideoTexture)` (l.517), ré-ajuste les AddRef, replace en tête de `LastUsedList`.
- `ReadyTexture()` (l.522-533) : appelé avant tout usage (rendertarget/texture/getdata) ; ramène sur le GPU si besoin et met à jour le LRU.
- Politique de libération : `CreateVideoTexture` (l.432-446) — si `D3DERR_OUTOFVIDEOMEMORY`, boucle `ReleaseMem(_MemUsage)` (l.535-545) qui parcourt `LastUsedList` **du moins récemment utilisé vers le plus récent** et `SwapToMem` jusqu'à libérer la taille demandée ; sinon MessageBox : « fft3dgpu needs more memory than there are available on the graphics card… use usefloat16=true or mode 0 or 2 » (l.444).
- Référence : `Release()` décrémentne et détruit l'objet C++ quand `_RefCount<1` (l.377-404) ; `operator delete` ne libère la mémoire que si `_RefCount<=0` (l.359-374). La shadow système n'est libérée qu'à la destruction.

### 3.2 TexturePool (`TexturePool.h:24-40`, `TexturePool.cpp`)

Simple `std::stack<TextureRT*>` de textures **toutes de même taille/type** (créées à la demande, l.34-42, 55-65). `pop(pTextureRTpair&)` garantit 2 textures (l.67-79). `push` met le pointeur appelant à 0 (l.49-53, 81-87). Le destructeur `delete` tout (l.27-32). Une seule pool globale par filtre (`pFreeFImgdPool`, passée aux FFT/filtres, `FFT3dGPU.cpp:366-385, 463, 470-507`).

### 3.3 GPUCache (`GPUCache.h:24-67`, `GPUCache.cpp`)

Cache **LRU simple** (liste chaînée, le plus récent en tête) de FFT 2D calculées, indexées par numéro de frame `n` :
- `AddtoCache` (`GPUCache.cpp:75-97`) : insertion en tête ; si `_cachesize == _maxcachesize`, `deletelast` (nœud le plus ancien).
- Taille du cache = `bt` (nombre de frames temporelles), `FFT3dGPU.cpp:419-430` ; un cache par plan (`caches[0]` luma, `caches[1..2]` chroma), partagé avec `pFreeFImgdPool` via `StreamPoolPointer`.
- Le nœud détenant les textures, leur destructeur **ne détruit pas** les textures mais les **rend au pool** : `~GPUCacheNode(){ _StreamPool->push(_stream1); if (_stream2)_StreamPool->push(_stream2); }` (`GPUCache.h:33`).
- `GetStream` (l.157-191) : recherche linéaire par `n`, AddRef des textures trouvées.
- `FlushCache`/`FlushAll` (l.195-213) : `FlushAll` est appelé après un reset de device (`FFT3dGPU.cpp:1400`).

Le GPUCache ne fait **pas** d'échange système ; c'est TextureRT (§3.1) qui le fait, de façon transparente, sous pression VRAM.

---

## 4. Création du device D3D9 (`core/ManageDirect3DWindow.cpp`)

Architecture : une **fenêtre cachée** « FFT3dGPU » et un **thread de message dédié** ; le device y est créé et toutes les opérations `Reset`/`TestCooperativeLevel` sont postées via `SendMessage` (le device doit rester sur un seul thread).

- `GetDirectDirect3DDevice()` (l.112-121) : premier appel → `Init()` (l.55-67) qui crée le thread `MessageLoop` et attend `einitdone`.
- `MessageLoop` (l.323-340) : `InitDevice()` puis boucle de messages ; à la sortie, `DestroyDevice()` (l.94-109).
- Version DirectX (`InitDevice`, l.153-171) : `GetDXVersion` (impl. Microsoft dans `core/getdxver.cpp:66+`, via dxdiag COM ou versions de fichiers) ; exige `dwDirectXVersion >= 0x00090003` (9.0c) — MessageBox « FFT3dGPU need at least DirectX version 9.0c installed » (l.160-164).
- Fenêtre : `CreateWindow("FFT3dGPU", "FFT3dGPU Debug", WS_OVERLAPPEDWINDOW, 100,100, 1024,768, GetDesktopWindow(), ...)` (l.173-175). `WM_CLOSE` est intercepté → minimise au lieu de fermer (l.301-304).
- `d3dpp` (l.183-191) :
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
- Choix de l'adaptateur (l.193-207) : `D3DADAPTER_DEFAULT` + `D3DDEVTYPE_HAL` (`DEVICETYPE`, l.30-34), **sauf** si un adaptateur dont `Description=="NVIDIA NVPerfHUD"` est trouvé → cet adaptateur + `D3DDEVTYPE_REF`.
- `CreateDevice` (l.208-213) avec flags :
```cpp
D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED | D3DCREATE_PUREDEVICE
```
(les sommets sont en `D3DFVF_XYZRHW` pré-transformés, donc le vertex processing est irrelevant en pratique).
- États initiaux (l.219-225) : `ZWRITEENABLE=FALSE`, `ZFUNC=LESS` ; samplers 0..5 : `ADDRESSU/V = D3DTADDRESS_BORDER`, `BORDERCOLOR = 0x00000000`. (Dans `GetFrame`, les samplers sont rebasculés en `MIRROR` — `FFT3dGPU.cpp:816-820`.)
- Caps vérifiés :
  - `PixelShaderVersion` majeure ≥ 2 : `if (((unsigned char*)&Cap.PixelShaderVersion)[1] < 2) env->ThrowError("Only pixelshader 2.0 or greater supported");` (`FFT3dGPU.cpp:205-211`).
  - Formats de textures : sondés par `GPUTYPES` avec `CheckDeviceFormat` (§1.2).
  - `Cap.NumSimultaneousRTs > 1` : choisi entre shaders « SP » (multi-RT) et « MP » (multi-pass) — `filters.cpp:373-376`.
- Shaders : compilés **au runtime depuis le fichier `fft3dgpu.hlsl`** situé à côté du DLL (`shader.cpp:56-68`, `D3DXCompileShaderFromFile`, profil = `D3DXGetPixelShaderProfile(pDevice)`), avec les paramètres passés en macros préprocesseur (`VecToMacroArray/IntToMacroArray/FloatToMacroArray/SetMacroArray`, `shader.cpp:136-227`).

### 4.1 Device lost

- Détection : `D3DDevice::DeviceLost()` (`FFT3dGPU.cpp:1397-1412`) appelle `DeviceReady()` = `SendMessage(hWnd, TESTCOOPERATIVELEVEL)` → `pDevice->TestCooperativeLevel()` (`ManageDirect3DWindow.cpp:238-241, 310-314`).
- Si perdu : `ResetDevice()` (`ManageDirect3DWindow.cpp:243-287`) : boucle `while (hr != D3DERR_DEVICENOTRESET && hr != D3D_OK) Sleep(100)` ; puis `TextureRT::Reset(pDevice, true)` (détruit toutes les textures DEFAULT via la liste LRU `LastUsedList`, `core/texture.cpp:690-703, 723-766`), `Pixelshader::Reset(...,true)`, `pDevice->Reset(&d3dpp)` via message `RESETDEVICE` (l.305-309), puis recréation `TextureRT::Reset(...,false)` (re-création + `UpdateTexture` depuis la shadow → **le contenu est préservé**), recréation du query event, reset des shaders.
- Ensuite (`FFT3dGPU.cpp:1400-1406`) : `GPUCache::FlushAll()` et marquage `lost=true` de toutes les instances `D3DDevice`.
- Dans `GetFrame` (`FFT3dGPU.cpp:1046-1064`) : si `d3ddevice.DeviceLost()` → `kalmans[i]->Restore()` (les filtres de Kalman ont un état persistant entre frames), renvoi du `dst` si mode all-plane, puis **relance récursive** `return this->GetFrame(n, env)`.

---

## 5. Mapping frames AviSynth → textures

### 5.1 Planar (YV12/YV16/YV24/Y8/YV411/planar RGB, 8/10-16/32 bits)

Un filtre `FFT3dGPU` traite soit la luma (`plane=0`), soit U+V (`plane=1`), soit les deux via deux instances chaînées par `FFT3dGPUallPlane` (`plane=4`, `FFT3dGPU.cpp:1260-1341`). RGB packed est rejeté : « FFT3dGPU: no packed RGB, only planar RGB is supported » (l.1243-1245). Greyscale + plane=1 rejeté (l.1251-1253).

Chaque plan est uploadé **séparement, un plan par texture** (`UploadImg[0]=Y/G, [1]=U/B, [2]=V/R`), jamais packés ensemble. La boucle de `GetFrame` itère `planeNo = 0` (luma) ou `1→2` (chroma) (`FFT3dGPU.cpp:794-1071`), avec `planes_y/planes_r` (l.789-791).

Upload (l.927-937) :
```cpp
src = child->GetFrame(n, env);
if (!isYUY2)
  UploadToTexture(upload, src->GetReadPtr(currentPlane), src->GetPitch(currentPlane) / pixelsize);
```
→ **stride** : le pitch AviSynth (octets) est divisé par `pixelsize` pour être exprimé en éléments (1/2/4 octets) ; la copie gère un pitch texture D3D différent (`LockedRect.Pitch`, `core/texture.cpp:885, 912`).

Préchargement optimisé : après le download, le filtre upload la frame **suivante** (`nuploaded = lastn + delta`, l.1093-1104) dans `UploadImg2` puis `std::swap(UploadImg2, UploadImg1)` (l.1115, 1133-1135) ; si la frame demandée est déjà en GPU (`nuploaded == n`), il utilise `upload1` sans re-upload (l.936-937, 992-993).

Download (l.1140-1166) :
```cpp
DownloadFromTexture(DownloadImg[0], dst->GetWritePtr(currentPlane), dst->GetPitch(currentPlane) / pixelsize, true);
```
Les plans non traités sont copiés tels quels par `env->BitBlt` (l.1081-1091) ; en mode all-plane, le `dst` est partagé pour éviter un BitBlt (`FFT3dGPUallPlane::GetDstFrame/ReturnDstFrame`, l.1216-1229).

### 5.2 YUY2 (packed)

`isYUY2 = vi.IsYUY2()` (`FFT3dGPU.cpp:169`). Upload désentrelacé par décalage d'octet :
```cpp
UploadInterleavedToFixedTexture(upload, src->GetReadPtr() + (planeNo == 0 ? 0 : planeNo == 1 ? 1 : 3), src->GetPitch(), planeNo == 0 ? 2 : 4);
```
(`FFT3dGPU.cpp:842, 882, 933, 988` ; prefetch l.1114, 1130-1131). `srcoffset=2` pour Y (tous les octets pairs), `+1/+3` et `srcoffset=4` pour U/V. L'implémentation (`core/texture.cpp:1129-1161`) :
```cpp
for (int x = 0, doffset = dstInc - 1 - elem, soffset = (elem - 1)*srcoffset; x < width; x++, doffset += dstInc, soffset += srcInc * srcoffset)
```
→ texel x = (Y[2x], Y[2x+1]) en A8L8 (L=pair, A=impair). Download symétrique : `DownloadFromFixedTextureInterleaved(DownloadImg[1], dst->GetWritePtr() + 1, dst->GetPitch(), true, 4)` (l.1163-1164, impl. `core/texture.cpp:1180-1233`). Note : en YUY2, `dst` est d'abord pré-rempli par un BitBlt de la source (l.782-786).

### 5.3 Résumé du packing

- 1 texel `FIXED2`/`FIXED2_16`/`FLOAT2` = 2 échantillons du plan (paires de pixels consécutifs, canaux xy/rg ou L/A).
- 1 texel `FIXED4`/`FIXED4_16`/`FLOAT4` (download) = 4 échantillons.
- L'ordre des 4 échantillons dans le texel de download dépend du format (commentaire `/*0123 becomes 2103 order at 16 bits*/`, `fft3dgpu.hlsl:984`), compensé pour permettre le memcpy final.

---

## 6. Mode entrelacé (`interlaced=true`, mode 1 uniquement)

Sélection des shaders : `ImgStream2` choisit `psImg2ToFloat4ohalfInterlacedSP/MP` et `psFloat4ToImg2ohalfInterlaced` (recouvrement demi-bloc) ou variantes `oInterlaced` (`filters.cpp:392-395, 406-409, 423-426, 436-439`).

- **Nombre de blocs doublé en Y** (`FFT3dGPU.cpp:262-265, 272-275`) :
```cpp
if (mode == 1 && interlaced)
  ny = 2 * (height / 2 + bh * 1.5 - 1) / bh;
...
if (mode == 1 && interlaced)
  ny = 2 * (((height / 2 + oh - 1) / (bh - oh) + 1 + 1) / 2);
```
- **Analyse** : le décalage `OFFSET` du shader passe de `(0, (bh/2)/height)` (progressif, `filtersps.cpp:1198`) à **`(0, bh/height)`** (entrelacé, `filtersps.cpp:1253`) — le second échantillon est pris un bloc `bh` plus bas, donc dans l'autre champ ; les blocs FFT couvrent 2 lignes de champ. Deux quads dessinés : `quad` (moitié haute de la destination, `qrec={0,0,dstwidth,dstheight/2}`) et `bottomquad` avec les RECT src décalés de **+1 ligne** (`rec[1].top += 1; rec[2].top += 1;`, l.1282-1288) — décalage d'un champ.
- **Réarrangement par StretchRect** (`Apply`, l.1345-1358) : 12 RECT `r[0..11]` (l.1293-1307) échangent des bandes de `bh/2` lignes entre le haut, le milieu et le bas des deux textures de destination, via une texture temporaire `temptex` (copie constructeur de `TextureRT`, l.1291) — remise en phase des champs avant FFT.
- **Synthèse** : séparation des champs par **Z-buffer** (`psFloat4ToImg2ohalfInterlaced`, `filtersps.cpp:1923-2100`). Un motif en lignes alternées est uploadé :
```cpp
for (int i = 0; i < height; i++)
  memset(t + i * width, (i & 1) * 255, width);
```
(l.1940-1941) puis copié dans le depth-stencil `D3DFMT_D24X8` par le shader `CopyTexToZBuffer` (`fft3dgpu.hlsl:1012-1018`, `dst.d=tex2D(Src,In).x;`) avec `ZFUNC=D3DCMP_ALWAYS` (l.1945-1959). Au download : premier quad avec `ZFUNC=D3DCMP_GREATER` (écrit les lignes paires), `evenfieldquad` avec `ZFUNC=D3DCMP_LESS` (lignes impaires), les RECT src du second quad décalés de `srcheight/2` (l.1993-2000). Le quad utilise `z=0.5` constant (`vertexbuffer.cpp:78`).
- Le path non-entrelacé équivalent (`psFloat4ToImg2ohalf`) dessine les deux champs dans un seul quad (`filtersps.cpp:1869-1920`).

---

## 7. CalcMeanSD et SigmaToPatternTexture

### 7.1 `SigmaToPatternTexture` (`FFT3dGPU.cpp:588-631`)

Construit une texture 1D-fréquences `Pattern` (TextureM `FLOAT`, taille `fbw*nx × bh*ny`, `fbw = bw/2+1`) donnant le **sigma par fréquence** quand sigma2≠sigma3≠sigma4 (`usePattern`, l.175). Appelé avec `s = sigma*sigma*bt*bw*bh` (l.396) et sigmas 2-4 pré-multipliés :
```cpp
_sigma2 *= _sigma2 * bt*bw*bh / (255 * 255);
_sigma3 *= _sigma3 * bt*bw*bh / (255 * 255);
_sigma4 *= _sigma4 * bt*bw*bh / (255 * 255);
```
(l.401-403). Formules exactes (l.599-620) :
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
Puis répétition du motif sur `nx*ny` blocs (l.622-627) et `UploadToTexture(Pattern, pattern2drep)`. Note : `sigma` constructeur = `_sigma / 255` (`FFT3dGPU.cpp:172`) ; `smin/=255; smax/=255` (l.187-188) ; tout est internalisé en 0..1 (« Internally everything is normalized to the 0..1.0 range », l.392-394).

### 7.2 `CalcMeanSD` (`FFT3dGPU.cpp:647-667`)

Mesure l'écart-type moyen du bruit depuis une texture FFT (PSD). **Non utilisé** (« Not currently used because the window function screws this function somewhat up », l.645). Télécharge la texture en float puis :
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
Somme de `re²+im²` (composantes .y et .w du float4), excluant la ligne DC `y==0` et la colonne `x==0`, et les 2 premiers blocs de bordure (`ny-2`, `nx-2`) ; remise à l'échelle 8 bits par `*255*255` et normalisation `1/(bh*bw*bt)`.

---

## 8. Conversions de coordonnées — pièges pour le port

1. **Pas de viewport explicite** : aucun `SetViewport` dans le code ; le viewport D3D9 implicite = taille du render target. Les quads sont en **RHW** (`D3DFVF_XYZRHW`, `x,y` en pixels du RT, `z=0.5f`, `rhw=1.0f`, `color=0x00808000` utilisé comme poids dans les shaders MODE2) — `vertexbuffer.cpp:76-86`.
2. **Demi-texel (texel centers)** : coordonnées de texture des quads (`NQuad`, `vertexbuffer.cpp:87-99`) :
```cpp
float etaW = 0.5 / W + (float)Rect[i + RectOffset].left / W;
float etaH = 0.5 / H + (float)Rect[i + RectOffset].top / H;
v[0].tex[i].u = etaW; v[0].tex[i].v = etaH;
v[1].tex[i].u = etaW + W0 / W; ...
```
Le `0.5/W` place l'échantillon au centre du texel. En OpenCL, il faut échantillonner `src[x + dx]` avec des décalages **entiers** et reproduire exactement les décalages de RECT ci-dessous, sinon décalage d'un demi-pixel.
3. **RECT négatifs = décalages de blocs** : le décalage d'échantillonnage est encodé dans le RECT source (avec addressing `MIRROR` du sampler pour le bord). Mode 0/2 (`filters.cpp:80-90`) :
```cpp
D3DXVECTOR2 offset = D3DXVECTOR2(0.25*(bw), 0.5*bh);
src.left -= 0.25*bw;  src.right -= 0.25*bw;
src.top -= 0.5*bh;    src.bottom -= 0.5*bh;
```
⚠️ unités = **texels de la texture packée 2-en-1** → `0.25*bw` texels = `bw/2` pixels (décalage horizontal d'un demi-bloc), `0.5*bh` = un demi-bloc vertical. Mode 1 SP (`filtersps.cpp:1212-1219`) : `rec[1].left = -bw/2; rec[1].top = -bh/2;` avec le paramètre `bw` déjà divisé par 2 à l'appel (`filters.cpp:399` passe `bw / 2`).
4. **OFFSET HLSL en coordonnées de texture normalisées**, calculé côté C++ : `Img2toImg4` : `(1.0/src->GetWidth(), 0)` = 1 texel = 2 pixels (`filtersps.cpp:339`) ; `Img2ToFloat4` progressif : `(0, (bh/2)/srcHeight)`, entrelacé : `(0, bh/srcHeight)` (`filtersps.cpp:1198, 1253`) ; `Img2ToFloat4MP` : `(0, (bh/2)/srcHeight)` (`filtersps.cpp:1369`).
5. **Addressing des samplers** : initialisé BORDER couleur 0 (`ManageDirect3DWindow.cpp:221-225`) mais **systématiquement remis à MIRROR** au début de chaque plan (`FFT3dGPU.cpp:816-820`) ; le sampler `Factor` passe temporairement en `WRAP` pendant les passes mode 1 (`filtersps.cpp:1239, 1246`). Le miroir des bords est donc **partie intégrante de l'algo** (le padding des blocs dépasse l'image via RECT négatifs) — à reproduire explicitement en OpenCL (miroir `x → |x|` avec bord, attention à la convention D3D « mirror » sans répétition du texel de bord).
6. **Échantillonnage UNORM** : la normalisation 8/16 bits vient du sampler D3D (valeur/255, /65535) et l'arrondi/clamp final de l'écriture dans les RT UNORM (round-to-nearest, clamp [0,1]) — voir §2.4. En OpenCL : `v/255.0f`, `v/65535.0f`, et `round` + clamp à la main, y compris le comportement de dépassement 10-14 bits (§2.4).
7. **FFT en packed 2:1** : les textures FFT font `totw/2` de large (`FFT3dGPU.cpp:350-358`) ; les paires `imgp->first/last` du mode 1 contiennent les deux demi-décalages (`Img2toFloat4_2` avec `texCoord[2]/[1]`, `fft3dgpu.hlsl:165-183`).
8. **Préfixe de thread** : `SetCacheHints → MT_SERIALIZED` (`FFT3dGPU.h:45-46, 74-77`) — le filtre n'est jamais appelé en parallèle ; le device D3D9 est sur un thread dédié avec `D3DCREATE_MULTITHREADED`. Un port OpenCL doit décider seul de sa politique de files/contexts.
9. **Quads FFT spéciaux** (`core/vertexbuffer.h:77-294`) : `FFTQuad0/2`, `R2CFFTQuad`, `CollectQuad`, `C2RQuad` utilisent des vertex buffers indexés par étape de papillon (`viLUT`, `DrawEven/DrawOdd`) — en OpenCL, remplacer par des indices calculés ; `MQuad` (mode 2) et `OQuad`/`OIQuad` (mode 1 recouvrement non demi-bloc, avec `CreateVBi(..., bool even)` pour l'entrelacé) construisent des vertex buffers par blocs avec `etaW = 0.5f/width` etc. (`vertexbuffer.cpp:379-382, 467-470`).

---

## 9. Divers utiles au port

- **Dxinput.h/.cpp** : clavier/souris DirectInput pour le debug NVPerfHUD uniquement (`FFT3dGPU.cpp:180, 196-200, 1168-1173`) — à supprimer dans le port.
- **getdxver.cpp** : détection de version DirectX (exemple Microsoft, dxdiag COM + versions de fichiers), appelée seulement pour exiger 9.0c — sans équivalent en OpenCL.
- **Messages d'erreur du constructeur** (`FFT3dGPU.cpp`) : « Error Creating Direct3D Device » (195), « Only pixelshader 2.0 or greater supported » (211), « Valid modes for bt are -1,0,1,2,3,4 » (233), « Failed Creating FFT3dGPU::Texture img » (360), « Faillure creating WiennerFilter » (410), « Faillure creating Sharp » (416), « Failed Creating ImgStream » (458), « Failed creating vertexbuffer » (524) ; MessageBox VRAM (`core/texture.cpp:444`) et DirectX (`ManageDirect3DWindow.cpp:162,168`).
- **Validation des paramètres** : bw/bh arrondis à la puissance de 2 inférieure via `(bw-1)` compté en divisions par 2, bornés [4,512] (`FFT3dGPU.cpp:213-231`) ; `ow *= (float)bw/_bw; ow &= 0xFFFFFFFE;` puis bornes `ow*2<=bw`, `oh*2<=bh` (236-242) ; `mode` hors 0..2 → 0 (234-235) ; `wintype` hors 0..2 → 0 (254-255) ; `bm=_bt` avec `bt=max(_bt,1)` (173) ; `current = bm==1?0 : bm==4?2 : 1` (176) ; `nx/ny` selon mode (258-282).
- **Kalman (bt=0)** : état persistant entre frames (`kalmans[]`, `FFT3dGPU.cpp:431-453`), restauré après device lost (`FFT3dGPU.cpp:1050-1056`) — à sérialiser dans un port.
- **Synchronisation GPU** : `StartGPU()` = query event `Issue(D3DISSUE_END)` + `GetData(D3DGETDATA_FLUSH)` ; `SleepWhileGPUwork()` boucle `Sleep(1)` tant que `S_FALSE` (max 10 s) (`ManageDirect3DWindow.cpp:345-367`) — équivalent OpenCL : `clFinish`/event.
- **Signature du filtre** : `fft3dGPU(c[sigma]f[beta]f[bw]i[bh]i[bt]i[sharpen]f[plane]i[mode]i[bordersize]i[precision]i[NVPerf]b[degrid]f[scutoff]f[svr]f[smin]f[smax]f[kratio]f[ow]i[oh]i[wintype]i[interlaced]b[sigma2]f[sigma3]f[sigma4]f[oldfft]b)` (`FFT3dGPU.cpp:1359`), version « fft3dGPU ver 0.8.4 » (l.1362).
