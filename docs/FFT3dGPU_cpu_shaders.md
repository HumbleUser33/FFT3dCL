# FFT3dGPU — Documentation technique exhaustive des classes CPU pilotant les pixel shaders HLSL (port OpenCL)

Sources analysées : `FFT3dGPU/FFT3dGPU/filtersps.h` (418 l.), `filtersps.cpp` (2100 l.), `filters.h` (126 l.), `filters.cpp` (1001 l.), `fft3dgpu.hlsl` (1205 l.), `FFT3dGPU.cpp/.h`, `core/shader.{h,cpp}`, `core/texture.{h,cpp}`, `core/vertexbuffer.{h,cpp}`, `core/ManageDirect3DWindow.cpp`.
Toutes les références sont `fichier:ligne`.

---

## 0. Mécanismes transversaux (indispensables pour le port)

### 0.1 Compilation des shaders
- Fichier HLSL unique : `#define SRC_SHADER "fft3dgpu.hlsl"` (`core/shader.h:28`), résolu en chemin absolu à côté du DLL (`core/shader.cpp:54-68`), compilé à l'exécution par `D3DXCompileShaderFromFile` avec la table de macros `defs` (`core/shader.cpp:68`).
- Chaque classe passe des **defines de préprocesseur** différentes → le même fichier produit des variantes. Pour OpenCL : une source, plusieurs `-D`.
- Helpers de macros (`core/shader.cpp`) :
  - `VecToMacroArray(vec,name)` → `#define name float2(x,y)` (`shader.cpp:136-146`, format `std::fixed`)
  - `FloatToMacroArray(f,name)` → `#define name f` (`shader.cpp:194-204`, `std::fixed`)
  - `IntToMacroArray(i,name)` → `#define name i` (`shader.cpp:165-175`)
  - `SetMacroArray(name)` → `#define name` (vide) (`shader.cpp:221-227`)
  - `CreateMacroArray(size)` → tableau de `size+1` entrées initialisées à 0 (`shader.cpp:253-262`)
- Attention piège : quand un define n'est pas voulu, le code passe parfois `" "` comme **nom** de macro (`filtersps.cpp:88`, `238`) — un define nommé espace, inoffensif en HLSL.

### 0.2 Registres de sampler HLSL (fixes)
`fft3dgpu.hlsl:44-50` :
```
sampler Src : register(s0);
sampler Src2 : register(s2);
sampler I : register(s3);
sampler J : register(s4);
sampler Factor : register(s1);
sampler FactorM : register(s5);
sampler LoadMap : register(s6);
```
Wienner (`hlsl:636-641`) : `prev2`, `prev1` non registrés (assignés par le compilateur), `current : s3`, `next : s4`, `degrid : s1`, `sigma` non registré.
Kalman (`hlsl:782-785`) : `Last : s1`, `Covar : s2`, `CovarProcess : s3`, `CovarNoise : s4`.
L'index de stage réel est lu dynamiquement via `GetSamplerIndex` (`core/shader.cpp:106-117`).

### 0.3 État des samplers (à reproduire en OpenCL)
- À la création du device, stages 0..5 : `ADDRESSU=ADDRESSV=D3DTADDRESS_BORDER`, `BORDERCOLOR=0x00000000` (`core/ManageDirect3DWindow.cpp:221-225`).
- **Au début de chaque frame**, stages 0..5 sont remis à `ADDRESSU=ADDRESSV=D3DTADDRESS_MIRROR` (`FFT3dGPU.cpp:816-820`, idem au constructeur pour la phase degrid : `FFT3dGPU.cpp:533-537`).
- Filtrage : jamais de `MINFILTER/MAGFILTER` posé dans le code de production (défaut D3D9 = **POINT**). Seul le code commenté `psMeanSD` touchait `MINFILTER` (`filtersps.cpp:858,869`). Donc **tous les tex2D sont en filtrage point**, sauf mention contraire.
- Exceptions temporaires dans les classes Img2ToFloat4/Float4ToImg2 « ohalf » : le stage de `Factor` passe en `ADDRESSU=WRAP` pendant le draw puis revient à `MIRROR` (`filtersps.cpp:1239/1246`, `1332/1360`, `1414/1427`, `1516/1549`, `1911/1918`, `2076/2094`). ADDRESSV reste MIRROR.
- Équivalent OpenCL : `CLK_ADDRESS_MIRRORED_REPEAT` + `CLK_FILTER_NEAREST` partout, avec `CLK_ADDRESS_CLAMP_TO_EDGE`+bordure 0 au démarrage du device, et `CLK_ADDRESS_REPEAT` sur l'axe U du LUT factor pour les variantes « ohalf ».

### 0.4 Formats de textures (`GPUTYPES`, `core/texture.cpp:42-193`)
| Type logique | Format D3D (RenderTarget / Managed) | Éléments |
|---|---|---|
| FLOAT4 | `D3DFMT_A32B32G32R32F` (`texture.cpp:51-54,118-121`) | 4×float32 |
| FLOAT2 | `D3DFMT_G32R32F` (`texture.cpp:56-59,123-131`) | 2×float32 |
| FLOAT1 | `D3DFMT_R32F` (`texture.cpp:61-64,129-131`) | float32 |
| HALF4 | `D3DFMT_A16B16G16R16F` (`texture.cpp:66-69,133-136`) | 4×float16 |
| HALF2 | `D3DFMT_G16R16F` (`texture.cpp:71-74,138-146`) | 2×float16 |
| HALF1 | `D3DFMT_R16F` (`texture.cpp:76-79,143-146`) | float16 |
| FIXED4 (8 bit) | `D3DFMT_A8R8G8B8` (`texture.cpp:82-85,149-152`) | 4×uint8 normalisé 0..1 |
| FIXED2 (8 bit) | `D3DFMT_A8L8` (`texture.cpp:88-91,154-157`) | 2×uint8 |
| FIXED1 (8 bit) | `D3DFMT_L8` (`texture.cpp:93-96,159-162`) | uint8 |
| FIXED4_16 (10-16 bit) | `D3DFMT_A16B16G16R16` (`texture.cpp:99-102,175-178`) | 4×uint16 |
| FIXED2_16 | `D3DFMT_G16R16` (`texture.cpp:107-110,180-183`) | 2×uint16 |
| FIXED1_16 | `D3DFMT_L16` (`texture.cpp:112-115,185-188`) | uint16 |
Fallbacks si le format n'est pas supporté en RT (ex. A8L8 RT refusé, cf. commentaire `texture.cpp:87`) : on retombe sur le format 4 composantes. `types` porte `elem_size` (nb de composantes logiques) et `real_elem_size` (composantes physiques) (`core/texture.h:40-58`).

Sélection selon bit depth (`FFT3dGPU.cpp:292-345`, `filters.cpp:65-74`) : `bits_per_pixel==8` → FIXED2/FIXED4 ; `<=16` → FIXED2_16/FIXED4_16 ; sinon (float) → FLOAT2/FLOAT4.
Textures de travail FFT (`FFT3dGPU.cpp:348-358`) : `img`/`imgp` en **HALF4** si `useHalf` (precision==FLOAT16) sinon **FLOAT4**, taille `totw/2 × toth` avec `totw=nx*bw`, `toth=ny*bh` (`FFT3dGPU.cpp:284-285`).
`precision` : enum `FLOAT16=0, FLOAT32_FFT=1, FLOAT32_ALL=2` (`FFT.h:40-42`) ; **si bits_per_pixel>8, precision est forcée à FLOAT32_ALL** (`FFT3dGPU.cpp:245-246`).

### 0.5 Mapping des coordonnées (NQuad) — point critique du port
`NQuad(pDevice, RECT* Rect, int size, int RectOffset=0, bool IndependentOffset=false)` (`core/vertexbuffer.h:179`, impl `core/vertexbuffer.cpp:61-110`) :
- `Rect[0]` = rectangle de la **cible de rendu** (le quad est dessiné en pixels écran 1:1 sur ce rectangle).
- Pour chaque stage `i` de 0 à `size-1`, les coordonnées de texture viennent de `Rect[i+RectOffset]` :
```
W  = Rect[i+RectOffset].right - left ; H = ...
etaW = 0.5/W + left/W          // correction demi-texel
etaH = 0.5/H + top/H
u: etaW → etaW + W0/W          (W0,H0 = taille du quad, ou W,H si IndependentOffset)
v: etaH → etaH + H0/H
```
(`vertexbuffer.cpp:88-99`). C'est ce qui encode les décalages d'un pixel source vers une autre texture. Un décalage `offset` en pixels se fait en translantant le RECT source (`rect[2].left += offset.x` etc.).
- `pTextureRTpair` = paire de `TextureRT*` (`first`,`last`) avec compteur de refs (`core/texture.h:209-244`) ; utilisée en mode 1 (2 textures de blocs au lieu d'une).

### 0.6 Macros HLSL communes aux conversions
- `SW1` (choix des composantes lues dans une texture 2-composantes) `hlsl:95-104` :
```
#if defined(G16R16) || defined(G32R32F)  → SW1 = rg
#elif defined(A8L8)                      → SW1 = xw
#else                                     SW1 = xy
```
- Macro CPU `GETPIXEL2TYPEMACRO` (`filtersps.cpp:335`) :
```
(src->GetType()->format==D3DFMT_A8L8 || ==D3DFMT_A8R8G8B8 ? "A8L8"
 : ==D3DFMT_G16R16 ? "G16R16" : ==D3DFMT_G32R32F ? "G32R32F" : "A8L8")
```
Posée seulement si `real_elem_size==2 || (real_elem_size==4 && elem_size==2)` (texture 2 composantes logiques).
- `BIT_DEPTH` (int) et normalisation 10-14 bit (`hlsl:35-39`) :
```
#if BIT_DEPTH>8 && BIT_DEPTH<16
BITDEPTHNORM     = float4(1<<(16-BIT_DEPTH), ...)      // multiplication à l'entrée
INV_BITDEPTHNORM = float4(1.0/(1<<(16-BIT_DEPTH)),...) // division à la sortie
MAX_PIXEL_VALUE  = float4((1<<BIT_DEPTH)-1, ...)       // clamp min(dst, MAX) à la sortie
```
- `CHROMA_BIT_DEPTH`/`CHROMANORM` (`hlsl:21-33`) : correction de décalage chroma YUY2 ; `CHROMA_CORR = RANGE_HALF/RANGE_FULL` = `32768/65535` si bit depth ≤16, `0` sinon ; `CHROMANORM = float4(c,c,c,c)`. Activée côté CPU par le booléen `chroma` = `!isLumaPlane && !isRGB` (`FFT3dGPU.cpp:462`).

---

## 1. psMinimize (classe « selectDC » du degrid)

**Déclaration** `filtersps.h:24-30` — **Impl** `filtersps.cpp:27-49`.

1. **Shader** : fonction HLSL `Minimize` (`hlsl:617-620`), **aucun define** :
```hlsl
float4 Minimize( PS_INPUT In) :COLOR
{
	return tex2D(Src,In.texCoord).xyxy;
}
```
2. **Entrées** : 1 texture `Src` (stage via `GetConstantByName("Src")`, `filtersps.cpp:29`). C'est la texture FFT complète `FImg2d[i]`/`ping`/`FImgDegridp.first` : FLOAT4 ou HALF4, `totw/2 × toth`. Sampler : MIRROR/MIRROR, point.
3. **Sortie** : `mintex` = `TextureRT` **FLOAT4**, taille **nx × ny** (`FFT3dGPU.cpp:516`) — un pixel par bloc, contenant la valeur DC du bloc (composantes .xy de la texture FFT au coin de chaque bloc, dupliquée en xyxy).
4. **Constantes** : aucune.
5. **Conditions** : créée seulement si `degrid != 0` (`FFT3dGPU.cpp:514-520`, nommée `selectDC`).
6. **Ordre d'appel** : avant `calcgridcorrection` à chaque frame (`FFT3dGPU.cpp:547, 856, 896, 898, 947, 1001, 1003`) ; au constructeur, une fois sur la frame 0 (`FFT3dGPU.cpp:547`).
7. **Géométrie** : `NQuad(rec[2]={dst->GetRect(), src->GetRect()}, 1, 1, true)` (`filtersps.cpp:31-32`) : 1 stage, RectOffset=1, IndependentOffset=true → le quad nx×ny échantillonne la texture FFT à l'échelle bloc→pixel.

---

## 2. psGridCorrection (degrid)

**Déclaration** `filtersps.h:32-40` — **Impl** `filtersps.cpp:51-83`.

1. **Shader** : fonction HLSL `GridCorrection` (`hlsl:622-632`) :
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
Deux variantes compilées :
- sans `DGRID` : instance `calcgridcorrection` (`FFT3dGPU.cpp:518`, ctor `filtersps.cpp:51-57`) ;
- avec `DGRID` = float `degrid` du script (`FloatToMacroArray(degrid,"DGRID",...)`, `filtersps.cpp:59-66`) : instance `setupgridcorrection`, utilisée une seule fois au constructeur (`FFT3dGPU.cpp:519, 548, 557, 570`).
2. **Entrées** :
   - `Src` : soit la texture FFT (setup : `FImgDegrid` = FFT de la frame 0 ; runtime : `mintex` nx×ny FLOAT4),
   - `Factor` : soit `mintex` (nx×ny FLOAT4) au setup, soit `gridsample` = `TextureM` **FLOAT4** `temp->GetWidth()×GetHeight()` téléchargée/uploadée en CPU au constructeur (`FFT3dGPU.cpp:560-565`).
3. **Sortie** : `TextureRT` du pool (FLOAT4/HALF4 `totw/2×toth`) → `FImgDegrid` / `FImgDegridp.first|last` : **motif de grille à soustraire** avant filtrage.
4. **Constantes** : `DGRID` = paramètre script `degrid` (défaut 1.0 si mode==1 sinon 0.0, `FFT3dGPU.cpp:1267`).
5. **Conditions** : `degrid != 0` uniquement.
6. **Ordre** : `selectDC` (Minimize) → `calcgridcorrection` → puis le résultat `FImgDegrid` est passé au Wienner et au Sharpen (`FFT3dGPU.cpp:855-859, 894-903, 945-954, 1000-1007`).
7. **Géométrie** : `NQuad(&target->GetRect(), 1, 0, false)` (`filtersps.cpp:56,65`) — même RECT pour la cible et la source (RectOffset=0).

---

## 3. psWiennerFilter (Wienner2d / Wienner3d2 / Wienner3d3 / Wienner3d4)

**Déclaration** `filtersps.h:43-55` — **Impl** `filtersps.cpp:86-233`.

### 3.1 Choix de la variante et defines
Constructeur (`filtersps.cpp:86-110`) :
```cpp
Pixelshader(pDevice, SRC_SHADER, n == 1 ? "Wienner2d" : n == 2 ? "Wienner3d2" : n == 3 ? "Wienner3d3" : "Wienner3d4",
  D3DXGetPixelShaderProfile(pDevice),
  SetMacroArray(degrid ? "DEGRID" : " ",
    VecToMacroArray(sigma,  pattern ? "" : "SIGMA",
      VecToMacroArray(beta, "BETA", CreateMacroArray(3)), 1), 2))
```
Defines : `BETA=float2(beta2)`, `SIGMA=float2(sigma2)` (absent si `pattern`), `DEGRID` (si degrid). Le bloc `#ifdef BETA` englobe tout le code HLSL Wienner (`hlsl:634-778`).

### 3.2 Constantes CPU (formules exactes)
`FFT3dGPU.cpp:172` : `sigma = _sigma / 255` (le paramètre script est ramené en 0..1).
`FFT3dGPU.cpp:396-398` :
```cpp
float s = sigma * sigma*bt*bw*bh;
sigma2 = D3DXVECTOR2(beta*s, s);                 // SIGMA.x = beta*s ; SIGMA.y = s
beta2  = D3DXVECTOR2((beta - 1.0) / beta, beta); // BETA.x = (beta-1)/beta ; BETA.y = beta
```
Création si `bm > 0` (bt∈{1..4}) : `wiennerfilter = NEW psWiennerFilter(pDevice, pFreeFImgdPool->top(), beta2, sigma2, bt, degrid != 0, usePattern)` (`FFT3dGPU.cpp:406-407`).
`usePattern = _sigma != _sigma2 || _sigma3 != _sigma4 || _sigma != _sigma3` (`FFT3dGPU.cpp:175`).

### 3.3 Texture « sigma » (motif de bruit) quand usePattern
`SigmaToPatternTexture` (`FFT3dGPU.cpp:588-631`), float CPU :
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
Puis tuilé `nx×ny` et uploadé dans `Pattern` = `TextureM` **FLOAT1** (`gtype->FLOAT()`), taille `fbw*nx × bh*ny` (`FFT3dGPU.cpp:597`). Passé en sampler `sigma` (non registré).

### 3.4 Fonctions HLSL et formules
Helper commun `Wienner(src,texCoord)` (`hlsl:643-664`) :
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
**Wienner2d** (`hlsl:666-686`) : `src=tex2D(current)` ; si DEGRID `correction=tex2D(degrid)`, `src-=correction` ; `retval=Wienner(src)` ; si DEGRID `retval+=correction`.

**Wienner3d2** (`hlsl:690-710`) — Hadamard temporel 2 images :
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
**Wienner3d3** (`hlsl:712-748`) — DFT 3 points :
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
**Wienner3d4** (`hlsl:751-777`, « modified from Fizick fft3dfilter ») — DFT 4 points :
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
(Ne pas oublier : la transformée inverse n'a **pas de facteur 1/N** — la normalisation est absorbée ailleurs.)

### 3.5 Textures, samplers, mapping
- Handles CPU (`filtersps.cpp:91-108`, attention au *fall-through* du switch) :
  - n=4 : `sSrc[3]="prev2"`, `sSrc[2]="prev1"`, `sSrc[1]="current"`, `sSrc[0]="next"` ;
  - n=3 : `sSrc[2]="prev1"`, `sSrc[1]="current"`, `sSrc[0]="next"` ;
  - n=2 : `sSrc[1]="prev1"`, `sSrc[0]="current"` ; n=1 : `sSrc[0]="current"`.
  - `sFactor="degrid"` si degrid ; `sSigma="sigma"` si pattern.
- Toutes les textures d'entrée/sortie sont des `TextureRT` du pool **FLOAT4/HALF4 `totw/2×toth`** ; `degrid` idem ; `sigma` = `Pattern` FLOAT1.
- Mapping vector→samplers (`filtersps.cpp:144-148, 170-174, 187-191`) : `for (i=0, j=size-1; j>=0; i++,j--) src[i] → sSrc[j]` — le vecteur est ordonné **ancien→récent** (`FImg2d[i]` = frame `n-current+i`, `FFT3dGPU.cpp:829-853`), donc `src[0]` (le plus ancien) va vers `sSrc[size-1]` = `prev2`/`prev1`, et le plus récent vers `next`. `current = bm==1?0 : bm==4?2 : 1` (`FFT3dGPU.cpp:176`).
- 4 surcharges `Apply` :
  - `Apply(Texture* src, TextureRT* dst, degrid, sigma)` — bt=1, mode 0/2 (`filtersps.cpp:114-134`) ;
  - `Apply(vector<TextureRT*>, TextureRT* dst, ...)` — bt≥2, mode 0/2 (`136-159`) ;
  - `Apply(vector<pTextureRTpair*>, pTextureRTpair* dst, ...)` — bt≥2, mode 1 : **deux draws** (sur `dst->first` puis `dst->last`) (`162-200`) ;
  - `Apply(pTextureRTpair*, pTextureRTpair*, ...)` — bt=1, mode 1 : deux draws (`202-233`).
- Géométrie : `NQuad(&_Rendertarget->GetRect(), 1)` (`filtersps.cpp:109`) — RectOffset=0, source=cible.

### 3.6 Place dans le pipeline (GetFrame)
- bt≥2 : FFT des bt frames (cache `GPUCache`) → degrid éventuel → `wiennerfilter->Apply(FImg2d…, FImg1d[current], FImgDegrid, Pattern)` → sharpen éventuel → iFFT (`FFT3dGPU.cpp:823-914`).
- bt=1 (bm==1) : FFT → degrid → `wiennerfilter->Apply(ping, pong, FImgDegrid, Pattern)` → swap → sharpen → iFFT (`FFT3dGPU.cpp:952-971`, mode 1 : `1010-1031`).

---

## 4. psKalman (abstrait), psKalmanSP, psKalmanMP

**Déclarations** `filtersps.h:58-89` — **Impl** `filtersps.cpp:236-332`.

### 4.1 psKalmanSP (single-pass, MRT 3 sorties)
Constructeur (`filtersps.cpp:236-247`) : fonction HLSL **`Kalman`**, defines :
```cpp
FloatToMacroArray(covarNoise, pattern ? " " : "COVARNOISE",
  FloatToMacroArray(kratiosquared, "KRATIOSQUARED", CreateMacroArray(2)), 1)
```
→ `KRATIOSQUARED=float` toujours ; `COVARNOISE=float` seulement **sans** pattern (sinon c'est un sampler).

HLSL (`hlsl:780-833`), garde `#ifdef KRATIOSQUARED` ; samplers `Last:s1, Covar:s2, CovarProcess:s3, CovarNoise:s4` ; si `COVARNOISE` non défini : `#define COVARNOISE tex2D(CovarNoise,In.texCoord).x` (`hlsl:794-796`). Sortie structure `PS_RETVAL {COLOR0 dst, COLOR1 covar, COLOR2 covarprocess}` (`hlsl:787-792`). Formule exacte :
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
`Apply` (`filtersps.cpp:249-277`) : 3 render targets simultanés — `dst` (RT0), `covar` (RT1), `covarprocess` (RT2) ; textures `Src`, `Last`, `CovarProcess`(←`covarprocesslast`), `Covar`(←`covarlast`) ; si pattern, `CovarNoise`←texture `pattern`. Après le draw : `SetRenderTarget(1,0); SetRenderTarget(2,0)` (`filtersps.cpp:274-275`).

### 4.2 psKalmanMP (multi-pass, cartes graphiques à 1 seul RT — old Riva128)
Constructeur (`filtersps.cpp:279-294`) : 3 instances de la fonction **`KalmanMP`** :
- `pass1` : define `PASS1_K` + `COVARNOISE`(si !pattern) + `KRATIOSQUARED` ;
- `pass2` : define `PASS2_K` + idem ;
- `pass3` : pas de define PASS (juste `COVARNOISE`/`KRATIOSQUARED`).

HLSL (`hlsl:839-877`) — commentaire `//PASS1=dst //PASS2=covar //PASS3=covarprocess` :
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
`Apply` (`filtersps.cpp:296-332`) : draw1 → RT=`dst` (résultat), draw2 → RT=`covar` avec `pass2`, draw3 → RT=`covarprocess` avec `pass3`. Les textures liées ne changent pas entre les 3 draws.

### 4.3 Constantes CPU
- `ps = NEW psKalmanSP(pDevice, StreamPoolPointer->top(), sigmaSquaredNoiseNormed2D, kratio*kratio, false)` si `Cap.NumSimultaneousRTs > 1`, sinon `psKalmanMP` (`filters.cpp:734-746`).
- Sans pattern : `sigmaSquaredNoiseNormed2D = sigma*sigma*bw*bh` avec `sigma=_sigma/255` (`FFT3dGPU.cpp:446-450`) — **attention : pas de facteur bt** (contrairement au Wienner).
- Avec pattern : covarNoise=0 en dur, le `COVARNOISE` vient de la texture `Pattern` (`FFT3dGPU.cpp:433-441`, ctor `filters.cpp:749-762`).
- `kratio` défaut script = 2.0 (`FFT3dGPU.cpp:1290`).

### 4.4 KalmanFilter (wrapper CPU)
**Déclaration** `filters.h:85-110` — **Impl** `filters.cpp:734-911`.
- `Init(useTexturepair)` (`filters.cpp:765-813`) : alloue depuis le pool `covar1/covar2` (+`covarprocess1/2`) ou les paires `covar1d…` ; remplit `covar1` et `covarprocess1` avec `f[i] = _sigmaSquaredNoiseNormed2D` (float), ou avec le motif : chaque texel de `_pattern` (FLOAT1) est répliqué dans les 4 composantes : `f[j]=f[j+1]=f[j+2]=f[j+3]=f1[i+x]` (`filters.cpp:774-776`).
- `Filter(Texture* src, TextureRT* dst)` (`filters.cpp:828-837`) : premier appel `last=src` ; `ps->Apply(src, last, covarprocess1, covar1, covarprocess2, covar2, dst, _pattern)` ; puis `swap(covar1,covar2)`, `swap(covarprocess1,covarprocess2)`, `last=dst` — **état temporel persistant ping-pong**.
- Version paire (`filters.cpp:839-856`) : 2 appels `ps->Apply` (first puis last).
- `Restore()` (`filters.cpp:858-911`) : réinitialise covar/covarprocess (device perdu) — mêmes formules qu'Init.
- Conditions : `bm == 0` (bt=0) uniquement (`FFT3dGPU.cpp:431-453`) ; appelé `kalman->Filter(ping, pong)` (`FFT3dGPU.cpp:959, 1017`).

---

## 5. psSharpen + Sharpen

**psSharpen** : `filtersps.h:389-397`, `filtersps.cpp:742-770`. **Sharpen** : `filters.h:112-125`, `filters.cpp:935-1001`.

### 5.1 Shader et defines
Constructeur (`filtersps.cpp:742-751`) : fonction HLSL **`Sharpen`**, defines :
```cpp
SetMacroArray(degrid ? "DEGRID" : " ",
  FloatToMacroArray(sigmaSquaredSharpenMax, "SSMAX",
    FloatToMacroArray(sigmaSquaredSharpenMin, "SSMIN", CreateMacroArray(3)), 1), 2)
```
HLSL (`hlsl:587-603`, garde `#ifdef SSMAX`) :
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
Samplers : `Src` (s0), `Factor` (s1), `FactorM` (s5, si DEGRID).

### 5.2 Constantes CPU
`FFT3dGPU.cpp:187-188` : `smin /= 255 ; smax /= 255` (défauts script : smin=4.0, smax=20.0, scutoff=0.3, svr=1.0 — `FFT3dGPU.cpp:1286-1289`).
`FFT3dGPU.cpp:413` :
```cpp
sharp = NEW Sharpen(sharpen, svr, scutoff, smin*smin*bw*bh*bt, smax*smax*bw*bh*bt,
                    bw, nx, bh, ny, _degrid != 0, pFreeFImgdPool, hr, pDevice, gtype);
```
→ `SSMIN = smin²·bw·bh·bt`, `SSMAX = smax²·bw·bh·bt` (le ctor psSharpen reçoit (max, min) dans cet ordre, `filtersps.cpp:742`).

### 5.3 LUT de facteur (calcul CPU, float)
`Sharpen::Sharpen` (`filters.cpp:935-976`), commentaire « from Fizick's code fft3dfilter 1.8 » :
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
Puis tuilage `FactorMap[offset++] = Factor[i + j*x]` sur `_ynum×_xnum` (`filters.cpp:963-968`) et upload dans `FilterLUT` = `TextureM` **FLOAT1** (`gtype->FLOAT()`), taille `width × height` (`filters.cpp:970-972`).
Note : `strength` = paramètre script `sharpen` (défaut 0.0 ; négatif = flou, cf. doc `filters.cpp:920`).

### 5.4 Textures / appel
- Entrées : `Src` = texture FFT filtrée (FLOAT4/HALF4 `totw/2×toth`), `Factor` = FilterLUT FLOAT1, `FactorM` = `FImgDegrid` (si degrid). Sortie : `TextureRT` du pool.
- `Apply` (`filtersps.cpp:753-770`) : 1 draw, `NQuad(&_Rendertarget->GetRect(), 1)` (`filtersps.cpp:750`, la « rendertarget » du ctor est en fait FilterLUT — même RECT que les textures FFT).
- `Sharpen::Filter` simple (`filters.cpp:984-987`) ; version paire : 2 draws, degrid first/last (`filters.cpp:989-1001`).
- Conditions : `sharpen != 0` (`FFT3dGPU.cpp:173, 412`). Ordre : **après** le filtre Wienner/Kalman, **avant** l'iFFT ; avec degrid, le motif est retiré puis remis dans le shader (`FFT3dGPU.cpp:860-865, 905-913, 964-969, 1023-1028`).

---

## 6. ImgStream (mode 0 et 2 ; mode 1 legacy) + classes psImg2toFloat4*/psFloat4toImg2*

**ImgStream** : `filters.h:28-61`, `filters.cpp:56-341`. Créée quand `mode != 1` (`FFT3dGPU.cpp:466`). Doc constructeur : `filters.cpp:29-53` — mode 0 = recouvrement 1:1 fenêtre cosinus modifiée ; mode 1 = recouvrement 2:1 demi-cosinus ; mode 2 = recouvrement 1:1 + bordure ajoutée.

### 6.1 Textures allouées (`filters.cpp:56-134`)
- `Img` `TextureM` : `((width+1)>>1) × height`, FIXED2 (8 bit) / FIXED2_16 (≤16 bit) / FLOAT2 (float) (`filters.cpp:65`). **2 pixels source par texel** (packing paires de colonnes).
- `Imgd` `TextureRT` : même taille/format (`filters.cpp:71`) — tampon intermédiaire float.
- `Imgr` `TextureRT` : `((width+3)>>2) × height`, FIXED4/FIXED4_16/FLOAT4 (`filters.cpp:74`) — sert au shader de conversion finale `conv` (4 pixels/texel). Détruit à la fin du ctor (`filters.cpp:132`) mais `conv` garde le RECT (`filtersps.cpp:345-346`).
- `FactorLUT` `TextureM` : mode≠2 : `xn2 × yn` (`xn=x*bw…` en fait `xn=x*xnum`, `x=bw`) en **HALF2** si useHalf sinon **FLOAT2** (`filters.cpp:95-98`) ; mode 2 : `bw/2 × bh` (`filters.cpp:118-121`).
- Offset CPU : `D3DXVECTOR2 offset = D3DXVECTOR2(0.25*(bw), 0.5*bh)` (`filters.cpp:80`) ; RECT src décalé de `-0.25*bw, -0.5*bh` (`filters.cpp:86-90`) ; rendertarget `(0,0,xn/2,yn)` (`filters.cpp:81-85`).

### 6.2 psImg2toFloat4m0 (mode 0)
`filtersps.h:103-113`, `filtersps.cpp:368-417`. Fonction HLSL **`Img2toFloat4`** (`hlsl:113-135`) ; defines : `BIT_DEPTH=int` + macro type de pixel (`GETPIXEL2TYPEMACRO`) si texture 2-composantes (`filtersps.cpp:369-372`).
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
Géométrie : `RECT rect[3]={Rendertarget, Src, Src+offset}` puis `NQuad(rect, 3)` (`filtersps.cpp:376-382`) — stage0=Factor (RECT cible), stage1=Src, stage2=Src décalé de `offset` (le pixel décalé de bw/4 en x et bh/2 en y). Sortie : `img` FLOAT4/HALF4 `totw/2×toth`. 1 draw.

### 6.3 psFloat4toImg2m0 (mode 0, inverse)
`filtersps.h:134-145`, `filtersps.cpp:473-539`. Fonction **`Float4toImg2`** (`hlsl:137-161`), **aucun define** (branche `#else`, pas MODE2) :
```hlsl
dst=tex2D(Src,In.texCoord[1]).xz*tex2D(Factor,In.texCoord[1]).xy
   +tex2D(Src,In.texCoord[0]).yw*tex2D(Factor,In.texCoord[0]).xy;
return dst.xyyy;
```
Géométrie : `rect[3]={Imgd->GetRect(), FactorMap->GetRect(), FactorMap->GetRect()+offset}`, `NQuad(rect, 2, 1)` (`filtersps.cpp:478-483`). Échantillonne la même texture Src à 2 adresses (décalage = la fenêtre glissante) et somme les 2 contributions de fenêtre. Sortie : `Imgd` (FIXED2/FIXED2_16/FLOAT2). `pback` = backbuffer restauré après le draw (`filtersps.cpp:484, 531`).

### 6.4 psImg2toFloat4m2 / psFloat4toImg2m2 (mode 2, avec bordure)
`filtersps.cpp:419-471` et `541-610`. Mêmes fonctions HLSL, mais :
- `psFloat4toImg2m2` compile avec define **`MODE2`** (`filtersps.cpp:542-543`) et constante sampler **`FactorM`** (`filtersps.cpp:545`). Branche MODE2 du HLSL (`hlsl:146-156`) :
```hlsl
float2 src_shifted=tex2D(Src,In.texCoord[1]).xz;
float2 factor_shifted=tex2D(FactorM,In.texCoord[3]).xy;
float2 src=tex2D(Src,In.texCoord[0]).yw;
float2 factor=tex2D(FactorM,In.texCoord[2]).xy;
dst=src_shifted*factor_shifted+src*factor;
dst=src_shifted+src;   // ⚠ écrase la ligne précédente : en MODE2 la fenêtre n'est PAS appliquée à la synthèse
```
- Géométrie : `MQuad::CreateVB/CreateVBi(srcW, srcH, bw/2, bh, xnum, ynum, border)` (`filtersps.cpp:427-429, 549-551`) — vertex buffer par bloc avec bordure (mode 2).
- Fenêtre CPU mode 2 : `CreateFactorMapBorder` (voir 6.7).

### 6.5 psImg2toFloat4_2 / psFloat4toImg2_2 (mode 1 de ImgStream, legacy)
`filtersps.cpp:613-662` et `665-738`. Fonctions **`Img2toFloat4_2`** (`hlsl:165-183`, identique à Img2toFloat4 mais texCoord[1]/[2] échangés) et **`Float4toImg2_2`** (`hlsl:185-198`) :
```hlsl
dst=tex2D(Src,In.texCoord[1]).xz*tex2D(Factor,In.texCoord[1]).xy+tex2D(Src,In.texCoord[0]).yw*tex2D(Factor,In.texCoord[0]).xy
+tex2D(I,In.texCoord[2]).xz*tex2D(Factor,In.texCoord[2]).xy+tex2D(I,In.texCoord[3]).yw*tex2D(I,In.texCoord[3]).xy;
```
(4 termes : les 2 moitiés du recouvrement 2:1, texture `Src`=première paire, `I`=seconde paire). `psFloat4toImg2_2::Apply(src1, src2, FactorLUT, dst)` lie `Src=s1, I=s2` (`filtersps.cpp:713-738`). Utilisé par `ImgStream::ImgToStream(pTextureRTpair*)` / `StreamToImg(pair)` (`filters.cpp:286-301, 335-341`) — ce chemin n'est plus pris par le filtre (mode 1 → `ImgStream2`), mais `ImgStream` l'instancie quand `mode==1` (`filters.cpp:111-114`).

### 6.6 psImg2toImg4 (conversion finale 2→4 composantes)
`filtersps.h:188-194`, `filtersps.cpp:337-366`. Fonction **`Img2toImg4`** (`hlsl:980-1003`), defines : `OFFSET=float2(1.0/src->GetWidth(), 0)` (`filtersps.cpp:339`), `BIT_DEPTH`, macro type :
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
Géométrie : `RECT rect[2]={Dst->GetRect(), src->GetRect()}`, `NQuad(rect, 2, 1, true)` (`filtersps.cpp:345-346`). Appelé par `ImgStream::StreamToImg` : `Float4toImg2->Apply(src,FactorLUT,Imgd); conv->Apply(Imgd,dst)` (`filters.cpp:322-332`) — `dst` = `DownloadImg[plane]` FIXED4/FIXED4_16/FLOAT4 `((width+3)>>2)×height`.

### 6.7 Fenêtres CPU ImgStream (double précision)
`const double pi(acos(-1.0));` (`filters.cpp:27`).
**CreateFactorMap** (`filters.cpp:157-185`) — fenêtre 2D `x×y` (bw×bh), tuilée `xnum×ynum` :
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
**CreateFactorMapBorder** (`filters.cpp:205-234`) — mode 2, `borderx=bordery=border*2` :
```cpp
cosy = cos(n1*pi / (y1 - 2*bordery));
int signy = cosy / abs(cosy);
cosx = cos(n2*pi / (x1 - 2*borderx));
int signx = cosx / abs(cosx);
double res = (0.5*(signx*cosx*cosx + signy * cosy*cosy));
Map[offset++] = res < 0 ? 0 : res;
```
(⚠ `abs()` sur un double = troncature entière selon les inclusions — à reproduire/en corriger consciemment au port.)

### 6.8 Résumé du flux ImgStream
- `ImgToStream(Texture* Img, TextureRT* dst)` : `Img2toFloat4->Apply(Img, FactorLUT, dst)` (`filters.cpp:265-283`).
- `StreamToImg(Texture* src, TextureRT* dst)` : `Float4toImg2->Apply(src, FactorLUT, Imgd)` puis `conv->Apply(Imgd, dst)` (`filters.cpp:322-332`).
- Upload CPU→GPU de la frame : `UploadToTexture(upload, src->GetReadPtr(plane), pitch/pixelsize)` (`FFT3dGPU.cpp:839, 931`) dans `UploadImg[plane]` (FIXED2/FIXED2_16/FLOAT2, `((width+1)>>1)×height`).

---

## 7. ImgStream2 (mode 1 — chemin principal moderne) + familles psImg2ToFloat4* / psFloat4ToImg2*

**ImgStream2** : `filters.h:63-83`, `filters.cpp:366-707`. Créée quand `mode == 1` (`FFT3dGPU.cpp:461-464`) :
```cpp
convert2 = NEW ImgStream2(bw, bh, ow, oh, isLumaPlane?UploadImg[0]:UploadImg[1],
    imgp->first, isLumaPlane?DownloadImg[0]:DownloadImg[1],
    useChromaDisplacementMacro /* = !isLumaPlane && !isRGB */, wintype, interlaced, pDevice, gtype, useHalf, bits_per_pixel, hr);
```
Note importante : les classes shader reçoivent **`bw/2` comme largeur de bloc** (le packing 2 colonnes/texel est déjà appliqué) et `ow/2` (`filters.cpp:425-444`).

### 7.1 Textures et choix de classe (`filters.cpp:366-478`)
- `Imgd` `TextureRT` `width×height` FIXED2/FIXED2_16/FLOAT2 (`filters.cpp:371`) ; `conv = psImg2toImg4(Imgd → dst)` (`filters.cpp:372`).
- `FactorLUTana` / `FactorLUTsyn` `TextureM` **FLOAT4** (`gtype->FLOAT4()`) :
  - si `Cap.NumSimultaneousRTs > 1 || ow*2 != bw || oh*2 != bh` : taille **`bw/2 × bh`** (petit LUT, addressing WRAP) (`filters.cpp:376-380`) ;
  - sinon (1 RT + recouvrement demi) : taille **`fdst->GetWidth() × fdst->GetHeight()`** (LUT pleine taille) (`filters.cpp:381-385`).
- Sélection de `i2f` (psImg2ToFloat4) et `f2i` (psFloat4ToImg2) (`filters.cpp:386-447`) :

| Cas | MRT>1 | MRT==1 |
|---|---|---|
| recouvr. demi (`ow*2==bw && oh*2==bh`), progressif | `psImg2ToFloat4ohalfSP` + `psFloat4ToImg2ohalf(fullsizefactor=false)` | `psImg2ToFloat4ohalfMP` + `psFloat4ToImg2ohalf(fullsizefactor=true)` |
| demi, entrelacé | `psImg2ToFloat4ohalfInterlacedSP` + `psFloat4ToImg2ohalfInterlaced` | `psImg2ToFloat4ohalfInterlacedMP` + `psFloat4ToImg2ohalfInterlaced` |
| recouvr. quelconque, progressif | `psImg2ToFloat4oSP` + `psFloat4ToImg2o` | `psImg2ToFloat4oMP` + `psFloat4ToImg2o` |
| quelconque, entrelacé | `psImg2ToFloat4oInterlacedSP` + `psFloat4ToImg2oInterlaced` | `psImg2ToFloat4oInterlacedMP` + `psFloat4ToImg2oInterlaced` |

- `ImgToTexture(src, dstpair)` : `i2f->Apply(src, FactorLUTana, dst->first, dst->last)` — **une passe, 2 RT** (ou 2 passes MP) (`filters.cpp:685-690`).
- `TextureToImg(srcpair, dst)` : `f2i->Apply(src->first, src->last, FactorLUTsyn, Imgd); conv->Apply(Imgd, dst)` (`filters.cpp:692-697`).
- `TextureToSrc` : idem mais `DownloadFromTexture(Imgd, dst, pitch)` (`filters.cpp:699-707`).

### 7.2 Fenêtres CPU ImgStream2 — `CreateFactorMap` (`filters.cpp:501-670`)
Doc : « based on the window function code from fft3dfilter 1.8.5 ». Fenêtres 1D sur la zone de recouvrement `ow`/`oh` (double) :
- **wintype 0** (défaut) : analyse = synthèse = demi-cosinus :
```cpp
wanx[i] = cosf(pi*(i - ow + 0.5f) / (ow * 2));   wsynx[i] = wanx[i];
wany[i] = cosf(pi*(i - oh + 0.5f) / (oh * 2));   wsyny[i] = wany[i];
```
- **wintype 1** : analyse = √(demi-cos), synthèse = (demi-cos)^1.5 :
```cpp
wanx[i] = sqrt(cosf(pi*(i - ow + 0.5f) / (ow * 2)));
wsynx[i] = wanx[i] * wanx[i] * wanx[i];   // = cos^1.5
```
- **wintype 2** : analyse = 1, synthèse = cos² (raised cosine) :
```cpp
wanx[i] = 1.0;
wsynx[i] = cosf(pi*(i - ow + 0.5f) / (ow * 2)); wsynx[i] *= wsynx[i];
```
Assemblage 2D (`filters.cpp:555-664`) : la fenêtre est **séparable et symétrique** ; chaque pixel de la LUT est un `float4` portant **2 échantillons horizontaux consécutifs** (x et x+1) pour la partie gauche, et les miroirs `wanx[(ow-1)-x]`, `wanx[ow-x-2]` pour la partie droite ; 3 bandes verticales : `oh` lignes en haut (produit `wanx*wany`), `bh-2*oh` lignes au milieu (x seulement, ou 1), `oh` lignes en bas en ordre inverse (`y = oh-1 → 0`). Le centre vaut `1` (analyse et synthèse) (`filters.cpp:606-616`).
Si `MRT==1 && recouvr. demi` : la LUT `bw*2×bh` est tuilée en `xnum×ynum` pour couvrir toute l'image (`filters.cpp:452-471`). Upload : `UploadToTexture(FactorLUTana, facana)` / `(FactorLUTsyn, facsyn)` (`filters.cpp:473-474`).

### 7.3 HLSL commun : Img2ToFloat4 / Img2ToFloat4MP (`hlsl:936-977`, garde `#ifdef OFFSET`)
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
Macros de bit depth via `DEFINE_A8L8_AND_CHROMA_MACROARRAY_MAX3/4/5` (`filtersps.cpp:874-876`) : posent `A8L8|G16R16|G32R32F`, `BIT_DEPTH`, et `CHROMA_BIT_DEPTH` si `chroma`.

### 7.4 psImg2ToFloat4oSP (recouvrement quelconque, MRT, 1 passe)
`filtersps.cpp:878-915`. Defines : `OFFSET=float2(0, (bh-oh)/(float)src->GetHeight())` (`filtersps.cpp:880`), `BIT_DEPTH`, chroma. Vertex buffer dédié `OQuad::CreateVB(srcW, srcH, bw, bh, ow, oh)` (`filtersps.cpp:886-888`) — un quad par bloc avec coordonnées xy/zw distinctes. `Apply` : MRT `dst1=RT0` (première image du recouvrement), `dst2=RT1` (`filtersps.cpp:896-915`). Sorties : `imgp->first/last` FLOAT4/HALF4.

### 7.5 psImg2ToFloat4oMP (1 RT, 2 passes)
`filtersps.cpp:918-965`. Pass1 : `OFFSET=float2(0, (bh/2)/srcH)` + define `PASS1` (`filtersps.cpp:919-924`) → écrit `dst1` ; Pass2 : `OFFSET=float2(0, (bh-oh)/srcH)` sans PASS1 (`filtersps.cpp:925-929`) → écrit `dst2`. Même `OQuad`.

### 7.6 psImg2ToFloat4oInterlacedSP / MP (entrelacé)
`filtersps.cpp:968-1076` / `1078-1194`. `OFFSET=float2(0, 2*(bh-oh)/(float)src->GetHeight())` (`filtersps.cpp:970, 1080, 1086`). Vertex buffer `OIQuad::CreateVB` (paires de lignes). Après le(s) draw(s), **4×2 `StretchRect`** pour échanger les moitiés haut/bas des champs dans dst1 et dst2 (`filtersps.cpp:1057-1073`, rectangles `r[0..11]` calculés au ctor `filtersps.cpp:986-1004`) :
```cpp
int ny = ((src->GetHeight() + oh - 1) / (bh - oh) + 1 + 1) / 2;
bottomheight = ny / 2 * (bh - oh) - src->GetHeight() / 2;
```
(échange r0↔r2, r4↔r6 dans dst1 ; r8↔r10 dans dst2, via `temptex` copie de `dst`). MP : pass1→dst1, pass2→dst2 puis mêmes échanges (`filtersps.cpp:1154-1191`).

### 7.7 psImg2ToFloat4ohalfSP (recouvrement exact 50 %, MRT)
`filtersps.cpp:1196-1249`. Defines : `OFFSET=float2(0, (bh/2)/(float)src->GetHeight())` (`filtersps.cpp:1198`). Géométrie `NQuad(rec[4], 3, 1, false)` (`filtersps.cpp:1205-1225`) :
```
rec[0] = destination  dstwidth=((srcW + bw*1.5 - 1)/bw)*bw, dstheight=((srcH + bh*1.5 -1)/bh)*bh
rec[1] = src décalé xy : left=-bw/2, right=srcW-bw/2, top=-bh/2, bottom=srcH-bh/2
rec[2] = src décalé zw : top=-bh/2, bottom=srcH-bh/2 (x non décalé)
rec[3] = factor : (0,0,bw,bh)  → LUT petit format, d'où ADDRESSU=WRAP
```
`Apply` : sampler Factor `ADDRESSU=WRAP` pendant le draw, remis à `MIRROR` après (`filtersps.cpp:1239-1247`) ; MRT dst1/dst2.

### 7.8 psImg2ToFloat4ohalfInterlacedSP (entrelacé, MRT)
`filtersps.cpp:1251-1364`. `OFFSET=float2(0, bh/(float)srcH)` (`filtersps.cpp:1253`). Deux quads : `quad` (champ pair, `qrec={0,0,dstW,dstH/2}`) et `bottomquad` (champ impair, RECTs source **+1 pixel** en y, `qrec` décalé de dstH/2) (`filtersps.cpp:1262-1288`). `Apply` : `Clear(0x00880000)` (`filtersps.cpp:1328`), draw `bottomquad` puis `quad` (`1338-1342`), Factor WRAP→MIRROR, puis échanges StretchRect (`1345-1358`) avec :
```cpp
int bottomheight = (dstheight - bh - src->GetHeight()) / 2;   // (filtersps.cpp:1290)
```

### 7.9 psImg2ToFloat4ohalfMP / ohalfInterlacedMP (1 RT)
`filtersps.cpp:1367-1429` / `1432-1551`. 2 shaders `Img2ToFloat4MP` : pass1 avec define `PASS1` et `OFFSET=float2(0,(bh/2)/srcH)` (`1368-1373`), pass2 sans PASS1, même OFFSET (`1374-1378`) — ici OFFSET pass1==pass2 car le décalage est porté par les RECTs du quad. Draw pass1→dst1, pass2→dst2 (`1405-1429`). Version entrelacée : `OFFSET=float2(0, bh/srcH)` pour les 2 passes (`1433-1443`), Clear `0x00880000`, draws bottomquad puis quad pour chaque passe (`1504-1532`), échanges StretchRect (`1534-1547`).

### 7.10 psFloat4ToImg2o (synthèse recouvrement quelconque, progressif)
`filtersps.h:348-363`, `filtersps.cpp:1556-1636`. **6 shaders** sur le même couple de textures FFT (`Src`=pair, `Src2`=impair) + `Factor` (LUT synthèse FLOAT4) :
- `Float4ToImg2Corner` (sans define, `hlsl:880-895`) — coins de bloc, mélange croisé des 4 textures :
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
- `Float4ToImg2Nooverlap` (`hlsl:905-910`) avec define **`XY`** pour la variante xy (sinon zw) :
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
- `Float4ToImg2Horizontal` (`hlsl:912-922`) : `dst=src1*factortex0+src2*factortex1` (SW0/SWF) `+CHROMANORM` ; define `XY` pour xy.
- `Float4ToImg2Vertical` (`hlsl:924-933`) :
```hlsl
float4 factor=tex2D(Factor,In.texCoord[2]);
float4 src1=tex2D(Src,In.texCoord[0]).xzxz;
float4 src2=tex2D(Src,In.texCoord[1]).ywyw;
dst=src1*factor.xyxy+src2*factor.zwzw;
return dst+CHROMANORM;
```
Ordre exact des draws dans `Apply` (`filtersps.cpp:1579-1630`), après `Clear(0xFFFFFFFF)` (`1586`) :
1. `DrawCorners` (shader Corner) ; 2. `DrawHorizontalxy` (XY) ; 3. `DrawHorizontalzw` ; 4. `DrawNonoverlaptex1xy` (XY) ; 5. relier `Src2→sSrc1` puis `DrawNonoverlaptex2xy` ; 6. `DrawNonoverlaptex2zw` ; 7. relier `Src1→sSrc1` puis `DrawNonoverlaptex1zw` ; 8. `DrawVerticaltex1` ; 9. relier `Src2` puis `DrawVerticaltex2`.
Défines bit-depth : `CHROMA_BIT_DEPTH` si chroma (`filtersps.cpp:1557-1573`).

### 7.11 psFloat4ToImg2oInterlaced (synthèse entrelacée — astuce z-buffer)
`filtersps.h:366-386`, `filtersps.cpp:1638-1865`. Mêmes 6 shaders. Au constructeur (`1658-1696`) :
- `ztex` = `TextureRT` `width×height` **FIXED1** (`gtype->FIXED()`) remplie en CPU : ligne i = `(i & 1) * 255` (`1667-1671`) ;
- `zbuffer` = `CreateDepthStencilSurface(width, height, D3DFMT_D24X8, ...)` (`1665`) ;
- shader **`CopyTexToZBuffer`** (`hlsl:1012-1018`, `dst.d=tex2D(Src,In).x`) copie ztex dans le z-buffer avec `ZWRITEENABLE=TRUE, ZFUNC=ALWAYS` (`1676-1688`).
`Apply` (`1747-1855`) : `Clear(0x00880000)` ; `ZENABLE=TRUE, ZWRITE=FALSE, ZFUNC=LESS` → draws du quad **pair** (`oevenquad`) ; puis `ZFUNC=GREATER` → draws du quad **impair** (`ooddquad`) — le test de profondeur filtre les paires de lignes. Les 2 quads viennent de `OIQuad::CreateVBi(width,height,bw,bh,ow,oh,srcwidth,srcheight, true/false)` (`1692-1693`).

### 7.12 psFloat4ToImg2ohalf (synthèse recouvr. 50 %, 1 draw)
`filtersps.h:321-329`, `filtersps.cpp:1869-1920`. Shader **`Float4ToImg2Corner`** seul, `NQuad(rec[5], 5, 1, false)` (`1876-1897`) :
```
rec[0] = dst (width×height)
rec[1] = tex0xy : left=bw/2, right=srcwidth+bw/2, top=bh/2, bottom=srcheight+bh/2
rec[2] = tex1zw : (0,0,srcwidth,srcheight)
rec[3] = factortex0 : left=bw/2, top=bh/2, right=(fullsizefactor?srcwidth:bw)+bw/2, bottom=(fullsizefactor?srcheight:bh)+bh/2
rec[4] = factortex1 : left=bw/2, top=0, right=(fullsizefactor?srcwidth:bw)+bw/2, bottom=(fullsizefactor?srcheight:bh)
```
Factor sampler `ADDRESSU=WRAP`→`MIRROR` (`1911-1918`). Sortie : `Imgd` FIXED2/FIXED2_16/FLOAT2.

### 7.13 psFloat4ToImg2ohalfInterlaced (synthèse entrelacée 50 %)
`filtersps.h:332-345`, `filtersps.cpp:1923-2100`. Même principe z-buffer que 7.11 (ztex FIXED1 + zbuffer D24X8 + `CopyTexToZBuffer`, `1937-1963`). Deux quads 5-RECT : `quad` (première moitié, `qrec={0,0,width,height}` mais rec[0].bottom=height/2) et `evenfieldquad` (RECTs source `+srcheight/2` en y, `qrec.top++/bottom++` — commentaire `//inc to get correct eta`, `1990-2000`). `Apply` (`2063-2100`) : Clear `0x00880000`, Factor WRAP, `ZFUNC=GREATER` draw `quad`, puis `ZFUNC=LESS` draw `evenfieldquad`, remise à l'état neutre.

### 7.14 Résumé du flux ImgStream2 (mode 1)
`UploadToTexture(UploadImg[plane])` → `convert2->ImgToTexture(upload, imgp)` (1 draw MRT ou 2 passes) → FFT×2 (`imgp->first`, `imgp->last`) → filtre (Wienner/Kalman en paires) → sharpen → iFFT×2 → `convert2->TextureToImg(imgp, download)` (synthèse + `psImg2toImg4`) → `DownloadFromTexture(DownloadImg[plane])` (`FFT3dGPU.cpp:873-1042, 1140-1166`).

---

## 8. psFFTtoFixed (debug/visualisation — non utilisé en production)

`filtersps.h:400-409`, `filtersps.cpp:778-840`. Fonction **`FFTtoFixed`** (`hlsl:606-615`), define optionnel **`FFT`** (`filtersps.cpp:779`) :
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
Sortie vers le backbuffer ou une RT (`filtersps.cpp:818-840`). Seule instanciation : **commentée** (`FFT3dGPU.cpp:509`). `psMeanSD` est entièrement commenté (`filtersps.h:410-418`, `filtersps.cpp:842-872`) — la logique équivalente CPU est `FFT3dGPU::CalcMeanSD` (`FFT3dGPU.cpp:647-667`, non appelée).

---

## 9. Ordre complet des passes dans GetFrame (rappel pour le port)

Par plane (luma puis chroma U,V — deux instances FFT3dGPU si plane=4, `FFT3dGPU.cpp:1260-1341`) :
1. Reset render/sampler state (MIRROR sur stages 0..5) — `FFT3dGPU.cpp:813-820`.
2. Upload frame → `UploadImg[plane]` (FIXED2/16/FLOAT2) — `UploadToTexture` / `UploadInterleavedToFixedTexture` (YUY2) (`FFT3dGPU.cpp:837-842`).
3. `ImgToStream`/`ImgToTexture` (classes du §6/§7).
4. FFT 2D (`FFT2d*`, hors périmètre de ce document) sur `img` ou `imgp->first/last`.
5. bt≥2 : FFT des bt frames via `GPUCache` (`FFT3dGPU.cpp:829-853`).
6. degrid : `psMinimize` puis `psGridCorrection` → `FImgDegrid` (`855-858`).
7. Filtre : `psWiennerFilter` (bt≥1) ou `KalmanFilter` (bt=0) (`859, 954-960`).
8. `Sharpen` si `sharpen!=0` (`860-863`).
9. iFFT.
10. `StreamToImg`/`TextureToImg` : synthèse (`psFloat4ToImg2*`) + `psImg2toImg4`.
11. `DownloadFromTexture(DownloadImg[plane])` (`FFT3dGPU.cpp:1140-1166`).
12. Upload anticipé de la frame suivante (`UploadImg2` puis swap, `FFT3dGPU.cpp:1093-1136`).

## 10. Points d'attention pour le port OpenCL
- **Demi-texel** : tout le mapping est dans les RECTs NQuad (`eta = 0.5/W + left/W`, `vertexbuffer.cpp:90-91`) ; en OpenCL, travailler en indices entiers de texels plutôt qu'en coordonnées normalisées.
- **Précision** : les defines float sont formatés en `std::fixed` avec 6 décimales (`shader.cpp:139,197`) → la constante compilée peut différer du float CPU ; à reproduire si l'on veut la bitwise-exactitude, ou passer les constantes en arguments de kernel (préférable).
- **Écritures multiples** : KalmanSP (3 RT), Img2ToFloat4 (2 RT) → 3/2 kernels ou `write_image` multiples.
- **Échantillonnage** : partout NEAREST + MIRRORED_REPEAT (sauf LUT factor « ohalf » : REPEAT sur U ; état initial device : CLAMP border 0).
- **Formats** : HALF4↔`A16B16G16R16F` (fp16 RGBA), FIXED2↔`A8L8` (attention : L=canal R ? mappage `SW1=xw` pour A8L8 : L→x, A→w ; pour G16R16 : `SW1=rg` : R→x? — vérifier l'ordre des composantes dans le port), packing 2 px/texel (FIXED2) et 4 px/texel (FIXED4).
- **Normalisation 10-14 bit** : `BITDEPTHNORM = 1<<(16-BIT_DEPTH)` à l'entrée, `INV_BITDEPTHNORM` + clamp `MAX_PIXEL_VALUE` à la sortie (`hlsl:35-39`, `hlsl:995-1000`).
- **CHROMANORM** : décalage `32768/65535` ajouté/retiré pour la chroma YUY2 (`hlsl:21-33`, `hlsl:951-952`, `hlsl:894`).
- **Bug connu à reproduire ou corriger consciemment** : branche MODE2 de `Float4toImg2` où la fenêtre de synthèse est écrasée par `dst=src_shifted+src` (`hlsl:155-156`) ; `abs()` entier sur double dans `CreateFactorMapBorder` (`filters.cpp:220,227`).
