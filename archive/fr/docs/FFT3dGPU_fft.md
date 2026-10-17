# FFT3dGPU — Documentation technique exhaustive de l'implémentation FFT

> Document préparé pour le portage OpenCL du plugin AviSynth **FFT3dGPU** (DirectX 9).
> Racine des sources : `FFT3dGPU\FFT3dGPU`. Toutes les références sont `fichier:ligne`.
> Les formules HLSL/C sont recopiées **verbatim** (indentation d'origine conservée dans les blocs de code).
> Date d'analyse : 2026-10-17. Aucun fichier source n'a été modifié.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture : ce qui tourne sur le GPU, ce qui tourne sur le CPU](#2-architecture)
3. [Conventions communes (adressage, textures, macros de shader)](#3-conventions-communes)
4. [Encodage des données (packing RGBA, astuce FFT réelle)](#4-encodage-des-données)
5. [Algorithme 1 — RADIX2LUT (`FFT2dRR`, fichier `FFT.cpp`)](#5-algorithme-1--radix2lut-fft2drr)
6. [Algorithme 2 — STOCKHAM (`FFT2dRR2`, fichier `fft2.cpp`)](#6-algorithme-2--stockham-fft2drr2)
7. [FFT temporelle 1D (2 et 3 points)](#7-fft-temporelle-1d-2-et-3-points)
8. [Le paramètre `precision` (0/1/2) et les formats de texture](#8-le-paramètre-precision-012-et-les-formats-de-texture)
9. [Mode MEASURE (choix automatique de l'implémentation)](#9-mode-measure)
10. [Reproductibilité numérique — points critiques pour le portage](#10-reproductibilité-numérique)
11. [Annexe A — Récapitulatif des shaders HLSL](#annexe-a--récapitulatif-des-shaders-hlsl)
12. [Annexe B — Récapitulatif des géométries de sommets](#annexe-b--récapitulatif-des-géométries-de-sommets)
13. [Annexe C — Correspondance DX9 → OpenCL](#annexe-c--correspondance-dx9--opencl)

---

## 1. Vue d'ensemble

Le plugin implémente la FFT 2D (par blocs) et 3D (2D spatiale + 1D temporelle) **entièrement sur le GPU**, via des pixel shaders et vertex shaders DirectX 9 compilés à l'exécution depuis le fichier `fft3dgpu.hlsl` situé à côté du DLL (`core/shader.h:28` définit `SRC_SHADER` ; `core/shader.cpp:54-76` compile le source au runtime avec D3DXCompileShader).

Il existe **deux implémentations GPU complètes** de la FFT 2D, sélectionnées par l'enum `FFTCODE` :

```cpp
// FFT3dGPU.h:58-61
enum FFTCODE {
  MEASURE,     // = 0 : benchmark les deux, garde la plus rapide
  RADIX2LUT,   // = 1 : FFT radix-2 avec tables d'indices (LUT) en textures  (classe FFT2dRR,  FFT.cpp)
  STOCKHAM,    // = 2 : FFT de Stockham (ping-pong), twiddles encodés dans les sommets (classe FFT2dRR2, fft2.cpp)
};
```

La dispatch se fait dans `FFT3dGPU.cpp:363-387` : `RADIX2LUT` crée un objet `FFT2dRR`, `STOCKHAM` crée un `FFT2dRR2`, `MEASURE` crée les deux, les chronomètre (`MeasureFFT`, `FFT3dGPU.cpp:103-127`) et garde la plus rapide.

**Point essentiel : il n'y a aucun FFT CPU.** Les fichiers `FFT.cpp` et `fft2.cpp` ne contiennent que de l'orchestration côté hôte (création de textures, compilation de shaders, envoi des draw calls). Le « code CPU » sert uniquement à **construire les tables** (twiddles, bit-reversal, cartes de chargement/sauvegarde) en `double` côté CPU, puis à les téléverser dans des textures GPU en `float` (ou `half` selon `precision`).

Tailles de bloc supportées : `bw`/`bh` arrondis à la puissance de 2 la plus proche, bornés à **[4, 512]** (`FFT3dGPU.cpp:214-231`).

---

## 2. Architecture

### 2.1 Ce qui tourne sur le GPU

- Tous les papillons (butterflies) de la FFT, dans des **pixel shaders**.
- Les twiddles (facteurs de rotation) : soit lus dans des **textures LUT** (RADIX2LUT), soit **interpolés depuis les coordonnées de sommets** (STOCKHAM), soit calculés par `sincos` **matériel** dans le shader (STOCKHAM stage 0, C2Rfft/R2Cfft).
- Les conversions complexe↔réelle (C2R/R2C), la collecte finale, la normalisation.

### 2.2 Ce qui tourne sur le CPU

Uniquement la génération des tables, en `double`, téléversées ensuite en `float`/`half` :

| Table | Constructeur | Fichier:ligne | Contenu |
|---|---|---|---|
| `TwiddleLUTforward/backward[i]` | `CreateTwiddleMap` | `FFT.cpp:743-768` | (cos, sin) par papillon, **récurrence Numerical Recipes** |
| `LoadLUT2[i]` | `CreateBitReverseMap` + `CreateLoadMap` + `CreateSaveMap` + `CreateLoadMap2` | `FFT.cpp:714-723`, `633-646`, `566-582`, `605-615` | coordonnées texture de lecture + sélecteur de texture I/J |
| `SaveLUT[i]` | `CreateSaveMap` | `FFT.cpp:566-582` | coordonnées texture d'écriture + sélecteur I/J |
| `R2CLUT` / `C2RLUT` | `CreateC2RMap` | `FFT.cpp:788-820` | coordonnées miroir + (cos, sin) de la conversion réelle |
| Géométries STOCKHAM | constructeurs `FFTQuad0/2`, `C2RQuad`, `R2CFFTQuad` | `core/vertexbuffer.cpp:2871+`, `2723+`, `2249+`, `2598+` | twiddles/angles en données de sommet (`cos(-pi*m/n)` etc., calculés en `double` par le compilateur C, stockés `float32`) |

### 2.3 Fichiers

| Fichier | Rôle |
|---|---|
| `fft3dgpu.hlsl` | **Tous** les shaders (pixel + vertex), les deux algorithmes. Le fichier `hlsl/original/fft3dgpu.hlsl` est une copie plus ancienne, non utilisée. |
| `FFT.h` / `FFT.cpp` | Classe `FFT2dRR` = orchestration RADIX2LUT + tous les constructeurs de tables CPU. |
| `FFTPS.h` / `FFTPS.cpp` | Enveloppeurs C++ des shaders RADIX2LUT (`psBitReverseButterFly*`, `psButterflyCollect*`, `psCollect*`, `psC2R`, `psR2C`, `psFFT2p/3p`), variantes MRT et multipass. |
| `fft2.h` / `fft2.cpp` | Classe `FFT2dRR2` = orchestration STOCKHAM. |
| `fft2ps.h` / `fft2ps.cpp` | Enveloppeurs C++ des shaders STOCKHAM (`psButterflyHSP/HMP/VSP/VMP`, `psC2Rfft`, `psR2Cfft`, `psCombine`). |
| `core/vertexbuffer.cpp` | Géométries : `NQuad` (plein écran), `C2RQuad`, `CollectQuad`, `R2CFFTQuad`, `FFTQuad2`, `FFTQuad0`. |
| `core/texture.cpp/.h` | `GPUTYPES` (formats), `TextureM` (LUT), `TextureRT` (rendu), `UploadToTexture` (conversion half). |
| `core/shader.cpp/.h` | Compilation runtime du HLSL, helpers de macros (`FloatToMacroArray`, `VecToMacroArray`, `SetMacroArray`, `CreateMacroArray`, `core/shader.cpp:136-262`). |
| `FFT3dGPU.cpp` | Plugin AviSynth : dispatch `FFTCODE`, clamp bw/bh, override precision, état des sampler. |

---

## 3. Conventions communes

### 3.1 Adressage « centre de texel »

`NQuad` (`core/vertexbuffer.cpp:40-169`) génère un quad plein écran dont les coordonnées de texture sont **pré-transformées** pour tomber exactement au centre des texels : `texCoord = (i + 0.5) / W` (`core/vertexbuffer.cpp:87-100`). Toutes les LUT utilisent le même décalage `+0.5` (voir `CreateBitReverseMap` `FFT.cpp:720`, `CreateSaveMap` `FFT.cpp:573/575`, `CreateC2RMap` `FFT.cpp:795-796`, etc.). Le filtre du sampler est **POINT** (défaut) ; l'adressage est mis à `MIRROR` pendant l'exécution (`FFT3dGPU.cpp:816-820`, créé en `BORDER` dans `core/ManageDirect3DWindow.cpp:221-225`).

> ⚠ Pour le port OpenCL : remplacer tout cela par des index entiers. Le `+0.5` n'existe que pour contourner l'interpolation des textures DX9.

### 3.2 MRT vs multipass

Si la carte ne supporte qu'un seul render target simultané (`Cap.NumSimultaneousRTs == 1`), le plugin compile des variantes `_MP` des shaders avec les macros `MULTIPASS`/`PASS0`/`PASS1`/`PASS2` et fait **deux draw calls** au lieu d'un (un par sortie). Choix fait dans `FFT.cpp:225-246` (RADIX2LUT) et `fft2.cpp:46-58` (STOCKHAM). Les shaders HLSL contiennent les deux branches (`#ifndef MULTIPASS` / `#ifdef PASS1`…).

> Pour OpenCL : sans objet — un kernel écrit partout. Mais il faut **préserver la sémantique** : chaque sortie est calculée par la **même expression**, donc le résultat est identique quel que soit le chemin.

### 3.3 Macros de compilation des shaders

Les shaders sont compilés avec des macros constantes injectées par le C++ (`core/shader.cpp:136-262`) :

- `PIXELOFFSET` : soit un offset d'une ligne de LUT (`stagex[0] = 0.5/log2x`), soit un décalage d'un texel (`1.0/x2n`) — voir §5.
- `NORM` : facteur de normalisation compilé dans le shader (`BitReverseButterFlyBH` RADIX2LUT, `Combine` STOCKHAM). Défaut `1.0` si absent (`fft3dgpu.hlsl:1148-1150`).
- `LOADOFFSET` : décalage de texture (macro `float2`) pour la lecture du second membre du papillon STOCKHAM (`fft2ps.cpp:9,13,55,59-61,116,120,161,165-167`).
- `STAGE0` : force le calcul `sincos` dans le pixel shader (stage 0 STOCKHAM).
- `MULTIPASS`/`PASS0/1/2` : variantes une sortie.

### 3.4 Structures HLSL de base

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

## 4. Encodage des données

### 4.1 Un texel RGBA = deux FFT complexes en parallèle

Commentaire verbatim du shader :

```hlsl
// fft3dgpu.hlsl:1103
//Does 2 fft in parallel. Data is stored in a float4 texture with Realfft1,Realfft2,Imagfft1,Imagfft2 in xyzw
```

Un texel `float4 = (Re1, Re2, Im1, Im2)` porte **deux nombres complexes** : le complexe n°1 = (x, z) = (Re1, Im1) et le complexe n°2 = (y, w) = (Re2, Im2). Les deux composantes (x,z) et (y,w) sont transformées indépendamment, ce qui fait deux FFT simultanées.

### 4.2 Astuce FFT réelle : deux signaux réels dans une FFT complexe demi-taille

Documenté verbatim dans la doc du constructeur :

```cpp
// FFT.cpp:396-397
 *     the real arrayes a and b must be packed like this in one float 4  a[0] b[0] a[1] b[1]
 *     and the complex result like this a_real b_real a_imag b_imag
```

Le signal réel `a` de largeur `_x` est stocké comme suite complexe de largeur `x = _x/2` : `c[k] = a[2k] + i·a[2k+1]` (échantillons pairs en Re, impairs en Im). Le signal `b` occupe la seconde composante complexe du même texel. On fait donc une FFT complexe de longueur `_x/2`, puis le shader **R2C** (forward) sépare et étend le spectre pour obtenir la FFT réelle complète de longueur `_x`, en ne gardant que la moitié utile : `xV = _x/2 + 1` colonnes (`FFT.cpp:59`).

En inverse, le shader **C2R** referme le spectre réel en suite complexe demi-taille avant la FFT inverse.

> ⚠ Nommage trompeur : dans le sens **forward**, c'est le shader `R2C` (`fft3dgpu.hlsl:314`) qui convertit le résultat de la FFT complexe demi-taille en spectre réel pleine taille ; dans le sens **inverse**, c'est le shader `C2R` (`fft3dgpu.hlsl:294`) qui fait l'opération inverse (`FFT.cpp:465` et `FFT.cpp:490-492`).

### 4.3 Dimensions

Avec `x = _x/2` (largeur complexe), `xnum` blocs en largeur, `y` hauteur de bloc, `ynum` blocs en hauteur, `bt` largeur temporelle :

- Texture d'entrée image : `xn × yn` = `(x·xnum) × y` (float4, 2 signaux).
- Spectre horizontal intermédiaire (complexe pleine taille) : `xn × yn`.
- Spectre final (moitié gardée) : `xVn × yn` = `xnum·(x+1) × y` (`FFT.cpp:140-141`).

---

## 5. Algorithme 1 — RADIX2LUT (`FFT2dRR`)

Classe : `FFT2dRR` (`FFT.h`, `FFT.cpp`). Shaders : `fft3dgpu.hlsl:202-454`. Enveloppeurs : `FFTPS.cpp`.

### 5.1 Principe

FFT radix-2 **décimée en temps** (bit-reversal en entrée), avec les papillons **fusionnés 2-en-1** : chaque pixel shader calcule un papillon complet et écrit les deux sorties dans **deux textures** `I` et `J` (MRT) ou deux passes. Les paires de textures sont réduites à une seule à la fin de l'axe par le shader `Collect*` guidé par `SaveLUT`. Tous les twiddles et tous les indices de lecture sont lus dans des **textures LUT** (`Factor`, `LoadMap`).

### 5.2 Textures et tailles (`FFT.cpp:130-149`)

| Texture | Taille | Rôle |
|---|---|---|
| `TempF[0]` | `xn × yn` | spectre horizontal complexe pleine taille (avant R2C / après C2R) |
| `TempI[0]`, `TempJ[0]`, `TempIpong[0]`, `TempJpong[0]` | `x2n × yn` (`x2n = xn/2`) | ping-pong horizontal (papillons, 2 textures par buffer) |
| `TempF[1]` | `xVn × yn` | spectre final (moitié de colonnes) |
| `TempI[1]`, `TempJ[1]`, `TempIpong[1]`, `TempJpong[1]` | `xVn × y2n` (`y2n = yn/2`) | ping-pong vertical |
| `TwiddleLUTforward/backward[i]` | `n2n[i] × log2n[i]`, `_float2` | twiddles par stage (i=0 horizontal, i=1 vertical) |
| `LoadLUT2[i]` | `n2n[i] × log2n[i]`, `_float2` | coordonnées de lecture + sélecteur I/J |
| `SaveLUT[i]` | `nn[i] × log2n[i]`, `_float2` | coordonnées d'écriture + sélecteur I/J |
| `R2CLUT` | `xVn × 1`, `_float4` | table conversion forward complexe→réel |
| `C2RLUT` | `xn × 1`, `_float4` | table conversion inverse réel→complexe |

Formats `_float4/_float2/_float1` : `FLOAT4/FLOAT2/FLOAT` si `precision > FLOAT16`, sinon `HALF4/HALF2/HALF` (`FFT.cpp:100-111`). Pool de sortie `FreeStreamPool` : `HALF4` sauf si `precision == FLOAT32_ALL` (`FFT.cpp:152`).

Constantes de stage (ligne de LUT au centre du texel) :

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

(`nnx`/`nny` sont calculés mais **jamais utilisés** — code mort, supprimé dans le destructeur `FFT.cpp:288-289`.)

### 5.3 Séquence exacte — forward (`FFT.cpp:399-431`, `448-496`, `512-546`)

`CalcFFT(src, dst, forward=true)` : `Img = src` ; `TempF[1]` = dst (ou texture de la pool) ; puis **horizontal puis vertical** ; `dst = TempF[1]`.

**Horizontal** (`CalcFFTHori`, forward) :

| # | Étape | Shader HLSL | Entrées | Sorties | Params |
|---|---|---|---|---|---|
| 1 | `BitReverseButterFlyH->Apply(Img, LoadLUT2[0], TempI[0], TempJ[0])` (`FFT.cpp:460`) | `BitReverseButterFlyH` (`fft3dgpu.hlsl:202-229`) | `Src`=Img, `LoadMap`=LoadLUT2[0] (ligne 0) | `TempI[0]` (o[0]), `TempJ[0]` (o[1]) | `PIXELOFFSET = stagex[0] = 0.5/log2x` |
| 2 | pour `i = 1 … log2x-1` : `ButterflyCollectH->Apply(TempI[0], TempJ[0], TwiddleLUTforward[0], LoadLUT2[0], stagex[i], TempIpong[0], TempJpong[0])` puis échange ping-pong (`FFT.cpp:475-485`) | `ButterflyCollectH` (`fft3dgpu.hlsl:338-376`) | `I`,`J`, `Factor`=TwiddleLUT, `LoadMap`=LoadLUT2, `_stage`=stagex[i] | pong I/J | `PIXELOFFSET = 1.0/x2n` (`FFT.cpp:240`) |
| 3 | `CollectH->Apply(TempI[0], TempJ[0], SaveLUT[0], stagex[log2x-1], TempF[0])` (`FFT.cpp:489`) | `CollectH` (`fft3dgpu.hlsl:421-437`) | `I`,`J`, `LoadMap`=SaveLUT[0], `_stage` | `TempF[0]` | — |
| 4 | `R2C->Apply(TempF[0], R2CLUT, Temp)` (`FFT.cpp:492`) | `R2C` (`fft3dgpu.hlsl:314-334`) | `Src`=TempF[0], `Factor`=R2CLUT | `Temp` (= ancien `TempF[1]`) | — |

**Vertical** (`CalcFFTVert`, forward) : mêmes étapes sur l'axe y, `LoadLUT2[1]`, `TwiddleLUTforward[1]`, `SaveLUT[1]`, ping-pong `TempI[1]/TempJ[1]` (hauteur `y2n`), entrée `Temp`, sortie finale `TempF[1]` (`FFT.cpp:520-543`).

### 5.4 Séquence exacte — inverse (`FFT.cpp:422-430`, `463-469`, `524-527`)

`CalcFFT(src, dst, forward=false)` : `Img = src` (spectre `xVn × yn`) ; `TempF[0] = dst` ; **vertical d'abord, puis horizontal** ; `dst = TempF[0]`.

| # | Étape | Détail |
|---|---|---|
| 1 | `BitReverseButterFlyV->Apply(Img, LoadLUT2[1], TempI[1], TempJ[1])` (`FFT.cpp:526`) | FFT verticale (longueur y, complexe pleine) sur les colonnes du spectre réel, twiddles `TwiddleLUTbackward[1]`, stages `stagey[i]`, `CollectV → TempF[1]` (`FFT.cpp:543`) |
| 2 | `C2R->Apply(TempF[1], C2RLUT, Temp)` (`FFT.cpp:465`) | referme le spectre réel horizontal en suite complexe demi-taille ; `Temp` = ancien `TempF[0]` (`xn × yn`) |
| 3 | `BitReverseButterFlyBH->Apply(Temp, LoadLUT2[0], TempI[0], TempJ[0])` (`FFT.cpp:469`) | bit-reversal + premier papillon **avec la normalisation 1/N** (voir §5.5) |
| 4 | stages `ButterflyCollectH` avec `TwiddleLUTbackward[0]` (`FFT.cpp:475-485`) | idem forward mais twiddles conjugués |
| 5 | `CollectH → TempF[0]` = dst (`FFT.cpp:489`) | sortie `xn × yn` |

### 5.5 Où est la normalisation 1/N (RADIX2LUT)

Uniquement dans le shader `BitReverseButterFlyBH`, appliqué **au tout début du premier papillon horizontal inverse** :

```cpp
// FFT.cpp:60 et 95
  , norm(1.0 / (_x*_y*_bt)), ...
  norm = 1.0 / (x*y*bt);
```

Note : à la ligne 60, `x`/`y` sont les membres déjà initialisés (`x = _x/2`), donc la valeur effective est `1/((_x/2)·_y·bt)` — la normalisation complète est répartie : le facteur 2 manquant vient du fait que la FFT horizontale est demi-taille et que la paire a+b est reconstituée sans facteur (à vérifier côté filtreur si besoin ; la valeur compilée dans le shader est `norm` telle que calculée ligne 95, recopiée dans la macro `NORM` en `FFTPS.cpp:163`).

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

Le facteur temporel `1/bt` est donc **fondu dans la normalisation spatiale** — il n'y a pas de passe de normalisation séparée pour le temporel.

### 5.6 Shaders RADIX2LUT — formules verbatim

**BitReverseButterFlyH** (`fft3dgpu.hlsl:202-229`) — bit-reversal + premier papillon horizontal :

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

`BitReverseButterFlyV` (`fft3dgpu.hlsl:264-290`) est identique avec les axes permutés (`Index.x=In.texCoord.y` pour l'index LUT, `Index.y=LM.x/y` pour la lecture).

**ButterflyCollectH** (`fft3dgpu.hlsl:338-376`) — papillon des stages 1+ avec twiddle :

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

Lecture de l'algèbre : `W = (Wr, Wi)` ; `temp = j·W` (multiplication complexe) ; sorties `i + temp` et `i − temp`. Le sélecteur `LM.y` (0 ou 1) dit dans quelle texture (I ou J) se trouve le partenaire du papillon. `PIXELOFFSET = 1/x2n` = un texel.

`ButterflyCollectV` (`fft3dgpu.hlsl:380-418`) : idem, axes permutés, `Offset.y=PIXELOFFSET` (`PIXELOFFSET = 1.0/y2n`, `FFT.cpp:242`).

**CollectH** (`fft3dgpu.hlsl:421-437`) — fusion I/J → texture unique :

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

`CollectV` (`fft3dgpu.hlsl:439-454`) : idem vertical.

**R2C** (forward, complexe demi-taille → spectre réel) (`fft3dgpu.hlsl:314-334`) :

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

`M = (posF, posN, cos, sin)` lu dans `R2CLUT` ; `F` = F(k), `N` = F(N−k) (positions miroir). Formule mathématique équivalente (composante 1) : `Re' = cos·Im(F−N)… ` — voir §10 pour la dérivation ; recopier le swizzle tel quel.

**C2R** (inverse, spectre réel → complexe demi-taille) (`fft3dgpu.hlsl:294-312`) :

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

C2R et R2C sont **conjugués** : les signes des termes en `M.zz` (cos) sont opposés.

### 5.7 Tables CPU RADIX2LUT — formules verbatim

**Twiddles — récurrence Numerical Recipes** (`FFT.cpp:743-768`) :

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

⚠ Les twiddles ne sont **pas** des `cos/sin` directs : ce sont les valeurs de la **récurrence de Numerical Recipes** (erreur accumulée le long du stage). `isign = +1` en forward, `−1` en inverse. Pour reproduire bit-à-bit, il faut reproduire cette récurrence en `double` à l'identique.

**Bit-reversal** (`FFT.cpp:714-723`) :

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

`ReverseBit` : version assembleur x86 (`ror`/`rcl`, `FFT.cpp:678-696`), version C portable `ReverseBit_c` (`FFT.cpp:664-676`) utilisée en x64. Résultat identique (permutation de bits).

**SaveMap** (`FFT.cpp:566-582`) :

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

**LoadMap** (`FFT.cpp:633-646`) :

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

**LoadMap2** (fusion bit-reversal + load par stage) (`FFT.cpp:605-615`) :

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

**Table C2R/R2C** (`FFT.cpp:788-820`) — attention, `forward=true` ici = « conversion complexe→réel », donc utilisée pour la **R2CLUT du sens forward** (`CreateC2RMap(R2CMap, x, true, xnum)`, `FFT.cpp:120`) et `forward=false` pour la `C2RLUT` inverse (`FFT.cpp:124`) :

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

Ici les twiddles de conversion sont des **`cos`/`sin` directs en double** (pas de récurrence), stockés en float. Les deux premiers champs sont les coordonnées texture des paires miroir `(i, N−i)`.

---

## 6. Algorithme 2 — STOCKHAM (`FFT2dRR2`)

Classe : `FFT2dRR2` (`fft2.h`, `fft2.cpp`). Shaders : `fft3dgpu.hlsl:1068-1204`. Enveloppeurs : `fft2ps.cpp`. Géométries : `core/vertexbuffer.cpp` (`FFTQuad0`, `FFTQuad2`, `C2RQuad`, `R2CFFTQuad`, `CollectQuad`).

### 6.1 Principe

FFT de **Stockham** (sans bit-reversal, ping-pong automatique). Les twiddles ne sont **pas** dans des textures : ils sont encodés dans les **coordonnées de texture des sommets** (interpolés par le rasteriseur), un quad par papillon. Chaque stage dessine deux groupes de quads (`DrawOdd`/`DrawEven`) pour répartir les deux sorties du papillon dans deux textures de destination.

Dimensions (`fft2.cpp:23-34`) : `cx = x/2` (largeur complexe), `rx = cx+1`, `_log2cx = log2(x/2)`, `_log2y = log2(y)`.

### 6.2 Textures (`fft2.cpp:36-45`)

| Texture | Taille | Rôle |
|---|---|---|
| `TempHorizontal1..4` | `srcwidth/2 × height` = `cx·xrep/2 × y·yrep` | ping-pong horizontal (2 textures par buffer) |
| `TempVertical1..4` | `width × height/2` = `(cx+1)·xrep × y·yrep/2` | ping-pong vertical |
| Pool de sortie `FreeStreamPool` | `width × height` | `HALF4` sauf `FLOAT32_ALL` (`fft2.cpp:65`) |

Format interne : `precision > FLOAT16 ? FLOAT4 : HALF4` (`fft2.cpp:36`).

### 6.3 Séquence exacte — forward (`fft2.cpp:96-123`)

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

### 6.4 Séquence exacte — inverse (`fft2.cpp:124-150`)

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

### 6.5 Normalisation (STOCKHAM)

```cpp
// fft2.cpp:25
  , norm(2.0 / (x*y*z))
// fft2.cpp:61-62
  CombineV = NEW psCombine(pDevice, xrep, rx, yrep, y, false, 1);
  CombineH = NEW psCombine(pDevice, xrep, cx, yrep, y, true, norm);
```

- Forward : `CombineV` avec `NORM=1` (aucune normalisation).
- Inverse : `CombineH` avec `NORM = 2.0/(x·y·z)` (x,y = taille du bloc, z = bt). Le facteur **2** compense le `0.5` appliqué deux fois dans le shader `R2Cfft` (voir §6.8) : le `0.5` du R2Cfft divise par 2, le `2.0` du Combine le rétablit.

Shader Combine (`fft3dgpu.hlsl:1152-1154`) :

```hlsl
float4 Combine(VS_OUTPUT1 In):COLOR{
return tex2D(Src0,In.texCoord)*NORM;
}
```

`NORM` est une macro compilée (`fft2ps.cpp:257`). Défaut `1.0` si non défini (`fft3dgpu.hlsl:1148-1150`).

### 6.6 Papillon STOCKHAM — vertex shaders et twiddles

**Vertex shaders** (`fft3dgpu.hlsl:1068-1080`) :

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

- `fft2` (stages ≥ 1) : le sommet porte `texCoord[1] = (cos, sin)` déjà calculés CPU ; le VS **inverse le signe de la composante sin** selon `dir` (`dir = forward ? 1 : -1`, `fft2ps.cpp:49-50` et `103-104`).
- `fft0` (stage 0) : le sommet porte un **angle** dans `texCoord[1].x` ; le VS inverse le signe de l'**angle** avant le `sincos` GPU.

**Twiddles des stages ≥ 1** — calculés CPU en double, stockés en float32 dans le vertex buffer (`core/vertexbuffer.cpp:2774`, répété en 2782/2790/2798) :

```cpp
v[nvertex].tex[1].u = cos(-pi * m / n); v[nvertex].tex[1].v = sin(-pi * m / n);
```

avec `n` = taille du bloc complexe, `m` = indice du papillon dans le stage (boucle `for (int l = 0, m = 0; l < parts; l++, offset += 2 * size, srcoffset += size, m += 2 * size)`, `core/vertexbuffer.cpp:2764`). Angle forward = `−π·m/n`.

**Stage 0** (`FFTQuad0`, `core/vertexbuffer.cpp:2871-2958`) : l'angle est encodé par coin de quad, interpolé linéairement sur le quad : `tex[1].u ∈ {0, −π}` (`core/vertexbuffer.cpp:2902, 2909, 2911, 2920, 2929`). Le pixel shader fait `sincos(I.x, T.y, T.x)` → `T = (cos, sin)` de l'angle interpolé.

**Stage 0 — détail important** : `FFTQuad0::Execute`/`psButterfly*::Apply(stage 0)` passe bien `dir` (`fft2ps.cpp:49-50`), mais l'angle étant 0 ou −π, `e^{−iπ} = e^{+iπ} = −1` : le signe du sens n'a aucun effet numérique au stage 0.

**Découpage odd/even** : `FFTQuad2` pré-construit la géométrie de tous les stages (`core/vertexbuffer.cpp:2756-2805`) et le stage `s` se dessine via `stageLUT[endstage - s]` (`core/vertexbuffer.cpp:2846-2856`) :

```cpp
HRESULT FFTQuad2::DrawOdd(int stage)
{
  int i = endstage - stage;
  return _pDevice->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, stageLUT[i].minvertexodd, stageLUT[i].nvertex, stageLUT[i].startindexodd, stageLUT[i].nvertex / 2);
}
```

Chaque stage dessine d'abord les quads « odd » (sortie `a+b` → dst1), puis « even » (sortie `(a−b)·W` → dst2) — voir le PS ci-dessous. `LOADOFFSET` (macro `float2`) donne le décalage de lecture du second membre : `1/(xrep·2)` en u pour l'horizontal (`fft2ps.cpp:9`), `1/(yrep·2)` en v pour le vertical (`fft2ps.cpp:116`), soit une demi-taille de bloc en texels.

### 6.7 Papillon STOCKHAM — pixel shader verbatim (`fft3dgpu.hlsl:1112-1144`)

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

Déroulé de l'algèbre (à reproduire en OpenCL) : `T = (c, s)` ; `T1 = (c, c, −s, s)` ; `temp = a − b = (tr1, tr2, ti1, ti2)` ;

```
Out = (ti1·(−s) + tr1·c,  ti2·(−s) + tr2·c,  tr1·s + ti1·c,  tr2·s + ti2·c)
    = (Re1', Re2', Im1', Im2')   avec  (Re'+i·Im') = (tr+i·ti)·(c+i·s)
```

soit exactement `(a−b)·W` composante par composante complexe, **plus** la sortie `a+b`. Les deux sorties sont les deux RT (MRT) ou deux passes (`PASS0` = `a+b`, l'autre = le produit).

### 6.8 C2Rfft / R2Cfft STOCKHAM — verbatim

**R2Cfft** (inverse, réel→complexe, `fft3dgpu.hlsl:1156-1171`) :

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

L'angle vient du sommet (`texCoord[2].x`), interpolé de **0 à π** sur la largeur du bloc (`R2CFFTQuad`, `core/vertexbuffer.cpp:2631, 2638, 2645, 2652` : `v[nvertex].tex[2].u = 0` à gauche, `= pi` à droite) → angle effectif `π·i/x`. Le `sincos` est fait **sur le GPU**. Les deux lectures `a` (colonne i) et `b` (colonne miroir x−i) viennent de la **même** texture (`Src0`), quads `DrawOdd`/`DrawEven` (`fft2ps.cpp:289-302`).

**C2Rfft_ps** (forward, complexe→réel, `fft3dgpu.hlsl:1177-1204`) :

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

La géométrie `C2RQuad` (`core/vertexbuffer.cpp:2249-2476`) dessine **5 quads par bloc** couvrant les `x+1` colonnes du spectre réel, en 4 groupes avec drapeaux constants (`fft2ps.cpp:232-253`) :

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

- **Draw00** (colonnes 0 et x, cas DC/Nyquist) : `t00=true` ; `texCoord[2].x = ±1` encodé par sommet (`core/vertexbuffer.cpp:2289` : `v[nvertex].tex[2].u = (j == 0 ? 1 : -1)`) → `Out.xy = a.xy ± a.zw`, imaginaire = 0.
- **Draw01 / Draw10** (cas général) : formule avec `sincos` GPU ; angles des coins `2 * pi * 1 / (2 * x)` et `2 * pi*(x / 2) / (2 * x)` (`core/vertexbuffer.cpp:2325, 2332`) et `2 * pi*(1 + x / 2) / (2 * x)`, `2 * pi*(x) / (2 * x)` (`core/vertexbuffer.cpp:2396, 2403`) — interpolés linéairement, donc angle exact `2π·i/(2x)` par pixel.
- **Draw11** (colonne centrale) : `t11=true` ; `Out.xy = a.xy ; Out.zw = -a.zw`.

Comparer avec le R2Cfft : le signe du terme en `T.x` est **conjugué** entre forward (C2Rfft, `+(a.zw+b.zw)*T.x`) et inverse (R2Cfft, `-(a.zw+b.zw)*T.x`), et le terme en `T.y` passe de `-(a.xy-b.xy)` à `+(b.xy-a.xy)`.

### 6.9 Combine (STOCKHAM)

`psCombine` (`fft2ps.cpp:256-278`) utilise `CollectQuad` (`core/vertexbuffer.cpp:2478-2594`) : deux draws (odd/even) qui rassemblent les deux demi-textures ping-pong dans la texture finale, avec multiplication par `NORM` (macro).

---

## 7. FFT temporelle 1D (2 et 3 points)

Quand `bt = 2` ou `3`, la dimension temporelle est traitée par un mini-FFT direct sur 2 ou 3 **textures** (une par image du buffer temporel), sans twiddle.

Shaders (`fft3dgpu.hlsl:456-535`) :

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

- `FFT3p` : FFT de 3 points ; `−0.8660254 = sin(π·4/3) = −√3/2` (constante **tronquée à 7 décimales** dans le shader — à recopier telle quelle pour la compatibilité bit-à-bit). La rotation ±√3/2 est faite par swizzle `zwxy` (échange Re/Im avec signe).
- Inverse : **une seule sortie** est conservée (`iFFT2p` renvoie `src1 − src2`, `iFFT3p` renvoie `y1 − y2`) — les filtres n'ont besoin que de l'image courante.
- Orchestration : `FFT2dRR::CalcFFT1d` (`FFT.cpp:312-331` pour des textures simples, `FFT.cpp:350-379` pour des paires de textures « mode 1 »). En inverse, le résultat est écrit dans `dst[1]` uniquement.
- La normalisation `1/bt` est **déjà fondue dans `norm` spatial** (§5.5), il n'y a pas de passe temporelle.

---

## 8. Le paramètre `precision` (0/1/2) et les formats de texture

```cpp
// FFT.h:39-43
enum { FLOAT16, FLOAT32_FFT, FLOAT32_ALL };
```

- `0 = FLOAT16` : tout le pipeline FFT (textures internes **et** LUT) en demi-flottant.
- `1 = FLOAT32_FFT` : FFT en float32, **sortie** en half.
- `2 = FLOAT32_ALL` : tout en float32, sortie comprise.

Mapping des formats (`core/texture.cpp:42-178`, `GPUTYPES`) :

| Type logique | Format D3D9 |
|---|---|
| `HALF4` | `D3DFMT_A16B16G16R16F` |
| `FLOAT4` | `D3DFMT_A32B32G32R32F` |
| `HALF2` | `D3DFMT_G16R16F` |
| `FLOAT2` | `D3DFMT_G32R32F` |

Choix des formats :

| Objet | RADIX2LUT | STOCKHAM |
|---|---|---|
| Textures de travail | `FFT.cpp:100-111` : `precision>0` → FLOAT*, sinon HALF* | `fft2.cpp:36` : `precision > FLOAT16 ? FLOAT4 : HALF4` |
| LUT (twiddle, load, save, C2R, R2C) | même règle (`FFT.cpp:165-172`) — donc **en half si precision=0** | n/a (twiddles dans les sommets, toujours float32) |
| Pool de sortie | `FFT.cpp:152` : `precision != FLOAT32_ALL ? HALF4 : FLOAT4` | `fft2.cpp:65` : idem |

Conversion au téléversement : `UploadToTexture` convertit les `float` CPU en `D3DXFLOAT16` quand la texture est half (`core/texture.cpp:891-903`).

**Point capital : il n'existe aucune macro de précision dans le HLSL.** Tous les calculs du pixel shader sont en `float32` (HLSL SM3.0), quel que soit `precision`. Le paramètre ne change **que le stockage** (textures de travail, LUT, sortie). Avec `precision=0`, les twiddles eux-mêmes sont arrondis en fp16 (précision ~3 chiffres décimaux) avant même d'entrer dans le shader.

Override : si `bits_per_pixel > 8`, le plugin force `FLOAT32_ALL` (`FFT3dGPU.cpp:244-246`).

---

## 9. Mode MEASURE

`FFT3dGPU.cpp:103-127` : `MeasureFFT` crée les deux implémentations (`FFT2dRR` et `FFT2dRR2`), exécute un certain nombre de `CalcFFT` sur des textures de test, chronomètre, et conserve la plus rapide ; le choix est exposé via l'argument `fftnode` (`MEASURE`/`RADIX2LUT`/`STOCKHAM`, dispatch `FFT3dGPU.cpp:363-387`). C'est un **benchmark pur GPU** — il n'y a pas d'implémentation CPU en secours.

---

## 10. Reproductibilité numérique

Liste exhaustive des facteurs qui déterminent les bits du résultat. Pour un port OpenCL « bit-exact », chacun doit être reproduit ou consciemment écarté.

### 10.1 Ordre des additions

- **RADIX2LUT** : radix-2 décimé en temps, ordre des stages `m = 2, 4, 8, …` (`CreateTwiddleMap` `FFT.cpp:748`), papillons `i ± j·W` dans l'ordre des paires données par `LoadMap`/`SaveMap`. Chaque sortie est la somme de **deux** termes seulement à chaque stage — l'ordre est entièrement fixé par les LUT.
- **STOCKHAM** : stages `1 … log2n-1` avec décalage `size = n/2^stage` (`core/vertexbuffer.cpp:2756`), sortie `a+b` puis `(a−b)·W`.
- Les deux algorithmes **ne produisent pas les mêmes bits** entre eux (ordre d'accumulation différent) — le choix `fftnode` change le résultat numérique, pas seulement la vitesse.

### 10.2 Origine des twiddles (la plus grosse source d'écart)

| Cas | Origine | Précision |
|---|---|---|
| RADIX2LUT stages | **récurrence NR** CPU (`FFT.cpp:750-763`), erreur accumulée le long du stage | double → float32 (ou **half** si precision=0) |
| RADIX2LUT C2R/R2C | `cos(2πi/N)`, `sin(2πi/N)` CPU directs (`FFT.cpp:810-811, 816-817`) | double → float32 (ou half) |
| STOCKHAM stages ≥ 1 | `cos(-pi*m/n)`, `sin(-pi*m/n)` CPU (`core/vertexbuffer.cpp:2774`) | double → float32 (vertex buffer) |
| STOCKHAM stage 0 | **`sincos` GPU** sur angle interpolé 0→−π (`fft3dgpu.hlsl:1122`) | dépend du matériel |
| STOCKHAM C2Rfft/R2Cfft | **`sincos` GPU** sur angle interpolé (`fft3dgpu.hlsl:1164, 1198`) | dépend du matériel |

⚠ Le `sincos` HLSL est compilé en instructions matérielles (souvent `sin/cos` MUFU sur NVIDIA, précision inférieure à 1 ulp non garantie) ; OpenCL `sincosf`/`sin`/`cos` n'est **pas** identique. Pour de la reproductibilité cross-vendor, remplacer par des tables pré-calculées CPU en double (comme le font déjà les stages ≥ 1 de STOCKHAM et toutes les LUT RADIX2LUT).

⚠ La récurrence NR (`FFT.cpp:762-763`) doit être recopiée **exactement**, avec la réutilisation de variable `(wtemp = wr)` qui fait partie de la formule.

### 10.3 Précision de stockage

- `precision=0` : **toutes** les LUT (twiddles, load, save, C2R, R2C) sont stockées en `A16B16G16R16F`/`G16R16F` — les twiddles sont arrondis en fp16 avant usage. C'est le mode historique par défaut (`bits_per_pixel ≤ 8`).
- Les sommets STOCKHAM sont toujours float32 (le vertex buffer n'est jamais half).
- Sortie : half sauf `FLOAT32_ALL`.

### 10.4 Constantes et facteurs figés

- `−0.8660254` (sin(4π/3) tronqué) dans `FFT3p`/`iFFT3p` (`fft3dgpu.hlsl:491, 530`).
- `0.5` des formules C2R/R2C et `0.5` du R2Cfft STOCKHAM (`fft3dgpu.hlsl:1167-1168`), compensé par le `2.0/(x·y·z)` du `CombineH` (`fft2.cpp:25, 62`).
- `norm = 1.0/(x*y*bt)` RADIX2LUT (`FFT.cpp:95`), compilé dans le shader `BitReverseButterFlyBH`.
- Angles STOCKHAM en `pi` double côté CPU (`const float pi(acos(-1.0))`, `fft2ps.cpp:3` ; `pi` global dans vertexbuffer.cpp).

### 10.5 Échantillonnage et adressage

- Sampler **POINT** partout dans les FFT (les coordonnées sont calées au centre des texels, §3.1). Un port OpenCL doit utiliser des index entiers équivalents ; tout écart d'arrondi dans le calcul des coordonnées (`(i+0.5)/W` en float) peut décaler d'un texel — à remplacer par de l'arithmétique entière.
- Adressage **MIRROR** pendant l'exécution (`FFT3dGPU.cpp:816-820`) : pertinent uniquement pour les lectures hors plage (ne devrait pas arriver avec des LUT correctes ; à vérifier en port).

### 10.6 Signe forward/inverse

- RADIX2LUT : deux tables de twiddles distinctes (`isign = ±1`, `FFT.cpp:745`).
- STOCKHAM : même géométrie, constante `dir` (`fft2ps.cpp:49-50, 103-104`) qui inverse soit la valeur sin (VS `fft2`), soit l'angle (VS `fft0`).

---

## Annexe A — Récapitulatif des shaders HLSL

| Shader (`fft3dgpu.hlsl`) | Ligne | Algo | Rôle |
|---|---|---|---|
| `Img2toFloat4` / `_2` | 113-135 / 165-183 | entrée | packing image → float4 (hors FFT) |
| `Float4toImg2` / `_2` | 137-161 / 185-198 | sortie | unpacking (hors FFT) |
| `BitReverseButterFlyH` | 202-229 | RADIX2LUT fwd H | bitrev + papillon 1 |
| `BitReverseButterFlyBH` | 233-259 | RADIX2LUT inv H | idem + ×NORM |
| `BitReverseButterFlyV` | 264-290 | RADIX2LUT H+V | idem vertical |
| `C2R` | 294-312 | RADIX2LUT inv | réel → complexe |
| `R2C` | 314-334 | RADIX2LUT fwd | complexe → réel |
| `ButterflyCollectH` | 338-376 | RADIX2LUT | papillon stages 1+ (H) |
| `ButterflyCollectV` | 380-418 | RADIX2LUT | papillon stages 1+ (V) |
| `CollectH` / `CollectV` | 421-437 / 439-454 | RADIX2LUT | fusion I/J → 1 texture |
| `FFT2p` / `iFFT2p` | 456-472 / 474-479 | temporel | FFT 2 points |
| `FFT3p` / `iFFT3p` | 481-519 / 521-535 | temporel | FFT 3 points |
| `fft2` (VS) | 1068-1073 | STOCKHAM | flip signe sin (dir) |
| `fft0` (VS) | 1075-1080 | STOCKHAM | flip signe angle (dir) |
| `passthough3` / `passthough` | 1082-1085 / 1087-1090 | STOCKHAM | VS identité |
| `Butterfly` | 1114-1143 | STOCKHAM | papillon (STAGE0 = sincos GPU) |
| `Combine` | 1152-1154 | STOCKHAM | ×NORM |
| `R2Cfft` | 1156-1171 | STOCKHAM inv | réel → complexe |
| `C2Rfft_ps` | 1177-1204 | STOCKHAM fwd | complexe → réel (t00/t11) |

## Annexe B — Récapitulatif des géométries de sommets

| Classe (`core/vertexbuffer.cpp`) | Ligne | Algo | Encodage |
|---|---|---|---|
| `NQuad` | 40-169 | RADIX2LUT | quad plein écran, texCoord = (i+0.5)/W |
| `C2RQuad` | 2249-2476 | STOCKHAM fwd | 5 quads/bloc, angles `2πi/(2x)` par coins, ±1 pour DC/Nyquist |
| `CollectQuad` | 2478-2594 | STOCKHAM | rassemblement odd/even pour `Combine` |
| `R2CFFTQuad` | 2598-2715 | STOCKHAM inv | angles interpolés 0→π, paires miroir |
| `FFTQuad2` | 2723-2867 | STOCKHAM | quads par stage, twiddles `cos/sin(-πm/n)` par sommet, stageLUT |
| `FFTQuad0` | 2871-2983 | STOCKHAM | stage 0, angles 0/−π par coin |

## Annexe C — Correspondance DX9 → OpenCL

| Concept DX9 | Équivalent OpenCL suggéré |
|---|---|
| Pixel shader + quad plein écran | kernel 2D, un work-item par texel de sortie |
| Texture LUT + `tex2D` (POINT, +0.5) | tableau en `__global`/`__constant`, index entier |
| MRT (2 sorties) | le work-item écrit les 2 sorties (ou 2 kernels identiques) |
| Vertex buffer (twiddles interpolés) | table de twiddles en mémoire, index calculé (attention : interpolation linéaire des angles sur les quads C2RQuad/R2CFFTQuad/FFTQuad0 = valeur exacte aux centres, donc table directe `i` → angle) |
| `sincos` HLSL | table CPU double→float (reproductibilité) ou `sincosf` (rapide, non reproductible) |
| `D3DFMT_A16B16G16R16F` | `half` (extension cl_khr_fp16) ou float + arrondi half explicite |
| Sampler MIRROR | non nécessaire si index validés |
| Compilation runtime HLSL | kernels pré-compilés + spécialisation par `#define` (NORM, PIXELOFFSET deviennent des paramètres) |

**Recommandations de portage** :

1. Porter les deux algorithmes (RADIX2LUT et STOCKHAM) : ils ne sont pas numériquement équivalents, le filtre historique utilise l'un ou l'autre selon `fftnode`.
2. Recopier la récurrence NR des twiddles à la virgule près (`FFT.cpp:750-763`) si l'objectif est le bit-exact avec le plugin d'origine en RADIX2LUT.
3. Éliminer les `sincos` GPU (STOCKHAM stage 0, C2Rfft, R2Cfft) en pré-calculant CPU les mêmes valeurs que `cos/sin(double)` du vertex buffer d'origine — c'est déjà le cas pour les stages ≥ 1 de STOCKHAM.
4. Conserver l'ordre des stages et des paires (LUT) tel quel ; ne pas « optimiser » la factorisation.
5. Geler les constantes : `−0.8660254`, `0.5`, `2.0/(x·y·z)`, `1/(x·y·bt)`.
6. Prévoir le mode fp16 (precision=0) : arrondir les LUT **et** les textures de travail en half au moment du stockage, pas des calculs.
