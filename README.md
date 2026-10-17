# FFT3dOpenCL

Port moderne de **FFT3dGPU v0.8.6** (filtre Avisynth+ uniquement, DirectX 9) en
plugin **VapourSynth natif (API 4)**, avec un backend **OpenCL 1.2** qui tourne
sur carte graphique **NVIDIA, AMD et Intel**.

La méthode de débruitage et **les 23 paramètres d'origine sont conservés à
l'identique**. Seul `NVPerf` a été supprimé (il ne concernait que DirectX 9).

```python
clip = core.fft3dgpu.FFT3DGPU(clip, sigma=2.0, bt=3)
```

---

## État du projet

| Élément | État |
|---|---|
| Backend CPU de référence | ✅ complet et validé |
| Backend GPU OpenCL | ✅ complet et validé |
| Plugin VapourSynth API 4 | ✅ |
| Conformité vs original 0.8.6 | ✅ écart ±1–2 LSB (8 bits), ±4 (10 bits) |
| Déterminisme (threads, relances) | ✅ 100 % identique |
| Rendus réels 4K HDR10 | ✅ |
| AMD / Intel | ⚠️ non testés (portabilité garantie par les règles OpenCL 1.2 strictes) |

Les écarts avec l'original viennent de deux implémentations FFT indépendantes
(bruit d'arrondi), pas d'un changement d'algorithme. Détail complet dans
[`docs/validation-phase2.md`](docs/validation-phase2.md),
[`docs/validation-phase3.md`](docs/validation-phase3.md) et
[`docs/validation-phase5.md`](docs/validation-phase5.md).

---

## Installation

Le DLL est **autonome** : CRT Visual C++ lié en statique, `OpenCL.dll` chargé à
l'exécution par le pilote graphique. Une seule dépendance : `KERNEL32.dll`.

Copier `release/fft3dgpu/fft3dgpu.dll` dans le dossier `plugins` de VapourSynth,
ou charger explicitement :

```python
core.std.LoadPlugin(r"C:\chemin\vers\fft3dgpu.dll")
```

Avec **StaxRip** : déposer le DLL dans le dossier plugins VapourSynth de StaxRip
(`Apps\Plugins\VapourSynth\`), puis appeler le filtre dans le script.
Voir [`release/fft3dgpu/README.md`](release/fft3dgpu/README.md) et
[`release/fft3dgpu/sample.vpy`](release/fft3dgpu/sample.vpy).

---

## Paramètres

Tous les paramètres d'origine sont acceptés :

`sigma`, `beta`, `bw`, `bh`, `bt`, `sharpen`, `plane`, `mode`, `bordersize`,
`precision`, `degrid`, `scutoff`, `svr`, `smin`, `smax`, `kratio`, `ow`, `oh`,
`wintype`, `interlaced`, `sigma2`, `sigma3`, `sigma4`, `oldfft`.

Un paramètre propre au port, **sans effet sur le débruitage** :

| `device` | Comportement |
|---|---|
| `2` (défaut) | GPU OpenCL, repli automatique sur le CPU si le GPU échoue |
| `1` | GPU obligatoire (erreur si indisponible) |
| `0` | CPU de référence |

Formats acceptés : 8 à 16 bits entiers et float 32 bits, gris ou YUV planaire
(le HDR10 10 bits fonctionne).

---

## Performance

Mesuré sur RTX 3050 : le chemin GPU est environ **5× plus rapide** que le chemin
CPU (1280×720 float : 1,02 s en GPU contre 5,49 s en CPU). Le 4K plein cadre
tient dans 8 Go de mémoire vidéo.

---

## Compilation

Windows, Visual Studio 2022 (MSVC), CMake. Les en-têtes VapourSynth sont
fournis dans `third_party/`.

```
cmake -S . -B build
cmake --build build --config Release
```

Le DLL result est `build/Release/fft3dgpu.dll`.

---

## Différences volontaires avec l'original

Liste complète dans [`docs/porting-map.md`](docs/porting-map.md). Les principales :

1. `bt=-1` fait le vrai aller-retour FFT (l'original sortait du noir sur la luminance).
2. Mode 0 déterministe (l'original sortait parfois des images noires).
3. Paramètre `device` ajouté (choix du moteur uniquement).
4. `NVPerf` supprimé ; `oldfft` conservé mais ignoré.
5. Recouvrement non standard (`ow*2 != bw` ou `oh*2 != bh`) approximatif :
   le recouvrement demi-bloc, qui est le réglage par défaut, est exact.

---

## Licence

GNU GPL v2, comme le projet d'origine
[pinterf/FFT3dGPU](https://github.com/pinterf/FFT3dGPU), dont ce port est dérivé.
