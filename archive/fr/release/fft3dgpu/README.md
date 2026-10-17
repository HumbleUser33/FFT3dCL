# FFT3DGPU — port VapourSynth (GPU OpenCL)

Port moderne du débruiteur GPU **FFT3dGPU v0.8.6** (Avisynth+) en plugin
**VapourSynth natif (API 4)**, fonctionnant sur carte graphique **NVIDIA, AMD
et Intel** via OpenCL 1.2. La méthode de débruitage et tous les paramètres
d'origine sont conservés à l'identique.

Espace de noms : `fft3dgpu` — fonction : `core.fft3dgpu.FFT3DGPU(clip, ...)`.

---

## 1. Installation

Copier `fft3dgpu.dll` dans le dossier des plugins VapourSynth. Le DLL est
**autonome** (CRT statique, aucune DLL Visual C++ requise ; OpenCL est chargé
à l'exécution depuis le pilote).

### Avec StaxRip
StaxRip embarque son propre VapourSynth. Deux options :

- **Dossier plugins** (recommandé) : déposer `fft3dgpu.dll` dans
  `<dossier StaxRip>\Apps\Plugins\VapourSynth\` (ou le dossier `plugins` de
  votre VapourSynth). Il sera chargé automatiquement.
- **Chargement explicite** : ajouter en tête du script VapourSynth
  (`*.vpy`) :
  ```python
  core.std.LoadPlugin(r"C\chemin\vers\fft3dgpu.dll")
  ```

### Vérification rapide
```
vspipe --info votre_script.vpy
```
doit lister le filtre `FFT3DGPU` sans erreur.

---

## 2. Utilisation

Exemple minimal (débruitage luminance, réglages par défaut identiques à
l'original) :

```python
import vapoursynth as vs
core = vs.core
clip = core.std.AcquireSource(...)          # votre source
clip = core.fft3dgpu.FFT3DGPU(clip)
clip.set_output()
```

Exemple avec paramètres typiques :

```python
clip = core.fft3dgpu.FFT3DGPU(
    clip,
    sigma=2.0,      # force de débruitage
    bt=3,           # fenêtre temporelle (-1..4 ; 0 = Kalman)
    beta=1.0,
    plane=0,        # 0 = luminance, 1 = chroma, 4 = tout
    mode=1,         # 0 ou 1 (recouvrement de blocs)
    bw=32, bh=32,   # taille de bloc (puissance de 2)
    degrid=1.0,     # correction de grille (mode 1)
    sharpen=0.0,
)
```

Tous les paramètres d'origine sont acceptés : `sigma, beta, bw, bh, bt,
sharpen, plane, mode, bordersize, precision, degrid, scutoff, svr, smin,
smax, kratio, ow, oh, wintype, interlaced, sigma2, sigma3, sigma4, oldfft`.
(`NVPerf` a été supprimé.)

### Paramètre propre au port : `device`
Sélection du moteur — **sans aucun effet sur le débruitage** :
- `device=2` (défaut) : GPU OpenCL, avec repli automatique sur le CPU si le
  GPU échoue ;
- `device=1` : GPU obligatoire (erreur si indisponible) ;
- `device=0` : CPU de référence.

---

## 3. Notes

- **Formats** : 8 à 16 bits entiers et float 32 bits, gris/YUV planaire.
  Le 10 bits HDR10 fonctionne (le chemin d'origine force `precision=2`).
- **Précision** : le port calcule en fp32. Comparé à l'original, l'écart est
  de ±1 à 2 LSB (8 bits) / ±4 (10 bits) — bruit d'arrondi entre implémentations
  FFT indépendantes. Voir `docs/validation-phase2.md` et `docs/validation-phase3.md`.
- **Déterminisme** : sortie identique à chaque exécution et quel que soit le
  nombre de threads.
- **Vitesse** : ~5× plus rapide en GPU qu'en CPU (mesuré RTX 3050).
- **AMD/Intel** : les kernels respectent les règles de portabilité OpenCL 1.2
  (pas de fp64, pas d'atomics, barrières seules). Validé numériquement sur
  NVIDIA RTX 3050 ; la compatibilité AMD/Intel repose sur ces règles.
- **Limite connue** : recouvrement non standard (`ow`/`oh` ≠ moitié du bloc)
  approximatif ; le réglage par défaut (recouvrement demi-bloc) est exact.

## 4. Différences volontaires vs original
1. `bt=-1` fait le vrai aller-retour FFT (l'original sortait du noir).
2. Mode 0 déterministe (l'original sortait parfois du noir).
3. Paramètre `device` ajouté (sélection moteur uniquement).
4. `NVPerf` supprimé ; `oldfft` conservé.
