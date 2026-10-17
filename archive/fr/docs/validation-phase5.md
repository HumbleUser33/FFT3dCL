# Validation — Phase 5 (emballage StaxRip)

Objectif : vérifier que le plugin se charge et rend correctement dans un
contexte réel (StaxRip / vspipe), sur les vrais clips 4K HDR, et que le DLL
est autonome.

## 1. DLL autonome

`dumpbin /DEPENDENTS fft3dgpu.dll` → une seule dépendance : `KERNEL32.dll`.
- CRT Visual C++ lié en statique (pas de `VCRUNTIME140.dll` / `MSVCP140.dll`).
- `OpenCL.dll` chargé dynamiquement à l'exécution (présent avec tout pilote GPU).
- Taille : ~258 Ko.

Conclusion : le DLL se dépose tel quel, rien d'autre à installer.

## 2. Chargement

Deux méthodes testées, toutes deux valides :
- **Auto-load** : copie du DLL dans le dossier `plugins` de VapourSynth
  (`...\site-packages\vapoursynth\plugins`) → chargé sans variable d'environnement.
- **`core.std.LoadPlugin(r"...fft3dgpu.dll")`** : fonctionne (exit 0).

StaxRip : déposer le DLL dans son dossier plugins VapourSynth, ou ajouter la
ligne `LoadPlugin` en tête du script.

## 3. Rendus réels (GPU, device=1)

| Clip | Résolution / format | Frames | Temps filtre | Résultat |
|------|--------------------|--------|--------------|----------|
| Sample.mkv | 3840x2160 HEVC Main10 HDR10 | 8 | 2,13 s total | OK, sans erreur |
| The Gorge 2025 | 1920x1038 AV1 10-bit HDR10 | 8 | 0,71 s (11,33 fps) | OK, sans erreur |

Le temps total de The Gorge (≈288 s) est dominé par le **décodage AV1 10-bit**
du fichier lossless de 6 Go (L-SMASH), pas par le filtre. La mémoire GPU
(RTX 3050, 8 Go) tient le 4K plein cadre sans problème.

## 4. Comparaison 1080p 10-bit vs original (GPU)

Crop 1920x1080, 8 frames, `yuv420p10le`, GPU vs original Avisynth+ 0.8.6
(`precision=2`) :
- échantillons identiques : **96,64 %**
- max écart : **4** (LSB 10-bit) — cohérent avec le petit clip 10-bit.

## 5. Fuite de frames (corrigée)

Symptôme initial : `Warning: Core freed but N filter instance(s) still exist`
+ octets encore alloués. Cause : les frames obtenues par `getFrameFilter`
(`src` + frames de la fenêtre temporelle) n'étaient jamais libérées.
Correctif : `vsapi->freeFrame()` ajouté sur `src` et sur chaque frame de
fenêtre (chemins CPU et GPU).

Après correctif : le compteur restant est **borné par le cache du core**
(~20 frames, indépendant du nombre total de frames) → pas de fuite qui croît.
`fft3dgpuFree` est bien appelé. Ce warning est **courant chez les filtres
temporels tiers** (blur, DFTTest, FrameRateConverter) : artefact bénin de fin
de vie, sans effet sur la sortie.

## 6. Paquet

`release\fft3dgpu\` :
- `fft3dgpu.dll` (identique au build courant — hash vérifié)
- `README.md` (installation StaxRip + usage + notes)
- `sample.vpy` (exemple de script)

## Verdict

Phase 5 validée : DLL autonome, chargement auto + LoadPlugin OK, rendus 4K
HDR réels OK, 1080p 10-bit conforme à l'original, fuite corrigée. Le plugin
est prêt pour StaxRip.
