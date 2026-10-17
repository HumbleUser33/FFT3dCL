# -*- coding: utf-8 -*-
# Exemple d'utilisation de FFT3DGPU dans un script VapourSynth (StaxRip ou vspipe).
#
# Si le DLL n'est pas dans le dossier plugins de VapourSynth, décommentez la
# ligne LoadPlugin ci-dessous et ajustez le chemin.

import vapoursynth as vs

core = vs.core

# core.std.LoadPlugin(r"C:\chemin\vers\fft3dgpu.dll")

# Source (exemple avec BestSource ; dans StaxRip la source est deja definie).
clip = core.bs.VideoSource(source=r"C:\chemin\vers\Sample.mkv")

# Eventuellement convertir en 16 bits / format interne si besoin (pas obligatoire :
# le filtre accepte 8-16 bits entiers et float 32 bits directement).

# Denoise FFT3DGPU (reglages identiques a l'original 0.8.6).
clip = core.fft3dgpu.FFT3DGPU(
    clip,
    sigma=2.0,        # force de denoising
    bt=3,             # fenetre temporelle (-1..4, 0 = Kalman)
    beta=1.0,
    plane=0,          # 0 = luminance seule
    mode=1,           # recouvrement de blocs
    bw=32, bh=32,     # taille de bloc (puissance de 2)
    degrid=1.0,       # correction de grille
    device=2,         # 2 = GPU avec repli CPU (defaut), 1 = GPU, 0 = CPU
)

clip.set_output()
