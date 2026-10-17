# Coarse map of (port - ref) luma averaged over 16x16 cells.
import subprocess, sys

FFMPEG = r"C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\env\avs\ffmpeg.exe"
W, H = 256, 192


def decode(path):
    p = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                       capture_output=True)
    return p.stdout


ref = decode(sys.argv[1]); port = decode(sys.argv[2])
frame = int(sys.argv[3]) if len(sys.argv) > 3 else 0
base = frame * W * H * 3 // 2
for cy in range(0, H, 16):
    row = []
    for cx in range(0, W, 16):
        s = 0
        for y in range(cy, cy + 16):
            for x in range(cx, cx + 16):
                s += port[base + y * W + x] - ref[base + y * W + x]
        row.append(f"{s / 256:6.1f}")
    print(" ".join(row))
