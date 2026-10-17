# Print a small luma patch from source, ref and port renders for visual diagnosis.
import subprocess, sys

FFMPEG = r"C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\env\avs\ffmpeg.exe"
W, H, N = 256, 192, 8
FRAME = W * H + (W // 2) * (H // 2) * 2


def decode(path):
    p = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                       capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"ffmpeg failed on {path}")
    return p.stdout


src = decode(sys.argv[1])
ref = decode(sys.argv[2])
port = decode(sys.argv[3])
frame = int(sys.argv[4]) if len(sys.argv) > 4 else 0
x0, y0, sz = (int(v) for v in (sys.argv[5:8] if len(sys.argv) > 7 else [0, 0, 16]))

print(f"frame {frame}, patch ({x0},{y0}) {sz}x{sz}  SRC / REF / PORT")
for y in range(y0, y0 + sz):
    rs = [src[frame * FRAME + y * W + x0 + x] for x in range(sz)]
    rr = [ref[frame * FRAME + y * W + x0 + x] for x in range(sz)]
    rp = [port[frame * FRAME + y * W + x0 + x] for x in range(sz)]
    print(" ".join(f"{a:3d}{b:3d}{c:3d}" for a, b, c in zip(rs, rr, rp)))
