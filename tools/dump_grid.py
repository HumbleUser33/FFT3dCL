# Print diff grids (ref-src, port-src) for one frame's luma.
import subprocess, sys

FFMPEG = r"C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\env\avs\ffmpeg.exe"
W, H = 256, 192


def decode(path):
    p = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                       capture_output=True)
    return p.stdout


src = decode(sys.argv[1]); ref = decode(sys.argv[2]); port = decode(sys.argv[3])
frame = int(sys.argv[4]); x0 = int(sys.argv[5]); y0 = int(sys.argv[6]); sz = int(sys.argv[7])
base = frame * W * H * 3 // 2

def grid(data, label):
    print(label)
    for y in range(y0, y0 + sz):
        print(" ".join(f"{data[base + y * W + x0 + x]:4d}" for x in range(sz)))

grid(src, f"SRC frame{frame} @({x0},{y0})")
grid(ref, "REF")
grid(port, "PORT")
