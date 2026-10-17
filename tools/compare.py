# Pixel-by-pixel comparison of two lossless renders (same clip geometry).
# Usage: python compare.py ref.mkv port.mkv [width height frames]
import subprocess
import sys
import collections

FFMPEG = r"C:\Users\Bruno\Desktop\OpenCode\FFT3DGPU VS\env\avs\ffmpeg.exe"


def decode(path, w, h, n, fmt):
    bps = 2 if "16le" in fmt or "10le" in fmt or "12le" in fmt else 1
    p = subprocess.run(
        [FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", fmt, "-"],
        capture_output=True,
    )
    if p.returncode != 0:
        raise SystemExit(f"ffmpeg failed on {path}: {p.stderr.decode(errors='replace')}")
    frame = (w * h + (w // 2) * (h // 2) * 2) * bps
    data = p.stdout
    if len(data) < frame * n:
        raise SystemExit(f"{path}: got {len(data)} bytes, expected >= {frame * n}")
    return data[: frame * n], frame, bps


def main():
    ref, port = sys.argv[1], sys.argv[2]
    if len(sys.argv) >= 6:
        w, h, n = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    else:
        w, h, n = 256, 192, 8
    fmt = sys.argv[6] if len(sys.argv) > 6 else "yuv420p"
    a, frame, bps = decode(ref, w, h, n, fmt)
    b, _, _ = decode(port, w, h, n, fmt)

    if bps == 1:
        sa, sb = a, b
    else:
        import array
        sa = array.array("H", a)
        sb = array.array("H", b)

    hist = collections.Counter()
    worst = (0, 0)  # diff, sample index
    for i in range(min(len(sa), len(sb))):
        d = abs(sa[i] - sb[i])
        hist[d] += 1
        if d > worst[0]:
            worst = (d, i)

    total = min(len(sa), len(sb))
    exact = hist[0]
    ysize = w * h
    csize = (w // 2) * (h // 2)
    print(f"frames={n} {w}x{h} {fmt}")
    print(f"identical samples: {exact}/{total} ({100.0 * exact / total:.2f}%)")
    for d in sorted(hist):
        if d > 0:
            print(f"  diff {d}: {hist[d]} px ({100.0 * hist[d] / total:.4f}%)")
        if d >= 64:
            break
    fi, off = divmod(worst[1], frame // bps)
    plane = "Y" if off < ysize else ("U" if off < ysize + csize else "V")
    print(f"max diff: {worst[0]} at frame {fi}, plane {plane}")


if __name__ == "__main__":
    main()
