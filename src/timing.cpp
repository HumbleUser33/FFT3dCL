#include "timing.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace fft3dcl {

const char* const kStageNames[kStageCount] = {
    "fetch source frame",     "allocate output frame", "copy untouched planes",
    "wait for instance lock", "upload to device",      "unpack + forward FFT",
    "filter+inverse+synth",   "pack + readback",       "prune spectra cache",
    "GPU: forward FFT",       "GPU: temporal filter",  "GPU: inverse FFT",
    "GPU: synthesis"};

std::atomic<long long> g_stageNs[kStageCount];
std::atomic<long long> g_stageCalls[kStageCount];

namespace {
bool g_on = false;
std::string g_path;
std::once_flag g_once;
} // namespace

long long nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool timeEnabled() { return g_on; }

void timeInit() {
  std::call_once(g_once, [] {
    const char* p = getenv("FFT3DCL_TIME");
    if (p && *p) {
      g_path = p;
      g_on = true;
    }
  });
}

void timeDump(const char* tag) {
  if (!g_on)
    return;
  FILE* f = fopen(g_path.c_str(), "a");
  if (!f)
    return;
  fprintf(f, "== %s ==\n", tag);
  long long total = 0;
  for (int i = 0; i < kStageCount; i++) {
    long long ns = g_stageNs[i].load();
    long long calls = g_stageCalls[i].load();
    total += ns;
    fprintf(f, "  %-24s calls=%8lld  %8.3f s  %7.2f ms/call\n", kStageNames[i], calls, ns / 1e9,
            calls ? ns / 1e6 / (double)calls : 0.0);
  }
  fprintf(f, "  %-24s %20.3f s\n", "total timed", total / 1e9);
  fclose(f);
}

} // namespace fft3dcl
