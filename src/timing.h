#pragma once
// Optional stage timing for the filter's frame path. Set the environment
// variable FFT3DCL_TIME to a file path and the plugin appends a table of
// accumulated wall-clock time per stage when a filter instance is destroyed.
// Everything here is a no-op (two predictable branches per stage) when the
// variable is not set, so it can stay in a release build.

#include <atomic>

namespace fft3dcl {

enum Stage {
  kFetch = 0,   // asking the source node for a frame
  kAlloc,       // allocating the output frame
  kChroma,      // copying the planes the filter does not process
  kLock,        // waiting for the instance mutex
  kUpload,      // host -> device transfer
  kFwd,         // pixel unpack + forward FFT launches
  kRun,         // temporal filter + inverse FFT + synthesis launches
  kReadback,    // pixel pack + device -> host transfer
  kPrune,       // spectra cache bookkeeping
  // GPU execution time per kernel, only filled in when FFT3DCL_TIME is set
  // (the queue is drained between kernels in that mode, which is slower but
  // shows where the device time actually goes).
  kGpuFwd,
  kGpuFil,
  kGpuInv,
  kGpuSyn,
  kStageCount
};

extern const char* const kStageNames[kStageCount];

extern std::atomic<long long> g_stageNs[kStageCount];
extern std::atomic<long long> g_stageCalls[kStageCount];

// True when FFT3DCL_TIME names a writable file. Read once, at create time.
bool timeEnabled();
void timeInit();
// Appends the accumulated table (process-wide totals) to the FFT3DCL_TIME file.
void timeDump(const char* tag);

// Adds one measurement without a clock read of its own (for spans that are
// already timed by the caller).
inline void addStage(Stage s, long long ns) {
  g_stageNs[s] += ns;
  g_stageCalls[s] += 1;
}

long long nowNs();

// RAII helper: StageTime t(kUpload); ... } records the span.
class StageTime {
public:
  explicit StageTime(Stage s) : s_(s) { t0_ = timeEnabled() ? nowNs() : 0; }
  ~StageTime() {
    if (t0_)
      addStage(s_, nowNs() - t0_);
  }

private:
  Stage s_;
  long long t0_;
};

} // namespace fft3dcl
