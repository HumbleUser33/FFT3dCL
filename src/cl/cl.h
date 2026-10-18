#pragma once
// Minimal OpenCL 1.2 host wrapper: loads OpenCL.dll at runtime (ICD),
// picks a GPU device, creates context/queue, builds programs from source.
// Only the API subset the plugin needs is declared - no OpenCL headers
// required at build time, and no link dependency on an import library.

#include <string>
#include <vector>
#include <cstdint>

#ifndef CL_API_CALL
#define CL_API_CALL __stdcall
#endif
#ifndef CL_CALLBACK
#define CL_CALLBACK __stdcall
#endif

namespace fft3dcl {

// --- minimal OpenCL types/constants (values fixed by the spec) ---
typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned int cl_bool;
typedef long long cl_long;
typedef intptr_t cl_context_properties;
typedef unsigned long cl_ulong; // matches Windows x64 cl_ulong (unsigned __int64)
typedef cl_ulong cl_bitfield;
typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_context* cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_program* cl_program;
typedef struct _cl_kernel* cl_kernel;
typedef struct _cl_mem* cl_mem;
typedef struct _cl_event* cl_event;

#define CL_DEVICE_TYPE_GPU 4
#define CL_PLATFORM_NAME 0x0902
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1031
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1001

#define CL_SUCCESS 0
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_MEM_ALLOC_HOST_PTR (1 << 4)
#define CL_TRUE 1
#define CL_FALSE 0
#define CL_MAP_READ (1 << 0)
#define CL_MAP_WRITE (1 << 1)
#define CL_COMPLETE 0x0

class ClApi {
public:
  // Loads OpenCL.dll; returns false if unavailable.
  bool load();
  bool available() const { return ok_; }

  cl_int getPlatformIDs(cl_uint num, cl_platform_id* plats, cl_uint* got) const;
  cl_int getDeviceIDs(cl_platform_id p, cl_bitfield type, cl_uint num, cl_device_id* devs, cl_uint* got) const;
  cl_int getDeviceInfo(cl_device_id d, cl_uint name, size_t sz, void* val, size_t* rqn) const;
  cl_int getPlatformInfo(cl_platform_id p, cl_uint name, size_t sz, void* val, size_t* rqn) const;
  cl_context createContext(const cl_device_id* devs, cl_uint n, cl_int* err) const;
  cl_command_queue createQueue(cl_context ctx, cl_device_id d, cl_uint props, cl_int* err) const;
  cl_program createProgramWithSource(cl_context ctx, cl_uint n, const char** src, const size_t* len, cl_int* err) const;
  cl_int buildProgram(cl_program p, cl_uint n, const cl_device_id* devs, const char* options, void* cb, void* ud) const;
  cl_int getProgramBuildInfo(cl_program p, cl_device_id d, cl_uint name, size_t sz, void* val, size_t* rqn) const;
  cl_kernel createKernel(cl_program p, const char* name, cl_int* err) const;
  cl_mem createBuffer(cl_context ctx, cl_uint flags, size_t sz, void* host, cl_int* err) const;
  cl_int setKernelArg(cl_kernel k, cl_uint idx, size_t sz, const void* val) const;
  cl_int enqueueNDRange(cl_command_queue q, cl_kernel k, cl_uint dim, const size_t* goff, const size_t* gsize, const size_t* lsize, cl_uint nw, const cl_event* we, cl_event* ev) const;
  cl_int enqueueReadBuffer(cl_command_queue q, cl_mem buf, cl_bool block, size_t off, size_t sz, void* dst, cl_uint nw, const cl_event* we, cl_event* ev) const;
  cl_int enqueueWriteBuffer(cl_command_queue q, cl_mem buf, cl_bool block, size_t off, size_t sz, const void* src, cl_uint nw, const cl_event* we, cl_event* ev) const;
  void* enqueueMapBuffer(cl_command_queue q, cl_mem buf, cl_bool block, cl_uint map, size_t off, size_t sz, cl_uint nw, const cl_event* we, cl_event* ev, cl_int* err) const;
  cl_int enqueueUnmapMemObject(cl_command_queue q, cl_mem buf, void* ptr, cl_uint nw, const cl_event* we, cl_event* ev) const;
  cl_int flush(cl_command_queue q) const;
  cl_int getEventInfo(cl_event e, cl_uint name, size_t sz, void* val, size_t* rqn) const;
  cl_int setEventCallback(cl_event e, cl_uint type, void(CL_CALLBACK* cb)(cl_event, cl_int, void*),
                          void* ud) const;
  cl_int finish(cl_command_queue q) const;
  cl_int releaseKernel(cl_kernel k) const;
  cl_int releaseProgram(cl_program p) const;
  cl_int releaseMemObject(cl_mem m) const;
  cl_int releaseEvent(cl_event e) const;
  cl_int releaseQueue(cl_command_queue q) const;
  cl_int releaseContext(cl_context c) const;

private:
  bool ok_ = false;
  void* mod_ = nullptr;
  // function pointers
  void* p_getPlatformIDs = nullptr;
  void* p_getDeviceIDs = nullptr;
  void* p_getDeviceInfo = nullptr;
  void* p_getPlatformInfo = nullptr;
  void* p_createContext = nullptr;
  void* p_createQueue = nullptr;
  void* p_createProgramWithSource = nullptr;
  void* p_buildProgram = nullptr;
  void* p_getProgramBuildInfo = nullptr;
  void* p_createKernel = nullptr;
  void* p_createBuffer = nullptr;
  void* p_setKernelArg = nullptr;
  void* p_enqueueNDRange = nullptr;
  void* p_enqueueReadBuffer = nullptr;
  void* p_enqueueWriteBuffer = nullptr;
  void* p_enqueueMapBuffer = nullptr;
  void* p_enqueueUnmap = nullptr;
  void* p_flush = nullptr;
  void* p_getEventInfo = nullptr;
  void* p_setEventCallback = nullptr;
  void* p_finish = nullptr;
  void* p_releaseKernel = nullptr;
  void* p_releaseProgram = nullptr;
  void* p_releaseMem = nullptr;
  void* p_releaseEvent = nullptr;
  void* p_releaseQueue = nullptr;
  void* p_releaseContext = nullptr;
};

struct GpuContext {
  ClApi api;
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  std::string deviceName;
  size_t maxWorkGroupSize = 0;
  size_t localMemSize = 0;

  // Picks the first GPU device (prefers NVIDIA/AMD/Intel GPU over CPU).
  bool init(std::string* err);
  void shutdown();
};

} // namespace fft3dcl
