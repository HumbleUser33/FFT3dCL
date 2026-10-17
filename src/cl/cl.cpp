#include "cl.h"
#include <windows.h>
#include <cstring>

namespace fft3dcl {

bool ClApi::load() {
  if (ok_)
    return true;
  mod_ = LoadLibraryA("OpenCL.dll");
  if (!mod_)
    return false;
#define GET(field, name) field = (void*)GetProcAddress((HMODULE)mod_, name); if (!field) return false;
  GET(p_getPlatformIDs, "clGetPlatformIDs")
  GET(p_getDeviceIDs, "clGetDeviceIDs")
  GET(p_getDeviceInfo, "clGetDeviceInfo")
  GET(p_getPlatformInfo, "clGetPlatformInfo")
  GET(p_createContext, "clCreateContext")
  GET(p_createQueue, "clCreateCommandQueue")
  GET(p_createProgramWithSource, "clCreateProgramWithSource")
  GET(p_buildProgram, "clBuildProgram")
  GET(p_getProgramBuildInfo, "clGetProgramBuildInfo")
  GET(p_createKernel, "clCreateKernel")
  GET(p_createBuffer, "clCreateBuffer")
  GET(p_setKernelArg, "clSetKernelArg")
  GET(p_enqueueNDRange, "clEnqueueNDRangeKernel")
  GET(p_enqueueReadBuffer, "clEnqueueReadBuffer")
  GET(p_enqueueWriteBuffer, "clEnqueueWriteBuffer")
  GET(p_enqueueMapBuffer, "clEnqueueMapBuffer")
  GET(p_enqueueUnmap, "clEnqueueUnmapMemObject")
  GET(p_finish, "clFinish")
  GET(p_releaseKernel, "clReleaseKernel")
  GET(p_releaseProgram, "clReleaseProgram")
  GET(p_releaseMem, "clReleaseMemObject")
  GET(p_releaseQueue, "clReleaseCommandQueue")
  GET(p_releaseContext, "clReleaseContext")
#undef GET
  ok_ = true;
  return true;
}

// All entry points use the __stdcall calling convention on Windows.
typedef cl_int(CL_API_CALL* fn_platIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int(CL_API_CALL* fn_devIDs)(cl_platform_id, cl_bitfield, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int(CL_API_CALL* fn_devInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_int(CL_API_CALL* fn_platInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*);
typedef cl_context(CL_API_CALL* fn_ctx)(const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue(CL_API_CALL* fn_queue)(cl_context, cl_device_id, cl_uint, cl_int*);
typedef cl_program(CL_API_CALL* fn_prog)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int(CL_API_CALL* fn_build)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_int(CL_API_CALL* fn_binfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_kernel(CL_API_CALL* fn_kern)(cl_program, const char*, cl_int*);
typedef cl_mem(CL_API_CALL* fn_buf)(cl_context, cl_uint, size_t, void*, cl_int*);
typedef cl_int(CL_API_CALL* fn_karg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int(CL_API_CALL* fn_nd)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int(CL_API_CALL* fn_rd)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int(CL_API_CALL* fn_wr)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef void*(CL_API_CALL* fn_map)(cl_command_queue, cl_mem, cl_bool, cl_uint, size_t, size_t, cl_uint, const cl_event*, cl_event*, cl_int*);
typedef cl_int(CL_API_CALL* fn_unmap)(cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int(CL_API_CALL* fn_fin)(cl_command_queue);
typedef cl_int(CL_API_CALL* fn_rel)(void*);

cl_int ClApi::getPlatformIDs(cl_uint n, cl_platform_id* p, cl_uint* g) const {
  return ((fn_platIDs)p_getPlatformIDs)(n, p, g);
}
cl_int ClApi::getDeviceIDs(cl_platform_id p, cl_bitfield t, cl_uint n, cl_device_id* d, cl_uint* g) const {
  return ((fn_devIDs)p_getDeviceIDs)(p, t, n, d, g);
}
cl_int ClApi::getDeviceInfo(cl_device_id d, cl_uint n, size_t s, void* v, size_t* r) const {
  return ((fn_devInfo)p_getDeviceInfo)(d, n, s, v, r);
}
cl_int ClApi::getPlatformInfo(cl_platform_id p, cl_uint n, size_t s, void* v, size_t* r) const {
  return ((fn_platInfo)p_getPlatformInfo)(p, n, s, v, r);
}
cl_context ClApi::createContext(const cl_device_id* d, cl_uint n, cl_int* e) const {
  return ((fn_ctx)p_createContext)(nullptr, n, d, nullptr, nullptr, e);
}
cl_command_queue ClApi::createQueue(cl_context c, cl_device_id d, cl_uint props, cl_int* e) const {
  return ((fn_queue)p_createQueue)(c, d, props, e);
}
cl_program ClApi::createProgramWithSource(cl_context c, cl_uint n, const char** s, const size_t* l, cl_int* e) const {
  return ((fn_prog)p_createProgramWithSource)(c, n, s, l, e);
}
cl_int ClApi::buildProgram(cl_program p, cl_uint n, const cl_device_id* d, const char* o, void* cb, void* ud) const {
  return ((fn_build)p_buildProgram)(p, n, d, o, cb, ud);
}
cl_int ClApi::getProgramBuildInfo(cl_program p, cl_device_id d, cl_uint n, size_t s, void* v, size_t* r) const {
  return ((fn_binfo)p_getProgramBuildInfo)(p, d, n, s, v, r);
}
cl_kernel ClApi::createKernel(cl_program p, const char* n, cl_int* e) const {
  return ((fn_kern)p_createKernel)(p, n, e);
}
cl_mem ClApi::createBuffer(cl_context c, cl_uint f, size_t s, void* h, cl_int* e) const {
  return ((fn_buf)p_createBuffer)(c, f, s, h, e);
}
cl_int ClApi::setKernelArg(cl_kernel k, cl_uint i, size_t s, const void* v) const {
  return ((fn_karg)p_setKernelArg)(k, i, s, v);
}
cl_int ClApi::enqueueNDRange(cl_command_queue q, cl_kernel k, cl_uint dim, const size_t* go, const size_t* gs, const size_t* ls, cl_uint nw, const cl_event* we, cl_event* ev) const {
  return ((fn_nd)p_enqueueNDRange)(q, k, dim, go, gs, ls, nw, we, ev);
}
cl_int ClApi::enqueueReadBuffer(cl_command_queue q, cl_mem b, cl_bool bl, size_t o, size_t s, void* d, cl_uint nw, const cl_event* we, cl_event* ev) const {
  return ((fn_rd)p_enqueueReadBuffer)(q, b, bl, o, s, d, nw, we, ev);
}
cl_int ClApi::enqueueWriteBuffer(cl_command_queue q, cl_mem b, cl_bool bl, size_t o, size_t s, const void* d, cl_uint nw, const cl_event* we, cl_event* ev) const {
  return ((fn_wr)p_enqueueWriteBuffer)(q, b, bl, o, s, d, nw, we, ev);
}
cl_int ClApi::enqueueMapBuffer(cl_command_queue q, cl_mem b, cl_bool bl, cl_uint m, size_t o, size_t s, cl_uint nw, const cl_event* we, cl_event* ev) const {
  // returns the mapped pointer; err via ev not used here (see GpuRef usage)
  cl_int err = 0;
  void* p = ((fn_map)p_enqueueMapBuffer)(q, b, bl, m, o, s, nw, we, ev, &err);
  return err == CL_SUCCESS && p ? CL_SUCCESS : (err ? err : -1);
}
cl_int ClApi::enqueueUnmapMemObject(cl_command_queue q, cl_mem b, void* p, cl_uint nw, const cl_event* we, cl_event* ev) const {
  return ((fn_unmap)p_enqueueUnmap)(q, b, p, nw, we, ev);
}
cl_int ClApi::finish(cl_command_queue q) const { return ((fn_fin)p_finish)(q); }
cl_int ClApi::releaseKernel(cl_kernel k) const { return ((fn_rel)p_releaseKernel)(k); }
cl_int ClApi::releaseProgram(cl_program p) const { return ((fn_rel)p_releaseProgram)(p); }
cl_int ClApi::releaseMemObject(cl_mem m) const { return ((fn_rel)p_releaseMem)(m); }
cl_int ClApi::releaseQueue(cl_command_queue q) const { return ((fn_rel)p_releaseQueue)(q); }
cl_int ClApi::releaseContext(cl_context c) const { return ((fn_rel)p_releaseContext)(c); }

bool GpuContext::init(std::string* err) {
  if (!api.load()) {
    if (err) *err = "OpenCL.dll not found";
    return false;
  }
  cl_platform_id plats[8];
  cl_uint np = 0;
  if (api.getPlatformIDs(8, plats, &np) != CL_SUCCESS || np == 0) {
    if (err) *err = "no OpenCL platforms";
    return false;
  }
  for (cl_uint i = 0; i < np && !device; i++) {
    cl_device_id devs[8];
    cl_uint nd = 0;
    if (api.getDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 8, devs, &nd) != CL_SUCCESS)
      continue;
    for (cl_uint j = 0; j < nd; j++) {
      platform = plats[i];
      device = devs[j];
      break;
    }
  }
  if (!device) {
    if (err) *err = "no OpenCL GPU device found";
    return false;
  }
  char name[256] = {};
  api.getDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
  deviceName = name;
  api.getDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
  api.getDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, nullptr);
  cl_int e = 0;
  context = api.createContext(&device, 1, &e);
  if (!context || e != CL_SUCCESS) {
    if (err) *err = "clCreateContext failed";
    return false;
  }
  queue = api.createQueue(context, device, 0, &e);
  if (!queue || e != CL_SUCCESS) {
    if (err) *err = "clCreateCommandQueue failed";
    return false;
  }
  return true;
}

void GpuContext::shutdown() {
  if (queue) api.releaseQueue(queue);
  if (context) api.releaseContext(context);
  queue = nullptr;
  context = nullptr;
  device = nullptr;
}

} // namespace fft3dcl
