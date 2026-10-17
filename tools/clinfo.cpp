// Minimal OpenCL ICD probe: loads OpenCL.dll dynamically and lists
// platforms/devices. Validates the runtime-loading approach for the plugin.
#include <windows.h>
#include <stdio.h>

typedef int cl_int;
typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_device_id* cl_device_id;
#ifndef CL_API_CALL
#define CL_API_CALL __stdcall
#endif
typedef cl_int(CL_API_CALL* clGetPlatformIDs_t)(cl_int, cl_platform_id*, cl_int*);
typedef cl_int(CL_API_CALL* clGetDeviceIDs_t)(cl_platform_id, cl_int, cl_int, cl_device_id*, cl_int*);
typedef cl_int(CL_API_CALL* clGetDeviceInfo_t)(cl_device_id, cl_int, size_t, void*, size_t*);
typedef cl_int(CL_API_CALL* clGetPlatformInfo_t)(cl_platform_id, cl_int, size_t, void*, size_t*);

#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_PLATFORM_NAME 0x0902
#define CL_PLATFORM_VERSION 0x0911
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_LOCAL_MEM_SIZE 0x1027
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_DOUBLE_FP_CONFIG 0x1032

int main() {
  HMODULE mod = LoadLibraryA("OpenCL.dll");
  if (!mod) { printf("LoadLibrary OpenCL.dll failed\n"); return 1; }
  auto getPlatforms = (clGetPlatformIDs_t)GetProcAddress(mod, "clGetPlatformIDs");
  auto getDevices = (clGetDeviceIDs_t)GetProcAddress(mod, "clGetDeviceIDs");
  auto getDevInfo = (clGetDeviceInfo_t)GetProcAddress(mod, "clGetDeviceInfo");
  auto getPlatInfo = (clGetPlatformInfo_t)GetProcAddress(mod, "clGetPlatformInfo");
  if (!getPlatforms || !getDevices || !getDevInfo || !getPlatInfo) {
    printf("missing entry points\n"); return 1;
  }
  cl_platform_id plats[8];
  int np = 0;
  cl_int err = getPlatforms(8, plats, &np);
  printf("clGetPlatformIDs -> %d, platforms=%d\n", err, np);
  for (int i = 0; i < np; i++) {
    char name[256] = {}, ver[256] = {};
    getPlatInfo(plats[i], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
    getPlatInfo(plats[i], CL_PLATFORM_VERSION, sizeof(ver), ver, nullptr);
    printf("platform %d: %s [%s]\n", i, name, ver);
    cl_device_id devs[8];
    int nd = 0;
    err = getDevices(plats[i], CL_DEVICE_TYPE_ALL, 8, devs, &nd);
    for (int j = 0; j < nd; j++) {
      char dn[256] = {}, dv[256] = {};
      getDevInfo(devs[j], CL_DEVICE_NAME, sizeof(dn), dn, nullptr);
      getDevInfo(devs[j], CL_DEVICE_VERSION, sizeof(dv), dv, nullptr);
      size_t lmem = 0, wgs = 0;
      getDevInfo(devs[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
      getDevInfo(devs[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wgs), &wgs, nullptr);
      printf("  device %d: %s [%s] localmem=%zuKB maxwg=%zu\n", j, dn, dv, lmem / 1024, wgs);
    }
  }
  return 0;
}
