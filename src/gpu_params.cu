#include "gpu_constants.cuh"
#include "gpu_fq.cuh"
#include "gpu_params.cuh"

__constant__ HostParams host_params;
__device__ GpuParams gpu_params;

template <typename fixnum>
struct init_params_gpu
{
  __device__ void operator()(fixnum dummy)
  {
    gpu_params.set_mnt_mod(modnum(array_to_fixnum(host_params.get_mnt_mod())));
    gpu_params.set_mnt_non_residue(array_to_fixnum(host_params.get_mnt_non_residue()));
    gpu_params.set_mnt_coeff_a(array_to_fixnum(host_params.get_mnt_coeff_a()));
    gpu_params.set_mnt_coeff_a2(array_to_fixnum(host_params.get_mnt_coeff_a2_c0()), array_to_fixnum(host_params.get_mnt_coeff_a2_c1()));
  }

  __device__ fixnum array_to_fixnum(fixnum *arr)
  {
    return arr[fixnum::layout::laneIdx()];
  }
};

void init_params(HostParams &params)
{
  my_fixnum_array *dummy = my_fixnum_array::create(1);
  cudaMemcpyToSymbol(host_params, &params, sizeof(HostParams));
  my_fixnum_array::template map<init_params_gpu>(dummy);

  delete dummy;
}

__device__
    GpuParams &
    get_gpu_params()
{
  return gpu_params;
}