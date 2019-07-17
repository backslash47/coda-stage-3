#include "params.hpp"

#include "gpu_constants.cuh"
#include "gpu_params.cuh"

void init_params_mnt4()
{
  HostParams params;
  params.set_mnt_mod((fixnum *)mnt4_modulus);
  params.set_mnt_non_residue((fixnum *)mnt4_non_residue);
  init_params(params);
}

void init_params_mnt6()
{
  HostParams params;
  params.set_mnt_mod((fixnum *)mnt6_modulus);
  params.set_mnt_non_residue((fixnum *)mnt6_non_residue);
  init_params(params);
}
