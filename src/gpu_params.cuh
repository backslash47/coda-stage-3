#pragma once

#include "gpu_constants.cuh"
#include "gpu_fq.cuh"

class GpuParams
{
  uint8_t mnt_mod[libms_per_elem * sizeof(modnum)];
  fixnum mnt_non_residue[libms_per_elem];
  fixnum mnt_coeff_a[libms_per_elem];

public:
  __device__
      modnum
      get_mnt_mod()
  {
    return ((modnum *)this->mnt_mod)[fixnum::layout::laneIdx()];
  }

  __device__ void set_mnt_mod(modnum mod)
  {
    ((modnum *)this->mnt_mod)[fixnum::layout::laneIdx()] = mod;
  }

  __device__
      GpuFq
      get_mnt_non_residue()
  {
    fixnum non_residue = this->mnt_non_residue[fixnum::layout::laneIdx()];
    modnum mod = this->get_mnt_mod();
    return GpuFq(non_residue, mod); // saved as MM, no need to load
  }

  __device__ void set_mnt_non_residue(fixnum non_residue)
  {
    modnum mod = this->get_mnt_mod();
    GpuFq fq = GpuFq::load(non_residue, mod);
    this->mnt_non_residue[fixnum::layout::laneIdx()] = fq.data; // save as MM
  }

  __device__
      GpuFq
      get_mnt_coeff_a()
  {
    fixnum coeff_a = this->mnt_coeff_a[fixnum::layout::laneIdx()];
    modnum mod = this->get_mnt_mod();
    return GpuFq(coeff_a, mod); // saved as MM, no need to load
  }

  __device__ void set_mnt_coeff_a(fixnum coeff_a)
  {
    modnum mod = this->get_mnt_mod();
    GpuFq fq = GpuFq::load(coeff_a, mod);
    this->mnt_coeff_a[fixnum::layout::laneIdx()] = fq.data; // save as MM
  }
};

class HostParams
{
  fixnum mnt_mod[libms_per_elem];
  fixnum mnt_non_residue[libms_per_elem];
  fixnum mnt_coeff_a[libms_per_elem];

public:
  __host__ void set_mnt_mod(fixnum *mod)
  {
    memcpy(this->mnt_mod, mod, bytes_per_elem);
  }

  __device__
      fixnum *
      get_mnt_mod()
  {
    return this->mnt_mod;
  }

  __host__ void set_mnt_non_residue(fixnum *non_residue)
  {
    memcpy(this->mnt_non_residue, non_residue, bytes_per_elem);
  }

  __device__
      fixnum *
      get_mnt_non_residue()
  {
    return this->mnt_non_residue;
  }

  __host__ void set_mnt_coeff_a(fixnum *coeff_a)
  {
    memcpy(this->mnt_coeff_a, coeff_a, bytes_per_elem);
  }

  __device__
      fixnum *
      get_mnt_coeff_a()
  {
    return this->mnt_coeff_a;
  }
};

void init_params(HostParams &params);
__device__ GpuParams &get_gpu_params();