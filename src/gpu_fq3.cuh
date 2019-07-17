#pragma once

#include "gpu_fq.cuh"

class GpuFq3
{
  GpuFq c0, c1, c2;
  GpuFq &non_residue;

public:
  __device__
  GpuFq3(const GpuFq &c0, const GpuFq &c1, const GpuFq &c2, GpuFq &non_residue) : c0(c0), c1(c1), c2(c2), non_residue(non_residue) {}

  __device__ __forceinline__ void save(fixnum &c0, fixnum &c1, fixnum &c2)
  {
    this->c0.save(c0);
    this->c1.save(c1);
    this->c2.save(c2);
  }

  __device__ __forceinline__ GpuFq3 operator*(const GpuFq3 &other) const
  {
    const GpuFq c0_c0 = this->c0 * other.c0;
    const GpuFq c1_c1 = this->c1 * other.c1;
    const GpuFq c2_c2 = this->c2 * other.c2;

    return GpuFq3(c0_c0 + this->non_residue * ((this->c1 + this->c2) * (other.c1 + other.c2) - c1_c1 - c2_c2),
                  (this->c0 + this->c1) * (other.c0 + other.c1) - c0_c0 - c1_c1 + this->non_residue * c2_c2,
                  (this->c0 + this->c2) * (other.c0 + other.c2) - c0_c0 + c1_c1 - c2_c2,
                  this->non_residue);
  }

  __device__ __forceinline__ GpuFq3 operator+(const GpuFq3 &other) const
  {
    return GpuFq3(this->c0 + other.c0, this->c1 + other.c1, this->c2 + other.c2, this->non_residue);
  }
};
