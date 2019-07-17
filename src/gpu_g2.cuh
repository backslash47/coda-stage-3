#pragma once

#include "gpu_fq.cuh"
#include "gpu_fq2.cuh"

class GpuG2
{
  GpuFq2 X, Y, Z;
  GpuFq2 &coeff_a;

public:
  __device__
  GpuG2(const GpuFq2 &X, const GpuFq2 &Y, const GpuFq2 &Z, GpuFq2 &coeff_a) : X(X), Y(Y), Z(Z), coeff_a(coeff_a) {}

  __device__ __forceinline__ void save(fixnum &xc0, fixnum &xc1, fixnum &yc0, fixnum &yc1, fixnum &zc0, fixnum &zc1)
  {
    this->X.save(xc0, xc1);
    this->Y.save(yc0, yc1);
    this->Z.save(zc0, zc1);
  }

  __device__ __forceinline__ bool is_zero() const
  {
    return this->X.is_zero() && this->Z.is_zero();
  }

  __device__
      GpuG2
      operator+(const GpuG2 &other) const
  {
    // handle special cases having to do with O
    if (this->is_zero())
    {
      return other;
    }

    if (other.is_zero())
    {
      return *this;
    }

    const GpuFq2 X1Z2 = this->X * other.Z; // X1Z2 = X1*Z2
    const GpuFq2 X2Z1 = this->Z * other.X; // X2Z1 = X2*Z1

    // (used both in add and double checks)

    const GpuFq2 Y1Z2 = this->Y * other.Z; // Y1Z2 = Y1*Z2
    const GpuFq2 Y2Z1 = this->Z * other.Y; // Y2Z1 = Y2*Z1

    if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
    {
      // perform dbl case
      const GpuFq2 XX = this->X.squared();                  // XX  = X1^2
      const GpuFq2 ZZ = this->Z.squared();                  // ZZ  = Z1^2
      const GpuFq2 w = this->coeff_a * ZZ + (XX + XX + XX); // w   = a*ZZ + 3*XX
      const GpuFq2 Y1Z1 = this->Y * this->Z;
      const GpuFq2 s = Y1Z1 + Y1Z1;                       // s   = 2*Y1*Z1
      const GpuFq2 ss = s.squared();                      // ss  = s^2
      const GpuFq2 sss = s * ss;                          // sss = s*ss
      const GpuFq2 R = this->Y * s;                       // R   = Y1*s
      const GpuFq2 RR = R.squared();                      // RR  = R^2
      const GpuFq2 B = (this->X + R).squared() - XX - RR; // B   = (X1+R)^2 - XX - RR
      const GpuFq2 h = w.squared() - (B + B);             // h   = w^2 - 2*B
      const GpuFq2 X3 = h * s;                            // X3  = h*s
      const GpuFq2 Y3 = w * (B - h) - (RR + RR);          // Y3  = w*(B-h) - 2*RR
      const GpuFq2 Z3 = sss;                              // Z3  = sss

      return GpuG2(X3, Y3, Z3, this->coeff_a);
    }

    // if we have arrived here we are in the add case
    const GpuFq2 Z1Z2 = this->Z * other.Z;      // Z1Z2 = Z1*Z2
    const GpuFq2 u = Y2Z1 - Y1Z2;               // u    = Y2*Z1-Y1Z2
    const GpuFq2 uu = u.squared();              // uu   = u^2
    const GpuFq2 v = X2Z1 - X1Z2;               // v    = X2*Z1-X1Z2
    const GpuFq2 vv = v.squared();              // vv   = v^2
    const GpuFq2 vvv = v * vv;                  // vvv  = v*vv
    const GpuFq2 R = vv * X1Z2;                 // R    = vv*X1Z2
    const GpuFq2 A = uu * Z1Z2 - (vvv + R + R); // A    = uu*Z1Z2 - vvv - 2*R
    const GpuFq2 X3 = v * A;                    // X3   = v*A
    const GpuFq2 Y3 = u * (R - A) - vvv * Y1Z2; // Y3   = u*(R-A) - vvv*Y1Z2
    const GpuFq2 Z3 = vvv * Z1Z2;               // Z3   = vvv*Z1Z2

    return GpuG2(X3, Y3, Z3, this->coeff_a);
  }
};
