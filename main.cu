#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"

using namespace std;
using namespace cuFIXNUM;


const unsigned int bytes_per_elem = 128;
const unsigned int io_bytes_per_elem = 96;
const unsigned int libms_per_elem = bytes_per_elem / 8;

const uint8_t mnt4_modulus[bytes_per_elem] = {1, 128, 94, 36, 222, 99, 144, 94, 159, 17, 221, 44, 82, 84, 157, 227, 240, 37, 196, 154, 113, 16, 136, 99, 164, 84, 114, 118, 233, 204, 90, 104, 56, 126, 83, 203, 165, 13, 15, 184, 157, 5, 24, 242, 118, 231, 23, 177, 157, 247, 90, 161, 217, 36, 209, 153, 141, 237, 160, 232, 37, 185, 253, 7, 115, 216, 151, 108, 249, 232, 183, 94, 237, 175, 143, 91, 80, 151, 249, 183, 173, 205, 226, 238, 34, 144, 34, 16, 17, 196, 146, 45, 198, 196, 1, 0 };
const uint8_t mnt6_modulus[bytes_per_elem] = {1, 0, 0, 64, 226, 118, 7, 217, 79, 58, 161, 15, 23, 153, 160, 78, 151, 87, 0, 63, 188, 129, 195, 214, 164, 58, 153, 52, 118, 249, 223, 185, 54, 38, 33, 41, 148, 202, 235, 62, 155, 169, 89, 200, 40, 92, 108, 178, 157, 247, 90, 161, 217, 36, 209, 153, 141, 237, 160, 232, 37, 185, 253, 7, 115, 216, 151, 108, 249, 232, 183, 94, 237, 175, 143, 91, 80, 151, 249, 183, 173, 205, 226, 238, 34, 144, 34, 16, 17, 196, 146, 45, 198, 196, 1, 0};

const uint8_t mnt4_non_residue[bytes_per_elem] = {13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
const uint8_t mnt6_non_residue[bytes_per_elem] = {11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

typedef warp_fixnum<bytes_per_elem, u64_fixnum> fixnum;
typedef fixnum_array<fixnum> my_fixnum_array;
// redc may be worth trying over cios
typedef modnum_monty_redc<fixnum> modnum;

class GpuFq
{
public:
  fixnum data;
  modnum& mod;
public:

  __device__
  GpuFq(const fixnum &data, modnum& mod) : data(data), mod(mod) {}

  __device__ static GpuFq load(const fixnum &data, modnum& mod) 
  {
    fixnum result;
    mod.to_modnum(result, data);
    return GpuFq(result, mod);
  }

  __device__ __forceinline__ void save(fixnum& result) {
    this->mod.from_modnum(result, this->data);
  }

  __device__ __forceinline__ GpuFq operator*(const GpuFq &other) const
  {
    fixnum result;
    this->mod.mul(result, this->data, other.data);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ GpuFq operator+(const GpuFq &other) const
  {
    fixnum result;
    this->mod.add(result, this->data, other.data);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ GpuFq operator-(const GpuFq &other) const
  {
    fixnum result;
    this->mod.sub(result, this->data, other.data);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ GpuFq squared() const
  {
    fixnum result;
    this->mod.sqr(result, this->data);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ bool is_zero() const
  {
    return fixnum::is_zero(this->data);
  }

  __device__ __forceinline__ bool operator==(const GpuFq& other) const
  {
    return fixnum::cmp(this->data, other.data) == 0;
  }
};

class GpuFq2 {  
  GpuFq c0, c1;

  GpuFq& non_residue;
public:
  __device__
  GpuFq2(const GpuFq &c0, const GpuFq &c1, GpuFq& non_residue) : c0(c0), c1(c1), non_residue(non_residue) {}

  __device__ __forceinline__ void save(fixnum& c0, fixnum& c1) {
    this->c0.save(c0);
    this->c1.save(c1);
  }

  __device__ __forceinline__ GpuFq2 operator*(const GpuFq2 &other) const {
    GpuFq a0_b0 = this->c0 * other.c0;
    GpuFq a1_b1 = this->c1 * other.c1;

    GpuFq a0_plus_a1 = this->c0 + this->c1;
    GpuFq b0_plus_b1 = other.c0 + other.c1;

    GpuFq c = a0_plus_a1 * b0_plus_b1;

    return GpuFq2(a0_b0 + a1_b1 * this->non_residue, c - a0_b0 - a1_b1, this->non_residue);
  }

  __device__ __forceinline__ GpuFq2 operator+(const GpuFq2 &other) const {
    return GpuFq2(this->c0 + other.c0, this->c1 + other.c1, this->non_residue);
  }

  __device__ bool operator==(const GpuFq2 &other) const
  {
    return (this->c0 == other.c0 && this->c1 == other.c1);
  }
};

class GpuFq3 {
  GpuFq c0, c1, c2;
  GpuFq& non_residue;
public:
  __device__
  GpuFq3(const GpuFq &c0, const GpuFq &c1, const GpuFq &c2, GpuFq& non_residue) : c0(c0), c1(c1), c2(c2), non_residue(non_residue) {}

  __device__ __forceinline__ void save(fixnum& c0, fixnum& c1, fixnum& c2) {
    this->c0.save(c0);
    this->c1.save(c1);
    this->c2.save(c2);
  }

  __device__ __forceinline__ GpuFq3 operator*(const GpuFq3 &other) const {
    const GpuFq c0_c0 = this->c0 * other.c0;
    const GpuFq c1_c1 = this->c1 * other.c1;
    const GpuFq c2_c2 = this->c2 * other.c2;

    return GpuFq3(c0_c0 + this->non_residue * ((this->c1 + this->c2) * (other.c1 + other.c2) - c1_c1 - c2_c2),
                  (this->c0 + this->c1) * (other.c0 + other.c1) - c0_c0 - c1_c1 + this->non_residue * c2_c2,
                  (this->c0 + this->c2) *(other.c0 + other.c2) - c0_c0 + c1_c1 - c2_c2, 
                  this->non_residue);
  }

  __device__ __forceinline__ GpuFq3 operator+(const GpuFq3 &other) const {
    return GpuFq3(this->c0 + other.c0, this->c1 + other.c1, this->c2 + other.c2, this->non_residue);
  }
};



class CpuParams {
  fixnum mnt_mod[libms_per_elem];
  fixnum mnt_non_residue[libms_per_elem];
  fixnum mnt_coeff_a[libms_per_elem];

public:
  __host__
  void set_mnt_mod(fixnum* mod) {
    memcpy(this->mnt_mod, mod, bytes_per_elem);
  }

  __device__
  fixnum* get_mnt_mod() {
    return this->mnt_mod;
  }

  __host__
  void set_mnt_non_residue(fixnum* non_residue) {
    memcpy(this->mnt_non_residue, non_residue, bytes_per_elem);
  }

  __device__
  fixnum* get_mnt_non_residue() {
    return this->mnt_non_residue;
  }

  __host__
  void set_mnt_coeff_a(fixnum* coeff_a) {
    memcpy(this->mnt_coeff_a, coeff_a, bytes_per_elem);
  }

  __device__
  fixnum* get_mnt_coeff_a() {
    return this->mnt_coeff_a;
  }
};

class GpuParams {
  uint8_t mnt_mod[libms_per_elem * sizeof(modnum)];
  fixnum mnt_non_residue[libms_per_elem];
  fixnum mnt_coeff_a[libms_per_elem];
  
public:
  __device__
  modnum get_mnt_mod() {
    return ((modnum*)this->mnt_mod)[fixnum::layout::laneIdx()];
  }

  __device__
  void set_mnt_mod(modnum mod) {
    ((modnum*)this->mnt_mod)[fixnum::layout::laneIdx()] = mod;
  }

  __device__
  GpuFq get_mnt_non_residue() {
    fixnum non_residue = this->mnt_non_residue[fixnum::layout::laneIdx()];
    modnum mod = this->get_mnt_mod();
    return GpuFq(non_residue, mod); // saved as MM, no need to load
  }

  __device__
  void set_mnt_non_residue(fixnum non_residue) {
    modnum mod = this->get_mnt_mod();
    GpuFq fq = GpuFq::load(non_residue, mod);
    this->mnt_non_residue[fixnum::layout::laneIdx()] = fq.data; // save as MM
  }

  __device__
  GpuFq get_mnt_coeff_a() {
    fixnum coeff_a = this->mnt_coeff_a[fixnum::layout::laneIdx()];
    modnum mod = this->get_mnt_mod();
    return GpuFq(coeff_a, mod); // saved as MM, no need to load
  }

  __device__
  void set_mnt_coeff_a(fixnum coeff_a) {
    modnum mod = this->get_mnt_mod();
    GpuFq fq = GpuFq::load(coeff_a, mod);
    this->mnt_coeff_a[fixnum::layout::laneIdx()] = fq.data; // save as MM
  }
};

__constant__ CpuParams cpu_params;
__device__ GpuParams gpu_params;

template<typename fixnum>
struct init_params_gpu {
  __device__ void operator()(fixnum dummy) {
    gpu_params.set_mnt_mod(modnum(array_to_fixnum(cpu_params.get_mnt_mod())));
    gpu_params.set_mnt_non_residue(array_to_fixnum(cpu_params.get_mnt_non_residue()));
    gpu_params.set_mnt_coeff_a(array_to_fixnum(cpu_params.get_mnt_coeff_a()));
  }

  __device__ fixnum array_to_fixnum(fixnum* arr) {
    return arr[fixnum::layout::laneIdx()];
  }  
};

template<typename fixnum>
struct fq_mul_gpu {
  __device__ void operator()(fixnum &r, fixnum a, fixnum b) {
      modnum mod = gpu_params.get_mnt_mod();
      GpuFq fqA = GpuFq::load(a, mod);
      GpuFq fqB = GpuFq::load(b, mod);
      GpuFq fqS = fqA * fqB;

      fqS.save(r);
  }
};

template<typename fixnum>
struct fq2_mul_gpu {
  __device__ void operator()(fixnum &r0, fixnum &r1, fixnum a0, fixnum a1, fixnum b0, fixnum b1) {
      modnum mod = gpu_params.get_mnt_mod();
      GpuFq non_residue = gpu_params.get_mnt_non_residue();

      GpuFq2 fqA = GpuFq2(GpuFq::load(a0, mod), GpuFq::load(a1, mod), non_residue);
      GpuFq2 fqB = GpuFq2(GpuFq::load(b0, mod), GpuFq::load(b1, mod), non_residue);
      GpuFq2 fqS = fqA * fqB;
      fqS.save(r0, r1);
  }
};

template<typename fixnum>
struct fq3_mul_gpu {
  __device__ void operator()(fixnum &r0, fixnum &r1, fixnum &r2, fixnum a0, fixnum a1, fixnum a2, fixnum b0, fixnum b1, fixnum b2) {
      modnum mod = gpu_params.get_mnt_mod();
      GpuFq non_residue = gpu_params.get_mnt_non_residue();

      GpuFq3 fqA = GpuFq3(GpuFq::load(a0, mod), GpuFq::load(a1, mod), GpuFq::load(a2, mod), non_residue);
      GpuFq3 fqB = GpuFq3(GpuFq::load(b0, mod), GpuFq::load(b1, mod), GpuFq::load(b2, mod), non_residue);
      GpuFq3 fqS = fqA * fqB;
      fqS.save(r0, r1, r2);
  }
};


uint8_t* get_1D_fixnum_array(my_fixnum_array* res, int n) {
    uint8_t* local_results = new uint8_t[bytes_per_elem * n];
    int ret_nelts;
    res->retrieve_all(local_results, bytes_per_elem * n, &ret_nelts);
    return local_results;
}

uint8_t* get_2D_fixnum_array(my_fixnum_array* res0, my_fixnum_array* res1, int nelts) {
    int lrl = bytes_per_elem * nelts;
    uint8_t* local_results0 = new uint8_t[lrl]; 
    uint8_t* local_results1 = new uint8_t[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results0[i] = 0;
      local_results1[i] = 0;
    }

    res0->retrieve_all(local_results0, bytes_per_elem * nelts, &ret_nelts);
    res1->retrieve_all(local_results1, bytes_per_elem * nelts, &ret_nelts);

    uint8_t* local_results = new uint8_t[2 * lrl]; 

    for (int i = 0; i < nelts; i++) {
      mempcpy(local_results + 2 * i * bytes_per_elem, local_results0 + i * bytes_per_elem, bytes_per_elem);
      mempcpy(local_results + 2 * i * bytes_per_elem + bytes_per_elem, local_results1 + i * bytes_per_elem, bytes_per_elem);
    }

    delete local_results0;
    delete local_results1;
    return local_results;
}

uint8_t* get_3D_fixnum_array(my_fixnum_array* res0, my_fixnum_array* res1, my_fixnum_array* res2, int nelts) 
{
  int lrl = bytes_per_elem*nelts;
  uint8_t* local_results0 = new uint8_t[lrl]; 
  uint8_t* local_results1 = new uint8_t[lrl];
  uint8_t* local_results2 = new uint8_t[lrl];
  int ret_nelts;
  for (int i = 0; i < lrl; i++) {
    local_results0[i] = 0;
    local_results1[i] = 0;
    local_results2[i] = 0;
  }

  res0->retrieve_all(local_results0, bytes_per_elem * nelts, &ret_nelts);
  res1->retrieve_all(local_results1, bytes_per_elem * nelts, &ret_nelts);
  res2->retrieve_all(local_results2, bytes_per_elem * nelts, &ret_nelts);

  uint8_t* local_results = new uint8_t[3 * lrl]; 

  for (int i = 0; i < nelts; i++) {
    mempcpy(local_results + 3 * i * bytes_per_elem, local_results0 + i * bytes_per_elem, bytes_per_elem);
    mempcpy(local_results + 3 * i * bytes_per_elem + bytes_per_elem, local_results1 + i * bytes_per_elem, bytes_per_elem);
    mempcpy(local_results + 3 * i * bytes_per_elem + 2 * bytes_per_elem, local_results2 + i * bytes_per_elem, bytes_per_elem);
  }

  delete[] local_results0;
  delete[] local_results1;
  delete[] local_results2;

  return local_results;
}

void init_params(CpuParams& params) {
  my_fixnum_array* dummy = my_fixnum_array::create(1);
  cudaMemcpyToSymbol(cpu_params, &params, sizeof(CpuParams));
  my_fixnum_array::template map<init_params_gpu>(dummy);

  delete dummy;
}

uint8_t* fq_mul(uint8_t* a, uint8_t* b, int nelts) {
    my_fixnum_array *in_a = my_fixnum_array::create(a, bytes_per_elem * nelts, bytes_per_elem);
    my_fixnum_array *in_b = my_fixnum_array::create(b, bytes_per_elem * nelts, bytes_per_elem);
    my_fixnum_array *res = my_fixnum_array::create(nelts);

    my_fixnum_array::template map<fq_mul_gpu>(res, in_a, in_b);
    uint8_t* v_res = get_1D_fixnum_array(res, nelts);

    delete in_a;
    delete in_b;
    delete res;

    return v_res;
}

uint8_t* fq2_mul(uint8_t* a, uint8_t* b, int nelts) {
  uint8_t *input_a0 = new uint8_t[bytes_per_elem * nelts];
  uint8_t *input_a1 = new uint8_t[bytes_per_elem * nelts];

  for (int i = 0; i < nelts; i++) {
    mempcpy(input_a0 + i * bytes_per_elem, a + 2 * i * bytes_per_elem, bytes_per_elem);
    mempcpy(input_a1 + i * bytes_per_elem, a + 2 * i * bytes_per_elem + bytes_per_elem, bytes_per_elem);
  }

  uint8_t *input_b0 = new uint8_t[bytes_per_elem * nelts];
  uint8_t *input_b1 = new uint8_t[bytes_per_elem * nelts];

  for (int i = 0; i < nelts; i++) {
    mempcpy(input_b0 + i * bytes_per_elem, b + 2 * i * bytes_per_elem, bytes_per_elem);
    mempcpy(input_b1 + i * bytes_per_elem, b + 2 * i * bytes_per_elem + bytes_per_elem, bytes_per_elem);
  }

  my_fixnum_array *in_a0 = my_fixnum_array::create(input_a0, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_a1 = my_fixnum_array::create(input_a1, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_b0 = my_fixnum_array::create(input_b0, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_b1 = my_fixnum_array::create(input_b1, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *res0 = my_fixnum_array::create(nelts);
  my_fixnum_array *res1 = my_fixnum_array::create(nelts);

  my_fixnum_array::template map<fq2_mul_gpu>(res0, res1, in_a0, in_a1, in_b0, in_b1);

  uint8_t* v_res = get_2D_fixnum_array(res0, res1, nelts);

  delete in_a0;
  delete in_a1;
  delete in_b0;
  delete in_b1;
  delete res0;
  delete res1;
  delete[] input_a0;
  delete[] input_a1;
  delete[] input_b0;
  delete[] input_b1;

  return v_res;
}

uint8_t* fq3_mul(uint8_t* a, uint8_t* b, int nelts) {
  uint8_t *input_a0 = new uint8_t[bytes_per_elem * nelts];
  uint8_t *input_a1 = new uint8_t[bytes_per_elem * nelts];
  uint8_t *input_a2 = new uint8_t[bytes_per_elem * nelts];

  for (int i = 0; i < nelts; i++) {
    mempcpy(input_a0 + i * bytes_per_elem, a + 3 * i * bytes_per_elem, bytes_per_elem);
    mempcpy(input_a1 + i * bytes_per_elem, a + 3 * i * bytes_per_elem + bytes_per_elem, bytes_per_elem);
    mempcpy(input_a2 + i * bytes_per_elem, a + 3 * i * bytes_per_elem + 2 * bytes_per_elem, bytes_per_elem);
  }

  uint8_t *input_b0 = new uint8_t[bytes_per_elem * nelts];
  uint8_t *input_b1 = new uint8_t[bytes_per_elem * nelts];
  uint8_t *input_b2 = new uint8_t[bytes_per_elem * nelts];

  for (int i = 0; i < nelts; i++) {
    mempcpy(input_b0 + i * bytes_per_elem, b + 3 * i * bytes_per_elem, bytes_per_elem);
    mempcpy(input_b1 + i * bytes_per_elem, b + 3 * i * bytes_per_elem + bytes_per_elem, bytes_per_elem);
    mempcpy(input_b2 + i * bytes_per_elem, b + 3 * i * bytes_per_elem + 2 * bytes_per_elem, bytes_per_elem);
  }

  my_fixnum_array *in_a0 = my_fixnum_array::create(input_a0, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_a1 = my_fixnum_array::create(input_a1, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_a2 = my_fixnum_array::create(input_a2, bytes_per_elem * nelts, bytes_per_elem);

  my_fixnum_array *in_b0 = my_fixnum_array::create(input_b0, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_b1 = my_fixnum_array::create(input_b1, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_b2 = my_fixnum_array::create(input_b2, bytes_per_elem * nelts, bytes_per_elem);
  
  my_fixnum_array *res0 = my_fixnum_array::create(nelts);
  my_fixnum_array *res1 = my_fixnum_array::create(nelts);
  my_fixnum_array *res2 = my_fixnum_array::create(nelts);

  my_fixnum_array::template map<fq3_mul_gpu>(res0, res1, res2, in_a0, in_a1, in_a2, in_b0, in_b1, in_b2);

  uint8_t* v_res = get_3D_fixnum_array(res0, res1, res2, nelts);

  delete in_a0;
  delete in_a1;
  delete in_a2;
  delete in_b0;
  delete in_b1;
  delete in_b2;
  delete res0;
  delete res1;
  delete res2;
  delete[] input_a0;
  delete[] input_a1;
  delete[] input_a2;
  delete[] input_b0;
  delete[] input_b1;
  delete[] input_b2;
  
  return v_res;
}

void read_mnt_fq(uint8_t* dest, FILE* inputs) {
  fread((void*)(dest), io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
}

void read_mnt_fq2(uint8_t* dest, FILE* inputs) {
  read_mnt_fq(dest, inputs);
  read_mnt_fq(dest + bytes_per_elem, inputs);
}

void read_mnt_fq3(uint8_t* dest, FILE* inputs) {
  read_mnt_fq(dest, inputs);
  read_mnt_fq(dest + bytes_per_elem, inputs);
  read_mnt_fq(dest + 2 * bytes_per_elem, inputs);
}

void write_mnt_fq(uint8_t* fq, FILE* outputs) {
  fwrite((void *) fq, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

void write_mnt_fq2(uint8_t* src, FILE* outputs) {
  write_mnt_fq(src, outputs);
  write_mnt_fq(src + bytes_per_elem, outputs);
}

void write_mnt_fq3(uint8_t* src, FILE* outputs) {
  write_mnt_fq(src, outputs);
  write_mnt_fq(src + bytes_per_elem, outputs);
  write_mnt_fq(src + 2 * bytes_per_elem, outputs);
}

void stage_0(FILE* inputs, FILE* outputs) {
  size_t n;

  while (true) {
    size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
    if (elts_read == 0) { break; }

    uint8_t* x0 = new uint8_t[n * bytes_per_elem];
    memset(x0, 0, n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq(x0 + i * bytes_per_elem, inputs);
    }

    uint8_t* x1 = new uint8_t[n * bytes_per_elem];
    memset(x1, 0, n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq(x1 + i * bytes_per_elem, inputs);
    }

    CpuParams params;
    params.set_mnt_mod((fixnum*)mnt4_modulus);
    init_params(params);
    
    uint8_t* res_x = fq_mul(x0, x1, n);

    for (size_t i = 0; i < n; ++i) {
      write_mnt_fq(res_x + i * bytes_per_elem, outputs);
    }

    delete []x0;
    delete []x1;
    delete []res_x;

    uint8_t* y0 = new uint8_t[n * bytes_per_elem];
    memset(y0, 0, n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq(y0 + i * bytes_per_elem, inputs);
    }

    uint8_t* y1 = new uint8_t[n * bytes_per_elem];
    memset(y1, 0, n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq(y1 + i * bytes_per_elem, inputs);
    }

    params.set_mnt_mod((fixnum*)mnt6_modulus);
    init_params(params);

    uint8_t* res_y = fq_mul(y0, y1, n);

    for (size_t i = 0; i < n; ++i) {
      write_mnt_fq(res_y + i * bytes_per_elem, outputs);
    }

    delete []y0;
    delete []y1;
    delete []res_y;
  }
}

void stage_1(FILE* inputs, FILE* outputs) {
  size_t n;

  while (true) {
    size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
    if (elts_read == 0) { break; }

    uint8_t* x0 = new uint8_t[2 * n * bytes_per_elem];
    memset(x0, 0, 2 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq2(x0 + 2 * i * bytes_per_elem, inputs);
    }

    uint8_t* x1 = new uint8_t[2 * n * bytes_per_elem];
    memset(x1, 0, 2 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq2(x1 + 2 * i * bytes_per_elem, inputs);
    }

    CpuParams params;
    params.set_mnt_mod((fixnum*)mnt4_modulus);
    params.set_mnt_non_residue((fixnum*)mnt4_non_residue);
    init_params(params);

    uint8_t* res_x = fq2_mul(x0, x1, n);

    for (size_t i = 0; i < n; ++i) {
      write_mnt_fq2(res_x + 2 * i * bytes_per_elem, outputs);
    }

    delete[] x0;
    delete[] x1;
    delete[] res_x;
  }
}

void stage_2(FILE* inputs, FILE* outputs) {
  size_t n;

  while (true) {
    size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
    if (elts_read == 0) { break; }

    uint8_t* x0 = new uint8_t[3 * n * bytes_per_elem];
    memset(x0, 0, 3 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq3(x0 + 3 * i * bytes_per_elem, inputs);
    }

    uint8_t* x1 = new uint8_t[3 * n * bytes_per_elem];
    memset(x1, 0, 3 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i) {
      read_mnt_fq3(x1 + 3 * i * bytes_per_elem, inputs);
    }

    CpuParams params;
    params.set_mnt_mod((fixnum*)mnt6_modulus);
    params.set_mnt_non_residue((fixnum*)mnt6_non_residue);
    init_params(params);

    uint8_t* res_x = fq3_mul(x0, x1, n);

    for (size_t i = 0; i < n; ++i) {
      write_mnt_fq3(res_x + 3 * i * bytes_per_elem, outputs);
    }

    delete[] x0;
    delete[] x1;
    delete[] res_x;
  }
}

void stage_3(FILE* inputs, FILE* outputs) {
}

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);

  bool is_stage_0 = strcmp(argv[1], "compute-stage-0") == 0;
  bool is_stage_1 = strcmp(argv[1], "compute-stage-1") == 0;
  bool is_stage_2 = strcmp(argv[1], "compute-stage-2") == 0;
  bool is_stage_3 = strcmp(argv[1], "compute-stage-3") == 0;

  FILE* inputs = fopen(argv[2], "r");
  FILE* outputs = fopen(argv[3], "w");

  if (is_stage_0) {
    stage_0(inputs, outputs);
  } else if (is_stage_1) {
    stage_1(inputs, outputs);
  } else if (is_stage_2) {
    stage_2(inputs, outputs);
  } else if (is_stage_3) {
    stage_3(inputs, outputs);
  }
  

  return 0;
}

