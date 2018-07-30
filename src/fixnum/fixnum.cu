#pragma once

#include <math.h>
#include <type_traits>

#include "util/primitives.cu"
#include "slot_layout.cu"

/*
 * This is an archetypal implementation of a fixnum instruction
 * set. It defines the de facto interface for such implementations.
 *
 * All methods are defined for the device. It is someone else's
 * problem to get the data onto the device.
 */
template< int FIXNUM_BYTES_, typename word_tp_ = uint32_t >
class default_fixnum_impl {
    static_assert(FIXNUM_BYTES_ > 0,
            "Fixnum bytes must be positive.");
    static_assert(FIXNUM_BYTES_ % sizeof(word_tp_) == 0,
            "Fixnum word size must divide fixnum bytes.");
    static_assert(std::is_integral< word_tp_ >::value,
            "word_tp must be integral.");

public:
    typedef word_tp_ word_tp;
    static constexpr int WORD_BITS = 8 * sizeof(word_tp_);
    static constexpr int FIXNUM_BYTES = FIXNUM_BYTES_;
    static constexpr int FIXNUM_BITS = 8 * FIXNUM_BYTES;
    static constexpr int SLOT_WIDTH = FIXNUM_BYTES_ / sizeof(word_tp_);
    // FIXME: slot_layout should not be exposed by this interface.
    typedef slot_layout< SLOT_WIDTH > slot_layout;
    typedef word_tp fixnum;

    /***************************
     * Representation functions.
     */

    /*
     * Set r using bytes, interpreting bytes as a base-256 unsigned
     * integer. Return the number of bytes used. If nbytes >
     * FIXNUM_BYTES, then the last nbytes - FIXNUM_BYTES are ignored.
     *
     * NB: Normally we would expect from_bytes to be exclusively a
     * device function, but it's the same for the host, so we leave it
     * in.
     */
    __host__ __device__ static int from_bytes(fixnum *r, const uint8_t *bytes, int nbytes) {
        uint8_t *s = reinterpret_cast< uint8_t * >(r);
        int n = min(nbytes, FIXNUM_BYTES);
        memcpy(s, bytes, n);
        memset(s + n, 0, FIXNUM_BYTES - n);
        return n;
    }

    /*
     * Set bytes using r, converting r to a base-256 unsigned
     * integer. Return the number of bytes written. If nbytes <
     * FIXNUM_BYTES, then the last FIXNUM_BYTES - nbytes are ignored.
     *
     * NB: Normally we would expect from_bytes to be exclusively a
     * device function, but it's the same for the host, so we leave it
     * in.
     */
    __host__ __device__ static int to_bytes(uint8_t *bytes, int nbytes, const fixnum *r) {
        int n = min(nbytes, FIXNUM_BYTES);
        memcpy(bytes, r, n);
        return n;
    }

    /*
     * load/set the value from ptr corresponding to this thread (lane) in
     * slot number idx.
     */
    __device__ static fixnum load(const fixnum *ptr, int idx = 0) {
        int off = idx * slot_layout::WIDTH + slot_layout::laneIdx();
        return ptr[off];
    }

    __device__ static fixnum &load(fixnum *ptr, int idx = 0) {
        int off = idx * slot_layout::WIDTH + slot_layout::laneIdx();
        return ptr[off];
    }

    /*
     * Return digit at index idx.
     *
     * FIXME: Not clear how to interpret this function with more exotic fixnum
     * implementations such as RNS.
     */
    __device__ static fixnum get(fixnum var, int idx) {
        return slot_layout::shfl(var, idx);
    }

    /*
     * Set var digit at index idx to be x.
     *
     * FIXME: Not clear how to interpret this function with more exotic fixnum
     * implementations such as RNS.
     */
    __device__ static void set(fixnum &var, fixnum x, int idx) {
        var = (slot_layout::laneIdx() == idx) ? x : var;
    }

    /*
     * Return digit in most significant place. Might be zero.
     *
     * TODO: Not clear how to interpret this function with more exotic fixnum
     * implementations such as RNS.
     */
    __device__ static fixnum top_digit(fixnum var) {
        return slot_layout::shfl(var, slot_layout::toplaneIdx);
    }

    /*
     * Return digit in the least significant place. Might be zero.
     *
     * TODO: Not clear how to interpret this function with more exotic fixnum
     * implementations such as RNS.
     */
    __device__ static fixnum bottom_digit(fixnum var) {
        return slot_layout::shfl(var, 0);
    }

    /***********************
     * Arithmetic functions.
     */

    // TODO: Handle carry in
    // TODO: A more consistent syntax might be
    // fixnum add(fixnum a, fixnum b)
    // fixnum add_cc(fixnum a, fixnum b, int &cy_out)
    // fixnum addc(fixnum a, fixnum b, int cy_in)
    // fixnum addc_cc(fixnum a, fixnum b, int cy_in, int &cy_out)
    __device__ static int add_cy(fixnum &r, fixnum a, fixnum b) {
        // FIXME: Can't call std::numeric_limits<fixnum>::max() on device.
        //static constexpr fixnum FIXNUM_MAX = std::numeric_limits<fixnum>::max();
        static constexpr fixnum FIXNUM_MAX = ~(fixnum)0;
        int cy, cy_hi;
        r = a + b;
        cy = r < a;
        // r propagates carries iff r = FIXNUM_MAX
        r += effective_carries(cy_hi, r == FIXNUM_MAX, cy);
        return cy_hi;
    }

    // TODO: Handle borrow in
    __device__ static int sub_br(fixnum &r, fixnum a, fixnum b) {
        static constexpr fixnum FIXNUM_MIN = 0;
        int br, br_hi;
        r = a - b;
        br = r > a;
        // r propagates borrows iff r = FIXNUM_MIN
        r -= effective_carries(br_hi, r == FIXNUM_MIN, br);
        return br_hi;
    }

    __device__ static fixnum zero() {
        return 0;
    }

    __device__ static fixnum one() {
        return (slot_layout::laneIdx() == 0);
    }

    __device__ static int incr_cy(fixnum &r) {
        return add_cy(r, r, one());
    }

    __device__ static int decr_br(fixnum &r) {
        return sub_br(r, r, one());
    }

    __device__ static void neg(fixnum &r, fixnum a) {
        (void) sub_br(r, zero(), a);
    }

    /*
     * r += a * u.
     *
     * The product a*u is done 'component-wise'; useful when a or u is
     * constant across the slot, though this is not necessary.
     */
    __device__ static fixnum mad_cy(fixnum &r, fixnum a, fixnum u) {
        fixnum cy_hi, hi;

        umad(hi, r, a, u, r);
        cy_hi = top_digit(hi);
        hi = slot_layout::shfl_up0(hi, 1);
        cy_hi += add_cy(r, hi, r);

        return cy_hi;
    }

    /*
     * r = a * u, where a is interpreted as a single word, and u a
     * full fixnum. a should be constant across the slot for the
     * result to make sense.
     *
     * TODO: Can this be refactored with mad_cy?
     * TODO: Come up with a better name for this function.
     */
    __device__ static word_tp muli(fixnum &r, word_tp a, fixnum u) {
        fixnum cy_hi, hi, lo;

        umul(hi, lo, a, u);
        cy_hi = top_digit(hi);
        hi = slot_layout::shfl_up0(hi, 1);
        cy_hi += add_cy(lo, lo, hi);

        return cy_hi;
    }

    /*
     * r = lo_half(a * b)
     *
     * The "lo_half" is the product modulo 2^(8*FIXNUM_BYTES),
     * i.e. the same size as the inputs.
     */
    __device__ static void mul_lo(fixnum &r, fixnum a, fixnum b) {
        // TODO: This should be smaller, probably uint16_t (smallest
        // possible for addition).  Strangely, the naive translation to
        // the smaller size broke; to investigate.
        fixnum cy = 0;

        r = 0;
        for (int i = slot_layout::WIDTH - 1; i >= 0; --i) {
            fixnum aa = slot_layout::shfl(a, i);

            // TODO: See if using umad.wide improves this.
            umad_hi_cc(r, cy, aa, b, r);
            // TODO: Could use rotate here, which is slightly
            // cheaper than shfl_up0...
            r = slot_layout::shfl_up0(r, 1);
            cy = slot_layout::shfl_up0(cy, 1);
            umad_lo_cc(r, cy, aa, b, r);
        }
        cy = slot_layout::shfl_up0(cy, 1);
        add_cy(r, r, cy);
    }

    __device__ static void sqr_lo(fixnum &r, fixnum a) {
        // TODO: Implement my smarter squaring algo.
        mul_lo(r, a, a);
    }

    /*
     * (s, r) = a * b
     *
     * r is the "lo half" (see mul_lo above) and s is the
     * corresponding "hi half".
     */
    __device__ static void mul_wide(fixnum &s, fixnum &r, fixnum a, fixnum b) {
        // TODO: See if we can get away with a smaller type for cy.
        fixnum cy = 0;
        int L = slot_layout::laneIdx();

        // TODO: Rewrite this using rotates instead of shuffles;
        // should be simpler and faster.
        r = s = 0;
        for (int i = slot_layout::WIDTH - 1; i >= 0; --i) {
            fixnum aa = slot_layout::shfl(a, i), t;

            // TODO: Review this code: it seems to have more shuffles than
            // necessary, and besides, why does it not use digit_addmuli?
            umad_hi_cc(r, cy, aa, b, r);

            t = slot_layout::shfl(cy, slot_layout::toplaneIdx);
            // TODO: Is there a way to avoid this add?  Definitely need to
            // propagate the carry at least one place, but maybe not more?
            // Previous (wrong) version: "s = (L == 0) ? s + t : s;"
            t = (L == 0) ? t : 0;
            add_cy(s, s, t);

            // shuffle up hi words
            s = slot_layout::shfl_up(s, 1);
            // most sig word of lo words becomes least sig of hi words
            t = slot_layout::shfl(r, slot_layout::toplaneIdx);
            s = (L == 0) ? t : s;

            r = slot_layout::shfl_up0(r, 1);
            cy = slot_layout::shfl_up0(cy, 1);
            umad_lo_cc(r, cy, aa, b, r);
        }
        // TODO: This carry propgation from r to s is a bit long-winded.
        // Can we simplify?
        // NB: cy_hi <= width.  TODO: Justify this explicitly.
        fixnum cy_hi = slot_layout::shfl(cy, slot_layout::toplaneIdx);
        cy = slot_layout::shfl_up0(cy, 1);
        cy = add_cy(r, r, cy);
        cy_hi += cy;  // Can't overflow since cy_hi <= width.
        assert(cy_hi >= cy);
        // TODO: Investigate: replacing the following two lines with
        // simply "s = (L == 0) ? s + cy_hi : s;" produces no detectible
        // errors. Can I prove that (MAX_UINT64 - s[0]) < width?
        cy = (L == 0) ? cy_hi : 0;
        cy = add_cy(s, s, cy);
        assert(cy == 0);
    }

    __device__ static void mul_hi(fixnum &s, fixnum a, fixnum b) {
        // TODO: implement this properly
        fixnum r;
        mul_wide(s, r, a, b);
    }

    __device__ static void sqr_wide(fixnum &s, fixnum &r, fixnum a) {
        // TODO: Implement my smarter squaring algo.
        mul_wide(s, r, a, a);
    }

    __device__ static void sqr_hi(fixnum &s, fixnum a) {
        // TODO: implement this properly
        fixnum r;
        sqr_wide(s, r, a);
    }

    /*
     * Return a mask of width bits whose ith bit is set if and only if
     * the ith digit of r is nonzero. In particular, result is zero
     * iff r is zero.
     */
    __device__ static uint32_t nonzero_mask(fixnum r) {
        return slot_layout::ballot(r != 0);
    }

    /*
     * Return -1, 0, or 1, depending on whether x is less than, equal
     * to, or greater than y.
     */
    __device__ static int cmp(fixnum x, fixnum y) {
        fixnum r;
        int br = sub_br(r, x, y);
        // r != 0 iff x != y. If x != y, then br != 0 => x < y.
        return nonzero_mask(r) ? (br ? -1 : 1) : 0;
    }

    /*
     * Return the index of the most significant digit of x, or -1 if x is
     * zero.
     */
    __device__ static int most_sig_dig(fixnum x) {
        // FIXME: Should be able to get this value from limits or numeric_limits
        // or whatever.
        enum { UINT32_BITS = 8 * sizeof(uint32_t) };
        static_assert(UINT32_BITS == 32, "uint32_t isn't 32 bits");

        uint32_t a = nonzero_mask(x);
        return UINT32_BITS - (clz(a) + 1);
    }

    /*
     * Return the index of the most significant bit of x, or -1 if x is
     * zero.
     *
     * TODO: Give this function a better name; maybe ceil_log2()?
     */
    __device__ static int msb(fixnum x) {
        int b = most_sig_dig(x);
        if (b < 0) return b;
        word_tp y = slot_layout::shfl(x, b);
        // TODO: These two lines are basically the same as most_sig_dig();
        // refactor.
        int c = clz(y);
        return WORD_BITS - (c + 1) + WORD_BITS * b;
    }

    /*
     * Return the 2-valuation of x, i.e. the integer k >= 0 such that
     * 2^k divides x but 2^(k+1) does not divide x.  Depending on the
     * representation, can think of this as CTZ(x) ("Count Trailing
     * Zeros").
     *
     * TODO: Refactor common code between here, msb() and
     * most_sig_dig(). Perhaps write msb in terms of two_valuation?
     *
     * FIXME: Pretty sure this function is broken; e.g. if x is 0 but width <
     * warpSize, the answer is wrong.
     */
    __device__ static int two_valuation(fixnum x) {
        // FIXME: Should be able to get this value from limits or numeric_limits
        // or whatever.
        enum { UINT32_BITS = 8 * sizeof(uint32_t) };
        static_assert(UINT32_BITS == 32, "uint32_t isn't 32 bits");

        uint32_t a = nonzero_mask(x);
        int b = ctz(a), c = 0;
        if (b < UINT32_BITS) {
            word_tp y = slot_layout::shfl(x, b);
            c = ctz(y);
        }
        return c + b * WORD_BITS;
    }

    /*
     * Set y to be x shifted by b bits to the left; effectively
     * multiply by 2^b. Return the top b bits of x.
     *
     * FIXME: Currently assumes that fixnum is unsigned.
     *
     * TODO: Think of better names for these functions. Something like
     * mul_2exp.
     */
    __device__ static fixnum lshift(fixnum &y, fixnum x, int b) {
        assert(b >= 0);
        assert(b <= FIXNUM_BITS);
        int q = b / WORD_BITS, r = b % WORD_BITS, rp = WORD_BITS - r;
        fixnum overflow;

        y = slot_layout::rotate_up(x, q);
        // Hi bits of y[i] (=overflow) become the lo bits of y[(i+1) % width]
        overflow = y >> rp;
        overflow = slot_layout::rotate_up(overflow, 1);
        y = (y << r) | overflow;

        int L = slot_layout::laneIdx();
        overflow = y & -(word_tp)(L <= q);   // Kill high (q-1) words of y;
        set(overflow, (overflow << rp) >> rp, q); // Kill high rp bits of overflow[q]
        y &= -(word_tp)(L >= q);             // Kill low q words of y;
        set(y, (y >> r) << r, q);            // Kill low r bits of y[q]
        return overflow;
    }

    /*
     * Set y to be x shifted by b bits to the right; effectively
     * divide by 2^b. Return the bottom b bits of x.
     *
     * FIXME: Currently assumes 0 <= b <= WORD_BITS, and that fixnum
     * is unsigned.
     *
     * TODO: Think of better names for these functions. Something like
     * mul_2exp.
     */
    __device__ static fixnum rshift(fixnum &y, fixnum x, int b) {
        fixnum z;
        y = lshift(z, x, FIXNUM_BITS - b);
        return z;
    }

private:
    __device__ static fixnum effective_carries(int &cy_hi, fixnum propagate, int cy) {
        int L = slot_layout::laneIdx();
        uint32_t allcarries, p, g;

        g = slot_layout::ballot(cy);              // carry generate
        p = slot_layout::ballot(propagate);       // carry propagate
        allcarries = (p | g) + g;                 // propagate all carries
        // NB: There is no way to unify these two expressions to remove the
        // conditional. The conditional should be optimised away though, since
        // WIDTH is a compile-time constant.
        cy_hi = (slot_layout::WIDTH == WARPSIZE) // detect hi overflow
            ? (allcarries < g)
            : ((allcarries >> slot_layout::WIDTH) & 1);
        allcarries = (allcarries ^ p) | (g << 1); // get effective carries
        return (allcarries >> L) & 1;
    }
};