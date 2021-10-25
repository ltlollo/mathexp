#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <time.h>


#define REP8(v) { (v), (v), (v), (v), (v), (v), (v), (v), }

_Alignas(32) static const float one[8] = REP8(+1);
_Alignas(32) static const float rph[8] = REP8(M_2_PI); // 2/π
_Alignas(32) static const unsigned sgnmsk[8] = REP8((1u<<31));
_Alignas(32) static const float A[8] = REP8(-4.3604892e-1f);
_Alignas(32) static const float B[8] = REP8(-1.7441979e-1f);
_Alignas(32) static const float C[8] = REP8(1.6104687f);
_Alignas(32) static const float D[8] = REP8(-1.3670794e-3f);


_Alignas(32) static const float A_1[8] = REP8(-5.52557920938175e-1);
_Alignas(32) static const float B_1[8] = REP8(1.5480661860589578);

_Alignas(32) static const float A_2[8] = REP8(7.1860854233159339e-2);
_Alignas(32) static const float B_2[8] = REP8(-6.4211316698626402e-1);
_Alignas(32) static const float C_2[8] = REP8(1.5703200191555205);

_Alignas(32) static const float M_4_PI[8] = REP8(1.2732395447351626861510701069801);

_Alignas(32) static const float A_3[8] = REP8(-1.9095735348735695e-2);
_Alignas(32) static const float B_3[8] = REP8(2.5258023913456068e-1);
_Alignas(32) static const float C_3[8] = REP8(-1.233484503785825);
_Alignas(32) static const float D_3[8] = REP8(9.9999329528216742e-1);

_Alignas(32) static const float A_4[8] = REP8(2.239902736935567e-1);
_Alignas(32) static const float B_4[8] = REP8(-1.2227967326409367);
_Alignas(32) static const float C_4[8] = REP8(9.9940322947369002e-1);

_Alignas(32) static const float A_5[8] = REP8(9.7199520202293612e-1);

// order 2 poly of sin(x*pi/2) in [0:1]
__attribute__((noinline)) void
sincos256_0(void *xinv, void *sin, void *cos) {
    __m256 xin         = _mm256_load_ps(xinv);
    __m256  x          = _mm256_mul_ps(xin, *(__m256 *)rph);
    __m256i intpart_i  = _mm256_cvttps_epi32(x);
    __m256  intpart_s = _mm256_cvtepi32_ps(intpart_i);

    __m256 signbit = _mm256_and_ps(xin, *(__m256 *)sgnmsk);
    __m256 p    = _mm256_sub_ps(x, intpart_s);
    __m256 p0   = _mm256_xor_ps(p, signbit);
    __m256 p1   = _mm256_sub_ps(*(__m256 *)one, p0);

    __m256 a = _mm256_fmadd_ps(p0, *(__m256 *)A, *(__m256 *)B);
    __m256 b = _mm256_fmadd_ps(p1, *(__m256 *)A, *(__m256 *)B);
    a = _mm256_fmadd_ps(a, p0, *(__m256 *)C);
    b = _mm256_fmadd_ps(b, p1, *(__m256 *)C);
    a = _mm256_fmadd_ps(a, p0, *(__m256 *)D);
    b = _mm256_fmadd_ps(b, p1, *(__m256 *)D);

    __m256 blend = (__m256)_mm256_slli_epi32(intpart_i, 31);
    __m256 s = _mm256_blendv_ps(a, b, blend);
    __m256 c = _mm256_blendv_ps(b, a, blend);
    intpart_i = _mm256_sub_epi32(intpart_i,
            _mm256_srli_epi32((__m256i)signbit, 31));
    s = _mm256_xor_ps(s,
            (__m256)_mm256_slli_epi32(
                _mm256_srli_epi32(intpart_i, 1), 31));
    intpart_i = _mm256_add_epi32(intpart_i, _mm256_set1_epi32(1));
    c = _mm256_xor_ps(c,
            (__m256)_mm256_slli_epi32(
                _mm256_srli_epi32(intpart_i, 1), 31));
    _mm256_store_ps(sin, s);
    _mm256_store_ps(cos, c);
}

// order 1 poly of xP(x^2), where P(x)=sin(x*pi/2) in [0:1]
__attribute__((noinline)) void
sincos256_1(void *xinv, void *sin, void *cos) {
    __m256 xin         = _mm256_load_ps(xinv);
    __m256  x          = _mm256_mul_ps(xin, *(__m256 *)rph);
    __m256i intpart_i  = _mm256_cvttps_epi32(x);
    __m256  intpart_s = _mm256_cvtepi32_ps(intpart_i);

    __m256 signbit = _mm256_and_ps(xin, *(__m256 *)sgnmsk);
    __m256 p0   = _mm256_sub_ps(x, intpart_s);
    __m256 os   = _mm256_xor_ps(signbit, *(__m256 *)one);
    __m256 p1   = _mm256_sub_ps(os, p0);

    __m256 p02  = _mm256_mul_ps(p0, p0);
    __m256 p12  = _mm256_mul_ps(p1, p1);

    __m256 a = _mm256_fmadd_ps(p02, *(__m256 *)A_1, *(__m256 *)B_1);
    __m256 b = _mm256_fmadd_ps(p12, *(__m256 *)A_1, *(__m256 *)B_1);
    a = _mm256_mul_ps(a, p0);
    b = _mm256_mul_ps(b, p1);

    __m256 blend = (__m256)_mm256_slli_epi32(intpart_i, 31);
    __m256 s = _mm256_blendv_ps(a, b, blend);
    __m256 c = _mm256_blendv_ps(b, a, blend);
    s = _mm256_xor_ps(s, signbit);
    c = _mm256_xor_ps(c, signbit);
    intpart_i = _mm256_sub_epi32(intpart_i,
            _mm256_srli_epi32((__m256i)signbit, 31));
    s = _mm256_xor_ps(s,
            (__m256)_mm256_slli_epi32(
                _mm256_srli_epi32(intpart_i, 1), 31));
    intpart_i = _mm256_add_epi32(intpart_i, _mm256_set1_epi32(1));
    c = _mm256_xor_ps(c,
            (__m256)_mm256_slli_epi32(
                _mm256_srli_epi32(intpart_i, 1), 31));

    _mm256_store_ps(sin, s);
    _mm256_store_ps(cos, c);
}


// order 2 poly of xP(x^2), where P(x)=sin(x*pi/2) in [0:1]
__attribute__((noinline)) void
sincos256_2(void *xinv, void *sin, void *cos) {
    __m256 xin         = _mm256_load_ps(xinv);
    __m256  x          = _mm256_mul_ps(xin, *(__m256 *)rph);
    __m256i intpart_i  = _mm256_cvttps_epi32(x);
    __m256  intpart_s = _mm256_cvtepi32_ps(intpart_i);

    __m256 signbit = _mm256_and_ps(xin, *(__m256 *)sgnmsk);
    __m256 p0   = _mm256_sub_ps(x, intpart_s);
    __m256 os   = _mm256_xor_ps(signbit, *(__m256 *)one);
    __m256 p1   = _mm256_sub_ps(os, p0);

    __m256 p02  = _mm256_mul_ps(p0, p0);
    __m256 p12  = _mm256_mul_ps(p1, p1);

    __m256 a = _mm256_fmadd_ps(p02, *(__m256 *)A_2, *(__m256 *)B_2);
    __m256 b = _mm256_fmadd_ps(p12, *(__m256 *)A_2, *(__m256 *)B_2);
    a = _mm256_fmadd_ps(a, p02, *(__m256 *)C_2);
    b = _mm256_fmadd_ps(b, p12, *(__m256 *)C_2);
    a = _mm256_mul_ps(a, p0);
    b = _mm256_mul_ps(b, p1);

    __m256 blend = (__m256)_mm256_slli_epi32(intpart_i, 31);
    __m256 s = _mm256_blendv_ps(a, b, blend);
    __m256 c = _mm256_blendv_ps(b, a, blend);
    s = _mm256_xor_ps(s, signbit);
    c = _mm256_xor_ps(c, signbit);
    intpart_i = _mm256_sub_epi32(intpart_i,
            _mm256_srli_epi32((__m256i)signbit, 31));
    s = _mm256_xor_ps(s,
            (__m256)_mm256_slli_epi32(
                _mm256_srli_epi32(intpart_i, 1), 31));
    intpart_i = _mm256_add_epi32(intpart_i, _mm256_set1_epi32(1));
    c = _mm256_xor_ps(c,
            (__m256)_mm256_slli_epi32(
                _mm256_srli_epi32(intpart_i, 1), 31));

    _mm256_store_ps(sin, s);
    _mm256_store_ps(cos, c);
}

// order 3 poly of P(x^2), where P(x)=cos(x*pi/2) in [0:1]
__attribute__((noinline)) void
sincos256_3(void *xinv, void *sin, void *cos) {
    __m256 xin         = _mm256_load_ps(xinv);
    __m256  x          = _mm256_mul_ps(xin, *(__m256 *)rph);
    __m256i intpart_i  = _mm256_cvttps_epi32(x);
    __m256  intpart_s = _mm256_cvtepi32_ps(intpart_i);

    __m256 signbit = _mm256_and_ps(xin, *(__m256 *)sgnmsk);
    __m256 p0   = _mm256_sub_ps(x, intpart_s);
    __m256 os   = _mm256_xor_ps(signbit, *(__m256 *)one);
    __m256 p1   = _mm256_sub_ps(os, p0);

    __m256 p02  = _mm256_mul_ps(p0, p0);
    __m256 p12  = _mm256_mul_ps(p1, p1);

    __m256 a = _mm256_fmadd_ps(p02, *(__m256 *)A_3, *(__m256 *)B_3);
    __m256 b = _mm256_fmadd_ps(p12, *(__m256 *)A_3, *(__m256 *)B_3);
    a = _mm256_fmadd_ps(a, p02, *(__m256 *)C_3);
    b = _mm256_fmadd_ps(b, p12, *(__m256 *)C_3);
    a = _mm256_fmadd_ps(a, p02, *(__m256 *)D_3);
    b = _mm256_fmadd_ps(b, p12, *(__m256 *)D_3);

    __m256 blend = (__m256)_mm256_slli_epi32(intpart_i, 31);
    __m256 s = _mm256_blendv_ps(b, a, blend);
    __m256 c = _mm256_blendv_ps(a, b, blend);

    intpart_i = _mm256_sub_epi32(intpart_i,
            _mm256_srli_epi32((__m256i)signbit, 31));
    s = _mm256_xor_ps(s,
            (__m256)_mm256_slli_epi32(
                _mm256_srli_epi32(intpart_i, 1), 31));
    intpart_i = _mm256_add_epi32(intpart_i, _mm256_set1_epi32(1));
    c = _mm256_xor_ps(c,
            (__m256)_mm256_slli_epi32(
                _mm256_srli_epi32(intpart_i, 1), 31));

    _mm256_store_ps(sin, s);
    _mm256_store_ps(cos, c);
}

// order 2 poly of P(x^2), where P(x)=cos(x*pi/2) in [0:1]
__attribute__((noinline)) void
sincos256_4(void *xinv, void *sin, void *cos) {
    __m256 xin         = _mm256_load_ps(xinv);
    __m256  x          = _mm256_mul_ps(xin, *(__m256 *)rph);
    __m256i intpart_i  = _mm256_cvttps_epi32(x);
    __m256  intpart_s = _mm256_cvtepi32_ps(intpart_i);

    __m256 signbit = _mm256_and_ps(xin, *(__m256 *)sgnmsk);
    __m256 p0   = _mm256_sub_ps(x, intpart_s);
    __m256 os   = _mm256_xor_ps(signbit, *(__m256 *)one);
    __m256 p1   = _mm256_sub_ps(os, p0);

    __m256 p02  = _mm256_mul_ps(p0, p0);
    __m256 p12  = _mm256_mul_ps(p1, p1);

    __m256 a = _mm256_fmadd_ps(p02, *(__m256 *)A_4, *(__m256 *)B_4);
    __m256 b = _mm256_fmadd_ps(p12, *(__m256 *)A_4, *(__m256 *)B_4);
    a = _mm256_fmadd_ps(a, p02, *(__m256 *)C_4);
    b = _mm256_fmadd_ps(b, p12, *(__m256 *)C_4);

    __m256 blend = (__m256)_mm256_slli_epi32(intpart_i, 31);
    __m256 s = _mm256_blendv_ps(b, a, blend);
    __m256 c = _mm256_blendv_ps(a, b, blend);

    intpart_i = _mm256_sub_epi32(intpart_i,
            _mm256_srli_epi32((__m256i)signbit, 31));
    s = _mm256_xor_ps(s,
            (__m256)_mm256_slli_epi32(
                _mm256_srli_epi32(intpart_i, 1), 31));
    intpart_i = _mm256_add_epi32(intpart_i, _mm256_set1_epi32(1));
    c = _mm256_xor_ps(c,
            (__m256)_mm256_slli_epi32(
                _mm256_srli_epi32(intpart_i, 1), 31));

    _mm256_store_ps(sin, s);
    _mm256_store_ps(cos, c);
}

// order 1 poly of P(x^2), where P(x)=cos(x*pi/2) in [0:1], -x^2+0.972 How bad can it be...
__attribute__((noinline)) void
sincos256_5(void *xinv, void *sin, void *cos) {
    __m256 xin         = _mm256_load_ps(xinv);
    __m256  x          = _mm256_mul_ps(xin, *(__m256 *)rph);
    __m256i intpart_i  = _mm256_cvttps_epi32(x);
    __m256  intpart_s = _mm256_cvtepi32_ps(intpart_i);

    __m256 signbit = _mm256_and_ps(xin, *(__m256 *)sgnmsk);
    __m256 p0   = _mm256_sub_ps(x, intpart_s);
    __m256 os   = _mm256_xor_ps(signbit, *(__m256 *)one);
    __m256 p1   = _mm256_sub_ps(os, p0);

    __m256 p02  = _mm256_mul_ps(p0, p0);
    __m256 p12  = _mm256_mul_ps(p1, p1);

    __m256 a = _mm256_fnmadd_ps(p0, p0, *(__m256 *)A_5);
    __m256 b = _mm256_fnmadd_ps(p1, p1, *(__m256 *)A_5);

    __m256 blend = (__m256)_mm256_slli_epi32(intpart_i, 31);
    __m256 s = _mm256_blendv_ps(b, a, blend);
    __m256 c = _mm256_blendv_ps(a, b, blend);

    intpart_i = _mm256_sub_epi32(intpart_i,
            _mm256_srli_epi32((__m256i)signbit, 31));
    s = _mm256_xor_ps(s,
            (__m256)_mm256_slli_epi32(
                _mm256_srli_epi32(intpart_i, 1), 31));
    intpart_i = _mm256_add_epi32(intpart_i, _mm256_set1_epi32(1));
    c = _mm256_xor_ps(c,
            (__m256)_mm256_slli_epi32(
                _mm256_srli_epi32(intpart_i, 1), 31));

    _mm256_store_ps(sin, s);
    _mm256_store_ps(cos, c);
}


void show(int i, int mode, float *vsin, float *vcos,
    struct timespec end, struct timespec beg
) {
    if (mode == 0) {
        printf("%d time: %06fms\n", i,
            ((end.tv_sec-beg.tv_sec)*1000000000+(end.tv_nsec-beg.tv_nsec)) /
            (double)(1000000)
        );
        printf("sin: %g %g %g %g %g %g %g %g\ncos: %g %g %g %g %g %g %g %g\n",
            vsin[0], vsin[1], vsin[2], vsin[3], vsin[4], vsin[5], vsin[6], vsin[7],
            vcos[0], vcos[1], vcos[2], vcos[3], vcos[4], vcos[5], vcos[6], vcos[7]
        );
    } else {
        printf("%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n",
            vsin[0]>0, vsin[1]>0, vsin[2]>0, vsin[3]>0,
            vsin[4]>0, vsin[5]>0, vsin[6]>0, vsin[7]>0,
            vcos[0]>0, vcos[1]>0, vcos[2]>0, vcos[3]>0,
            vcos[4]>0, vcos[5]>0, vcos[6]>0, vcos[7]>0
        );
    }

}


int main() {
    _Alignas(32) float vsin[8];
    _Alignas(32) float vcos[8];
    _Alignas(32) float x[8] = { -5, -4, -2, -1, 1, 2, 4, 5 };

    int mode = 0, ver = 0;
    size_t repeat = 100000000;
    //repeat = 1;

    struct timespec beg, end;

    clock_gettime(CLOCK_MONOTONIC_RAW, &beg);
    for (size_t i = 0; i < repeat; i++) {
        asm volatile ("":::"memory");
        sincos256_0(x, vsin, vcos);
        asm volatile ("":::"memory");
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    show(ver++, mode, vsin, vcos, end, beg);

    clock_gettime(CLOCK_MONOTONIC_RAW, &beg);
    for (size_t i = 0; i < repeat; i++) {
        asm volatile ("":::"memory");
        sincos256_1(x, vsin, vcos);
        asm volatile ("":::"memory");
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    show(ver++, mode, vsin, vcos, end, beg);

    clock_gettime(CLOCK_MONOTONIC_RAW, &beg);
    for (size_t i = 0; i < repeat; i++) {
        asm volatile ("":::"memory");
        sincos256_2(x, vsin, vcos);
        asm volatile ("":::"memory");
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    show(ver++, mode, vsin, vcos, end, beg);

    clock_gettime(CLOCK_MONOTONIC_RAW, &beg);
    for (size_t i = 0; i < repeat; i++) {
        asm volatile ("":::"memory");
        sincos256_3(x, vsin, vcos);
        asm volatile ("":::"memory");
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &beg);
    show(ver++, mode, vsin, vcos, end, beg);

    clock_gettime(CLOCK_MONOTONIC_RAW, &beg);
    for (size_t i = 0; i < repeat; i++) {
        asm volatile ("":::"memory");
        sincos256_4(x, vsin, vcos);
        asm volatile ("":::"memory");
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    show(ver++, mode, vsin, vcos, end, beg);

    clock_gettime(CLOCK_MONOTONIC_RAW, &beg);
    for (size_t i = 0; i < repeat; i++) {
        asm volatile ("":::"memory");
        sincos256_5(x, vsin, vcos);
        asm volatile ("":::"memory");
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    show(ver++, mode, vsin, vcos, end, beg);

    for (size_t i = 0; i < 8; i++) {
        vsin[i] = sin(x[i]), vcos[i] = cos(x[i]);
    }
    show(999, mode, vsin, vcos, end, beg);
}
