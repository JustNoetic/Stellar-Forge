// FFT Cooley-Tukey Radix-4 implementation with dual-channel packed RGBA support
// Based on huj31415's algorithm adapted for Stellar-Forge

#define PI 3.14159265358979323846
#define TAU 6.28318530717958647692

#ifndef FFT_SIZE
#define FFT_SIZE 1024
#endif

#ifndef KERNEL_FRACTION
#define KERNEL_FRACTION 0.375
#endif

const int KERNEL_SIZE = int(float(FFT_SIZE) * KERNEL_FRACTION);
const uint N_4_1 = uint(FFT_SIZE / 4);
const uint N_4_2 = uint(FFT_SIZE / 2);
const uint N_4_3 = uint(3 * FFT_SIZE / 4);
const uint START_Ns = 4u;

#ifdef IFFT
  #define INV -1.0
#else
  #define INV 1.0
#endif

const float NORM = 1.0 / float(FFT_SIZE);
shared vec2 row[FFT_SIZE];

vec2 cmul(in vec2 A, in vec2 B) {
  return vec2(A.x * B.x - A.y * B.y, dot(A, B.yx));
}

// Swizzle shared memory address to prevent 32-bank conflicts
uint swizzle(uint x) {
  return (x & ~31u) | ((x & 31u) ^ ((x >> 5u) & 31u));
}

// 5-digit base-4 reversal for size 1024
uint digit_reverse_4(uint x) {
  return ((x & 3u) << 8u) | ((x & 12u) << 4u) | (x & 48u) | ((x & 192u) >> 4u) | ((x >> 8u) & 3u);
}

// In-place Cooley-Tukey Radix-4 butterfly step
void fftR4_CT(uint x, uint Ns) {
  uint Ns4 = Ns >> 2;
  uint offset = x & (Ns4 - 1u);
  uint base = (x / Ns4) * Ns + offset;
  uint i0 = swizzle(base);
  uint i1 = swizzle(base + Ns4);
  uint i2 = swizzle(base + Ns4 * 2u);
  uint i3 = swizzle(base + Ns4 * 3u);

  float angle = -INV * TAU * (float(offset) / float(Ns));
  vec2 w1 = vec2(cos(angle), sin(angle));
  vec2 t0 = row[i0];
  vec2 t1 = cmul(w1, row[i1]);
  vec2 w2 = cmul(w1, w1);
  vec2 t2 = cmul(w2, row[i2]);
  vec2 t3 = cmul(cmul(w2, w1), row[i3]);

  row[i0] = t0 + t1 + t2 + t3;
  row[i1] = t0 - t2 + vec2(t1.y - t3.y, t3.x - t1.x) * INV;
  row[i2] = t0 - t1 + t2 - t3;
  row[i3] = t0 - t2 + vec2(t3.y - t1.y, t1.x - t3.x) * INV;
}

// 1D packed RGBA FFT/IFFT using dual-channel real-to-complex transform
void rgba1DFFT(uint x, inout vec4 t0, inout vec4 t1, inout vec4 t2, inout vec4 t3) {
  row[swizzle(digit_reverse_4(x))]           = t0.rg;
  row[swizzle(digit_reverse_4(x + N_4_1))]   = t1.rg;
  row[swizzle(digit_reverse_4(x + N_4_2))]   = t2.rg;
  row[swizzle(digit_reverse_4(x + N_4_3))]   = t3.rg;
  barrier();

  for (uint Ns = START_Ns; Ns <= uint(FFT_SIZE); Ns <<= 2) {
    fftR4_CT(x, Ns);
    barrier();
  }

  t0.rg = row[swizzle(x)];
  t1.rg = row[swizzle(x + N_4_1)];
  t2.rg = row[swizzle(x + N_4_2)];
  t3.rg = row[swizzle(x + N_4_3)];
  barrier();

  row[swizzle(digit_reverse_4(x))]           = t0.ba;
  row[swizzle(digit_reverse_4(x + N_4_1))]   = t1.ba;
  row[swizzle(digit_reverse_4(x + N_4_2))]   = t2.ba;
  row[swizzle(digit_reverse_4(x + N_4_3))]   = t3.ba;
  barrier();

  for (uint Ns = START_Ns; Ns <= uint(FFT_SIZE); Ns <<= 2) {
    fftR4_CT(x, Ns);
    barrier();
  }

  t0.ba = row[swizzle(x)];
  t1.ba = row[swizzle(x + N_4_1)];
  t2.ba = row[swizzle(x + N_4_2)];
  t3.ba = row[swizzle(x + N_4_3)];
}
