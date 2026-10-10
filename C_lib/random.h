#include <stdio.h>
#include <stdint.h>

#ifndef RANDOM_H
#define RANDOM_H

typedef float f32;

typedef uint32_t u32;

typedef struct {
    uint64_t state;
    uint64_t inc;
} prng_state;

void prng_seed_r(prng_state* rng, uint64_t initstate, uint64_t initseq);
void prng_seed(uint64_t initstate, uint64_t initseq);

u32 prng_rand_r(prng_state* rng);
u32 prng_rand(void);

f32 prng_randf_r(prng_state* rng);
f32 prng_randf(void);

#endif // RANDOM_H
