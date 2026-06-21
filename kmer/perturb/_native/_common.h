/* kmer/perturb/_native/_common.h — backward-compat shim.
 *
 * All shared utilities now live in kmer/_common.h. This file is kept
 * so the existing _shuffler.c and _chunker.c includes keep working
 * without modification. New code should include kmer/_common.h directly.
 *
 * The old names (NUC_LUT, IUPAC_RC, philox_stream_t, philox_init,
 * philox_u32, philox_below, philox_double, philox_derive_key,
 * splitmix64, is_strict_acgt, is_iupac) are #defined as aliases for
 * the new kmer_-prefixed versions.
 */
#ifndef KMER_PERTURB_COMMON_H_INCLUDED
#define KMER_PERTURB_COMMON_H_INCLUDED

#include "../../_common.h"  /* path relative to perturb/_native/ */

/* Backward-compat aliases. */
#define NUC_LUT              KMER_NUC_LUT
#define IUPAC_RC             KMER_IUPAC_RC
#define is_strict_acgt       kmer_is_strict_acgt
#define is_iupac             kmer_is_iupac
#define philox_stream_t      kmer_philox_stream_t
#define philox_init          kmer_philox_init
#define philox_u32           kmer_philox_u32
#define philox_below         kmer_philox_below
#define philox_double        kmer_philox_double
#define philox_derive_key    kmer_philox_derive_key
#define splitmix64           kmer_splitmix64

#endif  /* KMER_PERTURB_COMMON_H_INCLUDED */
