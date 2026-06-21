/* kmer/encoders/_native/_common.h — shim that re-exports kmer/_common.h.
 *
 * The build system adds kmer/encoders/_native/ to the include path for
 * the encoder C extensions, so they include "_common.h" by short name.
 * This file forwards to the root shared header.
 */
#ifndef KMER_ENCODERS_NATIVE_COMMON_H_INCLUDED
#define KMER_ENCODERS_NATIVE_COMMON_H_INCLUDED
#include "../../_common.h"
#endif
