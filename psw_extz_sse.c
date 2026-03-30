#include <limits.h>
#include <stdint.h>
#include "psw.h"

void psw_extz_ps(void *km, int qlen, const uint8_t *query,
                 int tlen, const psw_prof_t *target,
                 int8_t m, const int8_t *mat,
                 int8_t gapo, int8_t gape, int w, int zdrop, int flag, psw_extz_t *ez);

#if defined(__SSE2__)
#include <emmintrin.h>

typedef struct { int32_t h, e; } eh_t;

typedef struct {
	int16_t *qp;
	int16_t *tf;
	int32_t *go_q;
	int32_t *ge_q;
	int32_t *go_t;
	int32_t *ge_t;
	int16_t scale;
	int8_t scale_shift;
	int mpad;
} psw_sse_dp_t;

static inline int16_t psw_sat16(int32_t x)
{
	if (x > INT16_MAX) return INT16_MAX;
	if (x < INT16_MIN) return INT16_MIN;
	return (int16_t)x;
}

static inline int16_t psw_pick_safe_scale_pow2(int8_t m, const int8_t *mat, int8_t gapo, int8_t gape)
{
	int a, b;
	int32_t smax = 0;
	int32_t coeff, max_scale;
	int32_t pow2 = 1;

	for (a = 0; a < m; ++a)
		for (b = 0; b < m; ++b)
			if ((int32_t)mat[a * m + b] > smax)
				smax = (int32_t)mat[a * m + b];

	coeff = smax + (int32_t)2 * gapo + (int32_t)2 * gape;
	if (coeff <= 0) coeff = 1;
	max_scale = INT16_MAX / coeff;
	while ((pow2 << 1) > 0 && (pow2 << 1) <= max_scale)
		pow2 <<= 1;
	if (pow2 < 2) return 0;
	return (int16_t)pow2;
}

static inline int8_t psw_scale_to_shift(int16_t scale)
{
	int8_t sh = 0;
	if (scale <= 0) return -1;
	while ((scale & 1) == 0) {
		scale >>= 1;
		++sh;
	}
	return scale == 1 ? sh : -1;
}

static inline int16_t psw_to_scaled_ratio(int32_t num, int32_t den, int16_t scale)
{
	int64_t den64, scaled, rounded;

	if (den <= 0) den = 1;
	if (scale <= 0) scale = 1;
	den64 = (int64_t)den;
	scaled = (int64_t)num * (int64_t)scale;
	if (scaled >= 0) rounded = (scaled + den64 / 2) / den64;
	else             rounded = (scaled - den64 / 2) / den64;
	if (rounded > INT16_MAX) return INT16_MAX;
	if (rounded < INT16_MIN) return INT16_MIN;
	return (int16_t)rounded;
}

static inline int psw_scaled_to_int(int32_t x, int16_t scale)
{
	int64_t scaled = (int64_t)x;
	int64_t den = scale > 0 ? (int64_t)scale : 1;
	if (scaled >= 0) return (int)((scaled + den / 2) / den);
	return (int)((scaled - den / 2) / den);
}

static inline int psw_roundup8(int x)
{
	return (x + 7) & ~7;
}

static inline int16_t *psw_gen_base_freq_i16(void *km, int len, const psw_prof_t *p, int8_t m, int16_t scale)
{
	int i, a;
	int16_t *bf;

	bf = (int16_t*)kmalloc(km, (size_t)len * sizeof(int16_t));
	if (bf == 0) return 0;

	for (i = 0; i < len; ++i) {
		const uint32_t *col = p->prof + (size_t)i * p->dim;
		int32_t sum = 0;
		for (a = 0; a < m; ++a)
			sum += (int32_t)col[a];
		bf[i] = psw_sat16(sum);
		if (bf[i] < 0) bf[i] = 0;
		if (bf[i] > scale) bf[i] = scale;
	}
	return bf;
}

static inline int16_t *psw_gen_qp_i16_pad(void *km, int qlen, const psw_prof_t *query, int8_t m, int mpad, const int8_t *mat)
{
	int a, b, j;
	int16_t *qp;

	qp = (int16_t*)kmalloc(km, (size_t)qlen * mpad * sizeof(int16_t));
	if (qp == 0) return 0;

	for (j = 0; j < qlen; ++j) {
		const uint32_t *qcol = query->prof + (size_t)j * query->dim;
		int16_t *dst = qp + (size_t)j * mpad;
		for (b = 0; b < m; ++b) {
			int32_t s = 0;
			for (a = 0; a < m; ++a)
				s += (int32_t)qcol[a] * (int32_t)mat[a * m + b];
			dst[b] = psw_sat16(s);
		}
		for (; b < mpad; ++b)
			dst[b] = 0;
	}
	return qp;
}

static inline int16_t *psw_gen_qp_from_seq_i16_pad(void *km, int qlen, const uint8_t *query,
                                                    int8_t m, int mpad, const int8_t *mat,
                                                    int16_t scale)
{
	int i, b;
	int16_t *qp;

	qp = (int16_t*)kmalloc(km, (size_t)qlen * mpad * sizeof(int16_t));
	if (qp == 0) return 0;

	for (i = 0; i < qlen; ++i) {
		const int q = (int)query[i];
		const int8_t *row = mat + (size_t)q * m;
		int16_t *dst = qp + (size_t)i * mpad;
		for (b = 0; b < m; ++b)
			dst[b] = psw_sat16((int32_t)row[b] * (int32_t)scale);
		for (; b < mpad; ++b)
			dst[b] = 0;
	}
	return qp;
}

static inline int16_t *psw_gen_tf_i16_pad(void *km, int tlen, const psw_prof_t *target, int8_t m, int mpad)
{
	int i, b;
	int16_t *tf;

	tf = (int16_t*)kmalloc(km, (size_t)tlen * mpad * sizeof(int16_t));
	if (tf == 0) return 0;

	for (i = 0; i < tlen; ++i) {
		const uint32_t *tcol = target->prof + (size_t)i * target->dim;
		int16_t *dst = tf + (size_t)i * mpad;
		for (b = 0; b < m; ++b)
			dst[b] = psw_sat16((int32_t)tcol[b]);
		for (; b < mpad; ++b)
			dst[b] = 0;
	}
	return tf;
}

static inline int psw_make_norm_prof(void *km, const psw_prof_t *src, psw_prof_t *dst, int8_t m, int16_t scale)
{
	int i, a;
	int32_t depth;
	uint32_t *prof;

	if (src == 0 || dst == 0 || src->prof == 0 || src->len < 0 || src->dim <= 0 || m <= 0 || scale <= 0) return 0;
	if (src->dim < m) return 0;

	prof = (uint32_t*)kmalloc(km, (size_t)src->len * src->dim * sizeof(uint32_t));
	if (prof == 0) return 0;

	depth = src->depth > 0 ? (int32_t)src->depth : 1;
	for (i = 0; i < src->len; ++i) {
		const uint32_t *in = src->prof + (size_t)i * src->dim;
		uint32_t *out = prof + (size_t)i * src->dim;
		for (a = 0; a < src->dim; ++a) {
			if (a < m) {
				int16_t v = psw_to_scaled_ratio((int32_t)in[a], depth, scale);
				if (v < 0) v = 0;
				if (v > scale) v = scale;
				out[a] = (uint32_t)v;
			} else out[a] = in[a];
		}
	}

	dst->len = src->len;
	dst->dim = src->dim;
	dst->depth = scale;
	dst->prof = prof;
	return 1;
}

static inline void psw_free_norm_prof(void *km, psw_prof_t *p)
{
	if (p && p->prof) {
		kfree(km, (void*)p->prof);
		p->prof = 0;
	}
}

static inline int32_t psw_hsum_epi32_sse2(__m128i v)
{
	__m128i hi = _mm_srli_si128(v, 8);
	__m128i s = _mm_add_epi32(v, hi);
	hi = _mm_srli_si128(s, 4);
	s = _mm_add_epi32(s, hi);
	return _mm_cvtsi128_si32(s);
}

static inline int32_t psw_dot_scaled_sse_mpad8_i32(const int16_t *x, const int16_t *y, int8_t scale_shift)
{
	__m128i xv0 = _mm_loadu_si128((const __m128i*)x);
	__m128i yv0 = _mm_loadu_si128((const __m128i*)y);
	__m128i sum = _mm_madd_epi16(xv0, yv0);
	return psw_hsum_epi32_sse2(sum) >> scale_shift;
}

static inline int32_t psw_dot_scaled_sse_mpad16_i32(const int16_t *x, const int16_t *y, int8_t scale_shift)
{
	__m128i xv0 = _mm_loadu_si128((const __m128i*)x);
	__m128i yv0 = _mm_loadu_si128((const __m128i*)y);
	__m128i xv1 = _mm_loadu_si128((const __m128i*)(x + 8));
	__m128i yv1 = _mm_loadu_si128((const __m128i*)(y + 8));
	__m128i sum = _mm_add_epi32(_mm_madd_epi16(xv0, yv0), _mm_madd_epi16(xv1, yv1));
	return psw_hsum_epi32_sse2(sum) >> scale_shift;
}

static inline int32_t psw_dot_scaled_sse_mpad24_i32(const int16_t *x, const int16_t *y, int8_t scale_shift)
{
	__m128i xv0 = _mm_loadu_si128((const __m128i*)x);
	__m128i yv0 = _mm_loadu_si128((const __m128i*)y);
	__m128i xv1 = _mm_loadu_si128((const __m128i*)(x + 8));
	__m128i yv1 = _mm_loadu_si128((const __m128i*)(y + 8));
	__m128i xv2 = _mm_loadu_si128((const __m128i*)(x + 16));
	__m128i yv2 = _mm_loadu_si128((const __m128i*)(y + 16));
	__m128i sum = _mm_add_epi32(_mm_madd_epi16(xv0, yv0), _mm_madd_epi16(xv1, yv1));
	sum = _mm_add_epi32(sum, _mm_madd_epi16(xv2, yv2));
	return psw_hsum_epi32_sse2(sum) >> scale_shift;
}

static inline int32_t psw_dot_scaled_sse_mpad32_i32(const int16_t *x, const int16_t *y, int8_t scale_shift)
{
	__m128i xv0 = _mm_loadu_si128((const __m128i*)x);
	__m128i yv0 = _mm_loadu_si128((const __m128i*)y);
	__m128i xv1 = _mm_loadu_si128((const __m128i*)(x + 8));
	__m128i yv1 = _mm_loadu_si128((const __m128i*)(y + 8));
	__m128i xv2 = _mm_loadu_si128((const __m128i*)(x + 16));
	__m128i yv2 = _mm_loadu_si128((const __m128i*)(y + 16));
	__m128i xv3 = _mm_loadu_si128((const __m128i*)(x + 24));
	__m128i yv3 = _mm_loadu_si128((const __m128i*)(y + 24));
	__m128i sum0 = _mm_add_epi32(_mm_madd_epi16(xv0, yv0), _mm_madd_epi16(xv1, yv1));
	__m128i sum1 = _mm_add_epi32(_mm_madd_epi16(xv2, yv2), _mm_madd_epi16(xv3, yv3));
	return psw_hsum_epi32_sse2(_mm_add_epi32(sum0, sum1)) >> scale_shift;
}

static inline int32_t psw_dot_scaled_sse_padded_i32(const int16_t *x, const int16_t *y, int mpad, int8_t scale_shift)
{
	__m128i sum0 = _mm_setzero_si128();
	__m128i sum1 = _mm_setzero_si128();
	int i = 0;

	/* Fast paths for common alphabets (mpad is rounded to a multiple of 8). */
	switch (mpad) {
	case 8:  return psw_dot_scaled_sse_mpad8_i32(x, y, scale_shift);
	case 16: return psw_dot_scaled_sse_mpad16_i32(x, y, scale_shift);
	case 24: return psw_dot_scaled_sse_mpad24_i32(x, y, scale_shift);
	case 32: return psw_dot_scaled_sse_mpad32_i32(x, y, scale_shift);
	default: break;
	}

	for (; i + 15 < mpad; i += 16) {
		__m128i xv0 = _mm_loadu_si128((const __m128i*)(x + i));
		__m128i yv0 = _mm_loadu_si128((const __m128i*)(y + i));
		__m128i xv1 = _mm_loadu_si128((const __m128i*)(x + i + 8));
		__m128i yv1 = _mm_loadu_si128((const __m128i*)(y + i + 8));
		sum0 = _mm_add_epi32(sum0, _mm_madd_epi16(xv0, yv0));
		sum1 = _mm_add_epi32(sum1, _mm_madd_epi16(xv1, yv1));
	}
	sum0 = _mm_add_epi32(sum0, sum1);
	for (; i < mpad; i += 8) {
		__m128i xv = _mm_loadu_si128((const __m128i*)(x + i));
		__m128i yv = _mm_loadu_si128((const __m128i*)(y + i));
		sum0 = _mm_add_epi32(sum0, _mm_madd_epi16(xv, yv));
	}
	return psw_hsum_epi32_sse2(sum0) >> scale_shift;
}

static inline void psw_dot2_scaled_sse_padded_i32(const int16_t *x0, const int16_t *x1,
	                                               const int16_t *y, int mpad, int8_t scale_shift,
	                                               int32_t *d0, int32_t *d1)
{
	__m128i s00 = _mm_setzero_si128(), s01 = _mm_setzero_si128();
	__m128i s10 = _mm_setzero_si128(), s11 = _mm_setzero_si128();
	int i = 0;
	for (; i + 15 < mpad; i += 16) {
		__m128i yv0 = _mm_loadu_si128((const __m128i*)(y + i));
		__m128i yv1 = _mm_loadu_si128((const __m128i*)(y + i + 8));
		__m128i x00 = _mm_loadu_si128((const __m128i*)(x0 + i));
		__m128i x01 = _mm_loadu_si128((const __m128i*)(x0 + i + 8));
		__m128i x10 = _mm_loadu_si128((const __m128i*)(x1 + i));
		__m128i x11 = _mm_loadu_si128((const __m128i*)(x1 + i + 8));
		s00 = _mm_add_epi32(s00, _mm_madd_epi16(x00, yv0));
		s01 = _mm_add_epi32(s01, _mm_madd_epi16(x01, yv1));
		s10 = _mm_add_epi32(s10, _mm_madd_epi16(x10, yv0));
		s11 = _mm_add_epi32(s11, _mm_madd_epi16(x11, yv1));
	}
	s00 = _mm_add_epi32(s00, s01);
	s10 = _mm_add_epi32(s10, s11);
	for (; i < mpad; i += 8) {
		__m128i yv = _mm_loadu_si128((const __m128i*)(y + i));
		__m128i xv0 = _mm_loadu_si128((const __m128i*)(x0 + i));
		__m128i xv1 = _mm_loadu_si128((const __m128i*)(x1 + i));
		s00 = _mm_add_epi32(s00, _mm_madd_epi16(xv0, yv));
		s10 = _mm_add_epi32(s10, _mm_madd_epi16(xv1, yv));
	}
	*d0 = psw_hsum_epi32_sse2(s00) >> scale_shift;
	*d1 = psw_hsum_epi32_sse2(s10) >> scale_shift;
}

static inline void psw_fill_row_scores_sse(const int16_t *qp, const int16_t *tfi,
                                           int mpad, int8_t scale_shift,
                                           int st, int en, int32_t *sv)
{
	int j = st;
	const int16_t *qpj = qp + (size_t)st * mpad;
	int32_t *dst = sv;

	for (; j + 1 <= en; j += 2, qpj += (size_t)2 * mpad, dst += 2)
		psw_dot2_scaled_sse_padded_i32(qpj, qpj + mpad, tfi, mpad, scale_shift, &dst[0], &dst[1]);
	if (j <= en)
		dst[0] = psw_dot_scaled_sse_padded_i32(qpj, tfi, mpad, scale_shift);
}

static inline int psw_apply_zdrop_scaled(psw_extz_t *ez, int is_rot, int32_t H,
                                         int a, int b, int32_t zdrop, int32_t e)
{
	int r, t;
	if (is_rot) r = a, t = b;
	else r = a + b, t = a;

	if (H > (int32_t)ez->max) {
		ez->max = H;
		ez->max_t = t;
		ez->max_q = r - t;
	} else if (t >= ez->max_t && r - t >= ez->max_q) {
		int tl = t - ez->max_t;
		int ql = (r - t) - ez->max_q;
		int l = tl > ql ? tl - ql : ql - tl;
		if (zdrop >= 0 && (int32_t)ez->max - H > zdrop + (int32_t)l * e) {
			ez->zdropped = 1;
			return 1;
		}
	}
	return 0;
}

static void psw_extz_sse_core(void *km, int qlen, int tlen, const psw_sse_dp_t *dp,
                              int w, int zdrop, int flag, psw_extz_t *ez)
{
	eh_t *eh = 0;
	int32_t i, j, max_j = 0, n_col, *off = 0;
	int32_t ht0 = 0;
	int32_t zdrop_i32, gape_i32;
	int with_cigar;
	uint8_t *z = 0;
	int32_t *sv = 0;

	psw_reset_extz(ez);
	with_cigar = !(flag & PSW_FLAG_SCORE_ONLY);
	if (w < 0) w = tlen > qlen ? tlen : qlen;
	n_col = qlen < 2 * w + 1 ? qlen : 2 * w + 1;
	zdrop_i32 = zdrop < 0 ? -1 : zdrop * (int32_t)dp->scale;
	gape_i32 = dp->ge_q[0];

	eh = (eh_t*)kcalloc(km, qlen + 1, sizeof(eh_t));
	if (eh == 0) return;

	if (with_cigar) {
		z = (uint8_t*)kmalloc(km, (size_t)n_col * tlen);
		off = (int32_t*)kcalloc(km, tlen, sizeof(int32_t));
		if (z == 0 || off == 0) {
			if (z) kfree(km, z);
			if (off) kfree(km, off);
			kfree(km, eh);
			return;
		}
	}

	if (qlen == 0 || tlen == 0) {
		int32_t s32 = 0;
		if (qlen == 0) {
			for (i = 0; i < tlen; ++i)
				s32 -= i == 0 ? dp->go_t[i] + dp->ge_t[i] : dp->ge_t[i];
		} else {
			for (j = 0; j < qlen; ++j)
				s32 -= j == 0 ? dp->go_q[j] + dp->ge_q[j] : dp->ge_q[j];
		}
		ez->score = psw_scaled_to_int(s32, dp->scale);
		ez->mqe = ez->mte = ez->score;
		ez->mqe_t = tlen > 0 ? tlen - 1 : -1;
		ez->mte_q = qlen > 0 ? qlen - 1 : -1;
		ez->reach_end = 1;
		if (with_cigar) {
			ez->n_cigar = 0;
			if (qlen == 0 && tlen > 0)
				ez->cigar = psw_push_cigar(km, &ez->n_cigar, &ez->m_cigar, ez->cigar, PSW_CIGAR_DEL, tlen);
			else if (tlen == 0 && qlen > 0)
				ez->cigar = psw_push_cigar(km, &ez->n_cigar, &ez->m_cigar, ez->cigar, PSW_CIGAR_INS, qlen);
		}
		if (z) kfree(km, z);
		if (off) kfree(km, off);
		kfree(km, eh);
		return;
	}

	{
		int32_t hqj = 0;
		int32_t go_t0 = dp->go_t[0], ge_t0 = dp->ge_t[0];
		eh[0].h = 0;
		eh[0].e = -(go_t0 + ge_t0);
		for (j = 1; j <= qlen && j <= w; ++j) {
			if (j == 1) hqj = -(dp->go_q[j - 1] + dp->ge_q[j - 1]);
			else        hqj -= dp->ge_q[j - 1];
			eh[j].h = hqj;
			eh[j].e = hqj - (go_t0 + ge_t0);
		}
		for (; j <= qlen; ++j)
			eh[j].h = eh[j].e = PSW_NEG_INF;
	}

	sv = (int32_t*)kmalloc(km, (size_t)n_col * sizeof(int32_t));
	if (sv == 0) {
		if (z) kfree(km, z);
		if (off) kfree(km, off);
		kfree(km, eh);
		return;
	}

	for (i = 0; i < tlen; ++i) {
		int32_t f, h1, st, en, max = PSW_NEG_INF;
		const int16_t *tfi = dp->tf + (size_t)i * dp->mpad;
		int32_t go_e = dp->go_t[i], ge_e = dp->ge_t[i];
		const int32_t *goq;
		const int32_t *geq;

		if (i == 0) ht0 = -(go_e + ge_e);
		else        ht0 -= ge_e;

		st = i > w ? i - w : 0;
		en = i + w < qlen - 1 ? i + w : qlen - 1;
		h1 = st > 0 ? PSW_NEG_INF : ht0;
		f  = st > 0 ? PSW_NEG_INF : ht0 - (dp->go_q[0] + dp->ge_q[0]);
		goq = dp->go_q + st;
		geq = dp->ge_q + st;

		/* loop fission like ksw2_extz2_sse: score pass then DP pass */
		psw_fill_row_scores_sse(dp->qp, tfi, dp->mpad, dp->scale_shift, st, en, sv);

		if (!with_cigar) {
			for (j = st; j <= en; ++j, ++goq, ++geq) {
				eh_t *p = &eh[j];
				int32_t go_f = *goq, ge_f = *geq;
				int32_t h = p->h, e = p->e;
				int32_t s = sv[j - st];

				p->h = h1;
				h += s;
				h = h >= e ? h : e;
				h = h >= f ? h : f;
				h1 = h;
				max_j = max > h ? max_j : j;
				max   = max > h ? max   : h;

				{
					int32_t h_open_e = h1 - (go_e + ge_e);
					e -= ge_e;
					e = e > h_open_e ? e : h_open_e;
					p->e = e;
				}
				{
					int32_t h_open_f = h1 - (go_f + ge_f);
					f -= ge_f;
					f = f > h_open_f ? f : h_open_f;
				}
			}
		} else {
			uint8_t *zi = &z[(size_t)i * n_col];
			off[i] = st;
			for (j = st; j <= en; ++j, ++goq, ++geq) {
				eh_t *p = &eh[j];
				int32_t go_f = *goq, ge_f = *geq;
				int32_t h = p->h, e = p->e;
				int32_t s = sv[j - st];
				uint8_t d;

				p->h = h1;
				h += s;
				d = h >= e ? 0 : 1;
				h = h >= e ? h : e;
				d = h >= f ? d : 2;
				h = h >= f ? h : f;
				h1 = h;
				max_j = max > h ? max_j : j;
				max   = max > h ? max   : h;

				{
					int32_t h_open_e = h1 - (go_e + ge_e);
					e -= ge_e;
					d |= e > h_open_e ? 0x08 : 0;
					e = e > h_open_e ? e : h_open_e;
					p->e = e;
				}
				{
					int32_t h_open_f = h1 - (go_f + ge_f);
					f -= ge_f;
					d |= f > h_open_f ? 0x10 : 0;
					f = f > h_open_f ? f : h_open_f;
				}
				zi[j - st] = d;
			}
		}
		eh[j].h = h1;
		eh[j].e = PSW_NEG_INF;

		if (en == qlen - 1 && eh[qlen].h > ez->mqe) {
			ez->mqe = eh[qlen].h;
			ez->mqe_t = i;
		}
		if (i == tlen - 1) {
			ez->mte = max;
			ez->mte_q = max_j;
		}
		if (psw_apply_zdrop_scaled(ez, 0, max, i, max_j, zdrop_i32, gape_i32)) break;
		if (i == tlen - 1 && en == qlen - 1) {
			ez->score = eh[qlen].h;
			ez->reach_end = 1;
		}
	}

	if ((int32_t)ez->max > 0) ez->max = (uint32_t)psw_scaled_to_int((int32_t)ez->max, dp->scale);
	else ez->max = 0;
	ez->mqe = ez->mqe == PSW_NEG_INF ? PSW_NEG_INF : psw_scaled_to_int(ez->mqe, dp->scale);
	ez->mte = ez->mte == PSW_NEG_INF ? PSW_NEG_INF : psw_scaled_to_int(ez->mte, dp->scale);
	ez->score = ez->score == PSW_NEG_INF ? PSW_NEG_INF : psw_scaled_to_int(ez->score, dp->scale);

	if (with_cigar) {
		int rev_cigar = !!(flag & PSW_FLAG_REV_CIGAR);
		if (!ez->zdropped && (flag & PSW_FLAG_GLOBAL) && ez->reach_end) {
			psw_backtrack(km, 0, rev_cigar, 0, z, off, 0, n_col, tlen - 1, qlen - 1,
			              &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		} else if (ez->max_t >= 0 && ez->max_q >= 0) {
			psw_backtrack(km, 0, rev_cigar, 0, z, off, 0, n_col, ez->max_t, ez->max_q,
			              &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		}
	}

	if (sv) kfree(km, sv);
	if (z) kfree(km, z);
	if (off) kfree(km, off);
	kfree(km, eh);
}

void psw_extz_sse_pp(void *km, int qlen, const psw_prof_t *query,
                     int tlen, const psw_prof_t *target,
                     int8_t m, const int8_t *mat,
                     int8_t gapo, int8_t gape, int w, int zdrop, int flag, psw_extz_t *ez)
{
	psw_prof_t query_n = {0, 0, 0, 0}, target_n = {0, 0, 0, 0};
	const psw_prof_t *q_use, *t_use;
	int16_t *qbf = 0, *tbf = 0;
	psw_sse_dp_t dp;
	int j;

	dp.qp = 0; dp.tf = 0; dp.go_q = 0; dp.ge_q = 0; dp.go_t = 0; dp.ge_t = 0;
	dp.scale = 0; dp.scale_shift = -1; dp.mpad = 0;
	if (ez == 0) return;
	psw_reset_extz(ez);

	if (gapo < 0 || gape < 0) return;
	if (query == 0 || target == 0 || mat == 0) return;
	if (query->prof == 0 || target->prof == 0) return;
	if (qlen < 0 || tlen < 0 || m <= 0) return;
	if (query->len < qlen || target->len < tlen) return;
	if (query->dim < m || target->dim < m) return;

	dp.scale = psw_pick_safe_scale_pow2(m, mat, gapo, gape);
	if (dp.scale < 2) return;
	dp.scale_shift = psw_scale_to_shift(dp.scale);
	if (dp.scale_shift <= 0) return;
	dp.mpad = psw_roundup8(m);

	if (!psw_make_norm_prof(km, query, &query_n, m, dp.scale)) return;
	if (!psw_make_norm_prof(km, target, &target_n, m, dp.scale)) {
		psw_free_norm_prof(km, &query_n);
		return;
	}
	q_use = &query_n;
	t_use = &target_n;

	dp.qp = psw_gen_qp_i16_pad(km, qlen, q_use, m, dp.mpad, mat);
	dp.tf = psw_gen_tf_i16_pad(km, tlen, t_use, m, dp.mpad);
	qbf = psw_gen_base_freq_i16(km, qlen, q_use, m, dp.scale);
	tbf = psw_gen_base_freq_i16(km, tlen, t_use, m, dp.scale);
	dp.go_q = (int32_t*)kmalloc(km, (size_t)qlen * sizeof(int32_t));
	dp.ge_q = (int32_t*)kmalloc(km, (size_t)qlen * sizeof(int32_t));
	dp.go_t = (int32_t*)kmalloc(km, (size_t)tlen * sizeof(int32_t));
	dp.ge_t = (int32_t*)kmalloc(km, (size_t)tlen * sizeof(int32_t));
	if (dp.qp == 0 || dp.tf == 0 || qbf == 0 || tbf == 0 || dp.go_q == 0 || dp.ge_q == 0 || dp.go_t == 0 || dp.ge_t == 0) {
		if (dp.qp) kfree(km, dp.qp);
		if (dp.tf) kfree(km, dp.tf);
		if (qbf) kfree(km, qbf);
		if (tbf) kfree(km, tbf);
		if (dp.go_q) kfree(km, dp.go_q);
		if (dp.ge_q) kfree(km, dp.ge_q);
		if (dp.go_t) kfree(km, dp.go_t);
		if (dp.ge_t) kfree(km, dp.ge_t);
		psw_free_norm_prof(km, &query_n);
		psw_free_norm_prof(km, &target_n);
		return;
	}

	for (j = 0; j < qlen; ++j) {
		dp.go_q[j] = (int32_t)gapo * (int32_t)qbf[j];
		dp.ge_q[j] = (int32_t)gape * (int32_t)qbf[j];
	}
	for (j = 0; j < tlen; ++j) {
		dp.go_t[j] = (int32_t)gapo * (int32_t)tbf[j];
		dp.ge_t[j] = (int32_t)gape * (int32_t)tbf[j];
	}

	psw_extz_sse_core(km, qlen, tlen, &dp, w, zdrop, flag, ez);

	kfree(km, dp.qp);
	kfree(km, dp.tf);
	kfree(km, qbf);
	kfree(km, tbf);
	kfree(km, dp.go_q);
	kfree(km, dp.ge_q);
	kfree(km, dp.go_t);
	kfree(km, dp.ge_t);
	psw_free_norm_prof(km, &query_n);
	psw_free_norm_prof(km, &target_n);
}

void psw_extz_sse_ps(void *km, int qlen, const uint8_t *query,
                     int tlen, const psw_prof_t *target,
                     int8_t m, const int8_t *mat,
                     int8_t gapo, int8_t gape, int w, int zdrop, int flag, psw_extz_t *ez)
{
	psw_prof_t target_n = {0, 0, 0, 0};
	const psw_prof_t *t_use;
	int16_t *tbf = 0;
	psw_sse_dp_t dp;
	int i;

	dp.qp = 0; dp.tf = 0; dp.go_q = 0; dp.ge_q = 0; dp.go_t = 0; dp.ge_t = 0;
	dp.scale = 0; dp.scale_shift = -1; dp.mpad = 0;
	if (ez == 0) return;
	psw_reset_extz(ez);

	if (gapo < 0 || gape < 0) return;
	if (query == 0 || target == 0 || mat == 0) return;
	if (target->prof == 0) return;
	if (qlen < 0 || tlen < 0 || m <= 0) return;
	if (target->len < tlen || target->dim < m) return;

	for (i = 0; i < qlen; ++i)
		if ((int)query[i] >= m) return;

	dp.scale = psw_pick_safe_scale_pow2(m, mat, gapo, gape);
	if (dp.scale < 2) return;
	dp.scale_shift = psw_scale_to_shift(dp.scale);
	if (dp.scale_shift <= 0) return;
	dp.mpad = psw_roundup8(m);

	if (!psw_make_norm_prof(km, target, &target_n, m, dp.scale)) return;
	t_use = &target_n;

	dp.qp = psw_gen_qp_from_seq_i16_pad(km, qlen, query, m, dp.mpad, mat, dp.scale);
	dp.tf = psw_gen_tf_i16_pad(km, tlen, t_use, m, dp.mpad);
	tbf = psw_gen_base_freq_i16(km, tlen, t_use, m, dp.scale);
	dp.go_q = (int32_t*)kmalloc(km, (size_t)qlen * sizeof(int32_t));
	dp.ge_q = (int32_t*)kmalloc(km, (size_t)qlen * sizeof(int32_t));
	dp.go_t = (int32_t*)kmalloc(km, (size_t)tlen * sizeof(int32_t));
	dp.ge_t = (int32_t*)kmalloc(km, (size_t)tlen * sizeof(int32_t));
	if (dp.qp == 0 || dp.tf == 0 || tbf == 0 || dp.go_q == 0 || dp.ge_q == 0 || dp.go_t == 0 || dp.ge_t == 0) {
		if (dp.qp) kfree(km, dp.qp);
		if (dp.tf) kfree(km, dp.tf);
		if (tbf) kfree(km, tbf);
		if (dp.go_q) kfree(km, dp.go_q);
		if (dp.ge_q) kfree(km, dp.ge_q);
		if (dp.go_t) kfree(km, dp.go_t);
		if (dp.ge_t) kfree(km, dp.ge_t);
		psw_free_norm_prof(km, &target_n);
		return;
	}

	for (i = 0; i < qlen; ++i) {
		dp.go_q[i] = (int32_t)gapo * (int32_t)dp.scale;
		dp.ge_q[i] = (int32_t)gape * (int32_t)dp.scale;
	}
	for (i = 0; i < tlen; ++i) {
		dp.go_t[i] = (int32_t)gapo * (int32_t)tbf[i];
		dp.ge_t[i] = (int32_t)gape * (int32_t)tbf[i];
	}

	psw_extz_sse_core(km, qlen, tlen, &dp, w, zdrop, flag, ez);

	kfree(km, dp.qp);
	kfree(km, dp.tf);
	kfree(km, tbf);
	kfree(km, dp.go_q);
	kfree(km, dp.ge_q);
	kfree(km, dp.go_t);
	kfree(km, dp.ge_t);
	psw_free_norm_prof(km, &target_n);
}

#else

void psw_extz_sse_pp(void *km, int qlen, const psw_prof_t *query,
                     int tlen, const psw_prof_t *target,
                     int8_t m, const int8_t *mat,
                     int8_t gapo, int8_t gape, int w, int zdrop, int flag, psw_extz_t *ez)
{
	psw_extz(km, qlen, query, tlen, target, m, mat, gapo, gape, w, zdrop, flag, ez);
}

void psw_extz_sse_ps(void *km, int qlen, const uint8_t *query,
                     int tlen, const psw_prof_t *target,
                     int8_t m, const int8_t *mat,
                     int8_t gapo, int8_t gape, int w, int zdrop, int flag, psw_extz_t *ez)
{
	psw_extz_ps(km, qlen, query, tlen, target, m, mat, gapo, gape, w, zdrop, flag, ez);
}

#endif

