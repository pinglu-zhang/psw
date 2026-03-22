#include <limits.h>
#include <stdint.h>

#include "psw.h"

#ifndef PSW_NEG_INF_F
#define PSW_NEG_INF_F (-1e30f)
#endif

#define PSW_SW_NEG_INF_I32 (-0x3f3f3f3f)
#define PSW_SW_CLIP_OP PSW_CIGAR_SOFTCLIP

typedef struct {
	int16_t u, v, x, y;
} uvxy16_t;

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
	else rounded = (scaled - den64 / 2) / den64;

	if (rounded > INT16_MAX) return INT16_MAX;
	if (rounded < INT16_MIN) return INT16_MIN;
	return (int16_t)rounded;
}

static inline int psw_make_norm_prof(void *km, const psw_prof_t *src, psw_prof_t *dst, int8_t m, int16_t scale)
{
	int i, a;
	int32_t depth;
	uint32_t *prof;

	if (src == 0 || dst == 0 || src->prof == 0 || src->len < 0 || src->dim <= 0 || m <= 0 || scale <= 0) return 0;
	if (src->dim < m) return 0;

	prof = (uint32_t *)kmalloc(km, (size_t)src->len * src->dim * sizeof(uint32_t));
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
		kfree(km, (void *)p->prof);
		p->prof = 0;
	}
}

static inline int16_t psw_dot_scaled(const int16_t *x, const int16_t *y, int m, int8_t scale_shift)
{
	int32_t acc = 0;
	int b;

	if (scale_shift <= 0) return 0;
	for (b = 0; b < m; ++b)
		acc += x[b] * y[b];
	return psw_sat16(acc >> scale_shift);
}

static inline int16_t *psw_gen_base_freq_i16(void *km, int len, const psw_prof_t *p, int8_t m, int16_t scale)
{
	int i, a;
	int16_t *bf;

	bf = (int16_t *)kmalloc(km, (size_t)len * sizeof(int16_t));
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

static inline int16_t *psw_gen_qp_i16(void *km, int qlen, const psw_prof_t *query, int8_t m, const int8_t *mat)
{
	int a, b, j;
	int16_t *qp;

	qp = (int16_t *)kmalloc(km, (size_t)qlen * m * sizeof(int16_t));
	if (qp == 0) return 0;

	for (j = 0; j < qlen; ++j) {
		const uint32_t *qcol = query->prof + (size_t)j * query->dim;
		int16_t *dst = qp + (size_t)j * m;
		for (b = 0; b < m; ++b) {
			int32_t s = 0;
			for (a = 0; a < m; ++a)
				s += (int32_t)qcol[a] * (int32_t)mat[a * m + b];
			dst[b] = psw_sat16(s);
		}
	}
	return qp;
}

static inline int16_t *psw_gen_tf_i16(void *km, int tlen, const psw_prof_t *target, int8_t m)
{
	int i, b;
	int16_t *tf = (int16_t *)kmalloc(km, (size_t)tlen * m * sizeof(int16_t));
	if (tf == 0) return 0;

	for (i = 0; i < tlen; ++i) {
		const uint32_t *tcol = target->prof + (size_t)i * target->dim;
		int16_t *dst = tf + (size_t)i * m;
		for (b = 0; b < m; ++b)
			dst[b] = psw_sat16((int32_t)tcol[b]);
	}
	return tf;
}

static inline int16_t *psw_gen_tp_i16(void *km, int tlen, const psw_prof_t *target,
                                      int8_t m, const int8_t *mat)
{
	int a, b, i;
	int16_t *tp;

	tp = (int16_t *)kmalloc(km, (size_t)m * tlen * sizeof(int16_t));
	if (tp == 0) return 0;

	for (i = 0; i < tlen; ++i) {
		const uint32_t *tcol = target->prof + (size_t)i * target->dim;
		for (a = 0; a < m; ++a) {
			int32_t s = 0;
			for (b = 0; b < m; ++b)
				s += (int32_t)mat[a * m + b] * (int32_t)tcol[b];
			tp[(size_t)a * tlen + i] = psw_sat16(s);
		}
	}
	return tp;
}

static inline int psw_sw_in_band(int i, int j, int w)
{
	if (w < 0) return 1;
	return j >= i - w && j <= i + w;
}

static inline uint32_t *psw_sw_finish_cigar(void *km, uint32_t *cigar, int n_cigar)
{
	int i;
	for (i = 0; i < (n_cigar >> 1); ++i) {
		uint32_t tmp = cigar[i];
		cigar[i] = cigar[n_cigar - 1 - i];
		cigar[n_cigar - 1 - i] = tmp;
	}
	return cigar;
}

static inline uint32_t *psw_sw_backtrack(void *km, int qlen,
                                         int best_i, int best_j,
                                         const uint8_t *trace,
                                         int *m_cigar_, int *n_cigar_, uint32_t *cigar)
{
	int i = best_i, j = best_j;
	int state = 0; /* 0=H, 1=E(del), 2=F(ins) */
	int n_cigar = 0;
	int m_cigar = *m_cigar_;

	/* Local alignment ends at best_j; query suffix is clipped. */
	if (best_j < qlen)
		cigar = psw_push_cigar(km, &n_cigar, &m_cigar, cigar, PSW_SW_CLIP_OP, qlen - best_j);

	while (i > 0 && j > 0) {
		int idx = (i - 1) * qlen + (j - 1);
		uint8_t tr = trace[idx];

		if (state == 0) {
			int src = tr & 0x03;
			if (src == 0) break;
			if (src == 1) {
				cigar = psw_push_cigar(km, &n_cigar, &m_cigar, cigar, PSW_CIGAR_MATCH, 1);
				--i;
				--j;
			} else if (src == 2) state = 1;
			else state = 2;
		} else if (state == 1) {
			cigar = psw_push_cigar(km, &n_cigar, &m_cigar, cigar, PSW_CIGAR_DEL, 1);
			--i;
			if ((tr & 0x04) == 0) state = 0;
		} else {
			cigar = psw_push_cigar(km, &n_cigar, &m_cigar, cigar, PSW_CIGAR_INS, 1);
			--j;
			if ((tr & 0x08) == 0) state = 0;
		}
	}

	/* Local alignment starts at j; query prefix is clipped. */
	if (j > 0)
		cigar = psw_push_cigar(km, &n_cigar, &m_cigar, cigar, PSW_SW_CLIP_OP, j);

	cigar = psw_sw_finish_cigar(km, cigar, n_cigar);
	*n_cigar_ = n_cigar;
	*m_cigar_ = m_cigar;
	return cigar;
}

float psw_sw_pp(void *km, int qlen, const psw_prof_t *query,
                int tlen, const psw_prof_t *target,
                int8_t m, const int8_t *mat,
                int8_t gapo, int8_t gape, int w,
                int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{
	int i, j;
	int16_t *qp = 0, *tf = 0, *qbf = 0, *tbf = 0;
	int16_t *go_q = 0, *ge_q = 0, *go_t = 0, *ge_t = 0, *go_ge_q = 0, *go_ge_t = 0;
	int32_t *h_prev = 0, *h_cur = 0, *e_col = 0;
	uint8_t *trace = 0;
	int32_t best = 0;
	int best_i = 0, best_j = 0;
	int16_t scale;
	int8_t scale_shift;
	psw_prof_t query_n = {0, 0, 0, 0}, target_n = {0, 0, 0, 0};
	const psw_prof_t *q_use, *t_use;
	float ret;

	if (gapo < 0 || gape < 0) return PSW_NEG_INF_F;
	if (query == 0 || target == 0 || mat == 0) return PSW_NEG_INF_F;
	if (query->prof == 0 || target->prof == 0) return PSW_NEG_INF_F;
	if (qlen < 0 || tlen < 0 || m <= 0) return PSW_NEG_INF_F;
	if (query->len < qlen || target->len < tlen) return PSW_NEG_INF_F;
	if (query->dim < m || target->dim < m) return PSW_NEG_INF_F;
	if (m_cigar_ && n_cigar_ && cigar_) *n_cigar_ = 0;
	if (qlen == 0 || tlen == 0) return 0.0f;

	scale = psw_pick_safe_scale_pow2(m, mat, gapo, gape);
	if (scale < 2) return PSW_NEG_INF_F;
	scale_shift = psw_scale_to_shift(scale);
	if (scale_shift <= 0) return PSW_NEG_INF_F;
	if (!psw_make_norm_prof(km, query, &query_n, m, scale)) return PSW_NEG_INF_F;
	if (!psw_make_norm_prof(km, target, &target_n, m, scale)) {
		psw_free_norm_prof(km, &query_n);
		return PSW_NEG_INF_F;
	}
	q_use = &query_n;
	t_use = &target_n;

	qp = psw_gen_qp_i16(km, qlen, q_use, m, mat);
	tf = psw_gen_tf_i16(km, tlen, t_use, m);
	qbf = psw_gen_base_freq_i16(km, qlen, q_use, m, scale);
	tbf = psw_gen_base_freq_i16(km, tlen, t_use, m, scale);
	if (qp == 0 || tf == 0 || qbf == 0 || tbf == 0) {
		kfree(km, qp); kfree(km, tf); kfree(km, qbf); kfree(km, tbf);
		psw_free_norm_prof(km, &query_n); psw_free_norm_prof(km, &target_n);
		return PSW_NEG_INF_F;
	}

	go_q = (int16_t *)kmalloc(km, (size_t)qlen * sizeof(int16_t));
	ge_q = (int16_t *)kmalloc(km, (size_t)qlen * sizeof(int16_t));
	go_t = (int16_t *)kmalloc(km, (size_t)tlen * sizeof(int16_t));
	ge_t = (int16_t *)kmalloc(km, (size_t)tlen * sizeof(int16_t));
	go_ge_q = (int16_t *)kmalloc(km, (size_t)qlen * sizeof(int16_t));
	go_ge_t = (int16_t *)kmalloc(km, (size_t)tlen * sizeof(int16_t));
	h_prev = (int32_t *)kmalloc(km, (size_t)(qlen + 1) * sizeof(int32_t));
	h_cur = (int32_t *)kmalloc(km, (size_t)(qlen + 1) * sizeof(int32_t));
	e_col = (int32_t *)kmalloc(km, (size_t)(qlen + 1) * sizeof(int32_t));
	trace = (uint8_t *)kcalloc(km, (size_t)qlen * tlen, 1);
	if (go_q == 0 || ge_q == 0 || go_t == 0 || ge_t == 0 || go_ge_q == 0 || go_ge_t == 0 ||
	    h_prev == 0 || h_cur == 0 || e_col == 0 || trace == 0) {
		kfree(km, go_q); kfree(km, ge_q); kfree(km, go_t); kfree(km, ge_t);
		kfree(km, go_ge_q); kfree(km, go_ge_t);
		kfree(km, h_prev); kfree(km, h_cur); kfree(km, e_col);
		kfree(km, trace);
		kfree(km, qp); kfree(km, tf); kfree(km, qbf); kfree(km, tbf);
		psw_free_norm_prof(km, &query_n); psw_free_norm_prof(km, &target_n);
		return PSW_NEG_INF_F;
	}

	for (j = 0; j < qlen; ++j) {
		go_q[j] = psw_sat16((int32_t)gapo * qbf[j]);
		ge_q[j] = psw_sat16((int32_t)gape * qbf[j]);
		go_ge_q[j] = (int16_t)(go_q[j] + ge_q[j]);
	}
	for (i = 0; i < tlen; ++i) {
		go_t[i] = psw_sat16((int32_t)gapo * tbf[i]);
		ge_t[i] = psw_sat16((int32_t)gape * tbf[i]);
		go_ge_t[i] = (int16_t)(go_t[i] + ge_t[i]);
	}

	for (j = 0; j <= qlen; ++j) {
		h_prev[j] = 0;
		e_col[j] = PSW_SW_NEG_INF_I32;
	}

	for (i = 1; i <= tlen; ++i) {
		int32_t f = PSW_SW_NEG_INF_I32;
		h_cur[0] = 0;
		for (j = 1; j <= qlen; ++j) {
			int32_t e_open, e_ext, e;
			int32_t f_open, f_ext;
			int32_t diag, s, h;
			uint8_t tr = 0;

			if (!psw_sw_in_band(i, j, w)) {
				h_cur[j] = PSW_SW_NEG_INF_I32;
				e_col[j] = PSW_SW_NEG_INF_I32;
				f = PSW_SW_NEG_INF_I32;
				continue;
			}

			e_open = h_prev[j] <= PSW_SW_NEG_INF_I32 / 2 ? PSW_SW_NEG_INF_I32 : h_prev[j] - go_ge_t[i - 1];
			e_ext = e_col[j] <= PSW_SW_NEG_INF_I32 / 2 ? PSW_SW_NEG_INF_I32 : e_col[j] - ge_t[i - 1];
			if (e_open >= e_ext) {
				e = e_open;
			} else {
				e = e_ext;
				tr |= 0x04;
			}
			e_col[j] = e;

			f_open = h_cur[j - 1] <= PSW_SW_NEG_INF_I32 / 2 ? PSW_SW_NEG_INF_I32 : h_cur[j - 1] - go_ge_q[j - 1];
			f_ext = f <= PSW_SW_NEG_INF_I32 / 2 ? PSW_SW_NEG_INF_I32 : f - ge_q[j - 1];
			if (f_open >= f_ext) {
				f = f_open;
			} else {
				f = f_ext;
				tr |= 0x08;
			}

			diag = h_prev[j - 1];
			s = psw_dot_scaled(qp + (size_t)(j - 1) * m, tf + (size_t)(i - 1) * m, m, scale_shift);
			diag = diag <= PSW_SW_NEG_INF_I32 / 2 ? PSW_SW_NEG_INF_I32 : diag + s;

			h = 0;
			if (diag > h) {
				h = diag;
				tr = (uint8_t)((tr & 0x0c) | 1);
			}
			if (e > h) {
				h = e;
				tr = (uint8_t)((tr & 0x0c) | 2);
			}
			if (f > h) {
				h = f;
				tr = (uint8_t)((tr & 0x0c) | 3);
			}
			if (h <= 0) {
				h = 0;
				tr = 0;
			}

			h_cur[j] = h;
			trace[(size_t)(i - 1) * qlen + (j - 1)] = tr;
			if (h > best) {
				best = h;
				best_i = i;
				best_j = j;
			}
		}
		for (j = 0; j <= qlen; ++j)
			h_prev[j] = h_cur[j];
	}

	if (m_cigar_ && n_cigar_ && cigar_) {
		*n_cigar_ = 0;
		if (best > 0) {
			*cigar_ = psw_sw_backtrack(km, qlen, best_i, best_j, trace, m_cigar_, n_cigar_, *cigar_);
		} else if (qlen > 0) {
			int n_cigar = 0;
			int m_cigar = *m_cigar_;
			uint32_t *cigar = *cigar_;
			cigar = psw_push_cigar(km, &n_cigar, &m_cigar, cigar, PSW_SW_CLIP_OP, qlen);
			*cigar_ = cigar;
			*m_cigar_ = m_cigar;
			*n_cigar_ = n_cigar;
		}
	}

	ret = (float)best / (float)scale;
	kfree(km, go_q); kfree(km, ge_q); kfree(km, go_t); kfree(km, ge_t);
	kfree(km, go_ge_q); kfree(km, go_ge_t);
	kfree(km, h_prev); kfree(km, h_cur); kfree(km, e_col);
	kfree(km, trace);
	kfree(km, qp); kfree(km, tf); kfree(km, qbf); kfree(km, tbf);
	psw_free_norm_prof(km, &query_n); psw_free_norm_prof(km, &target_n);
	return ret;
}

float psw_sw_ps(void *km, int qlen, const uint8_t *query,
                int tlen, const psw_prof_t *target,
                int8_t m, const int8_t *mat,
                int8_t gapo, int8_t gape, int w,
                int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{
	int i, j;
	int16_t *tp = 0, *tbf = 0;
	int16_t *go_t = 0, *ge_t = 0, *go_ge_t = 0;
	int16_t go_q, ge_q, go_ge_q;
	int32_t *h_prev = 0, *h_cur = 0, *e_col = 0;
	uint8_t *trace = 0;
	int32_t best = 0;
	int best_i = 0, best_j = 0;
	int16_t scale;
	psw_prof_t target_n = {0, 0, 0, 0};
	const psw_prof_t *t_use;
	float ret;

	if (gapo < 0 || gape < 0) return PSW_NEG_INF_F;
	if (query == 0 || target == 0 || mat == 0) return PSW_NEG_INF_F;
	if (target->prof == 0) return PSW_NEG_INF_F;
	if (qlen < 0 || tlen < 0 || m <= 0) return PSW_NEG_INF_F;
	if (target->len < tlen || target->dim < m) return PSW_NEG_INF_F;
	if (m_cigar_ && n_cigar_ && cigar_) *n_cigar_ = 0;
	if (qlen == 0 || tlen == 0) return 0.0f;

	scale = psw_pick_safe_scale_pow2(m, mat, gapo, gape);
	if (scale < 2) return PSW_NEG_INF_F;
	go_q = psw_sat16((int32_t)gapo * scale);
	ge_q = psw_sat16((int32_t)gape * scale);
	go_ge_q = (int16_t)(go_q + ge_q);

	if (!psw_make_norm_prof(km, target, &target_n, m, scale)) return PSW_NEG_INF_F;
	t_use = &target_n;
	for (j = 0; j < qlen; ++j) {
		if ((int)query[j] < 0 || (int)query[j] >= m) {
			psw_free_norm_prof(km, &target_n);
			return PSW_NEG_INF_F;
		}
	}

	tp = psw_gen_tp_i16(km, tlen, t_use, m, mat);
	tbf = psw_gen_base_freq_i16(km, tlen, t_use, m, scale);
	if (tp == 0 || tbf == 0) {
		kfree(km, tp); kfree(km, tbf);
		psw_free_norm_prof(km, &target_n);
		return PSW_NEG_INF_F;
	}

	go_t = (int16_t *)kmalloc(km, (size_t)tlen * sizeof(int16_t));
	ge_t = (int16_t *)kmalloc(km, (size_t)tlen * sizeof(int16_t));
	go_ge_t = (int16_t *)kmalloc(km, (size_t)tlen * sizeof(int16_t));
	h_prev = (int32_t *)kmalloc(km, (size_t)(qlen + 1) * sizeof(int32_t));
	h_cur = (int32_t *)kmalloc(km, (size_t)(qlen + 1) * sizeof(int32_t));
	e_col = (int32_t *)kmalloc(km, (size_t)(qlen + 1) * sizeof(int32_t));
	trace = (uint8_t *)kcalloc(km, (size_t)qlen * tlen, 1);
	if (go_t == 0 || ge_t == 0 || go_ge_t == 0 || h_prev == 0 || h_cur == 0 || e_col == 0 || trace == 0) {
		kfree(km, go_t); kfree(km, ge_t); kfree(km, go_ge_t);
		kfree(km, h_prev); kfree(km, h_cur); kfree(km, e_col);
		kfree(km, trace);
		kfree(km, tp); kfree(km, tbf);
		psw_free_norm_prof(km, &target_n);
		return PSW_NEG_INF_F;
	}

	for (i = 0; i < tlen; ++i) {
		go_t[i] = psw_sat16((int32_t)gapo * tbf[i]);
		ge_t[i] = psw_sat16((int32_t)gape * tbf[i]);
		go_ge_t[i] = (int16_t)(go_t[i] + ge_t[i]);
	}

	for (j = 0; j <= qlen; ++j) {
		h_prev[j] = 0;
		e_col[j] = PSW_SW_NEG_INF_I32;
	}

	for (i = 1; i <= tlen; ++i) {
		int32_t f = PSW_SW_NEG_INF_I32;
		h_cur[0] = 0;
		for (j = 1; j <= qlen; ++j) {
			int32_t e_open, e_ext, e;
			int32_t f_open, f_ext;
			int32_t diag, s, h;
			uint8_t tr = 0;
			int aidx;

			if (!psw_sw_in_band(i, j, w)) {
				h_cur[j] = PSW_SW_NEG_INF_I32;
				e_col[j] = PSW_SW_NEG_INF_I32;
				f = PSW_SW_NEG_INF_I32;
				continue;
			}

			e_open = h_prev[j] <= PSW_SW_NEG_INF_I32 / 2 ? PSW_SW_NEG_INF_I32 : h_prev[j] - go_ge_t[i - 1];
			e_ext = e_col[j] <= PSW_SW_NEG_INF_I32 / 2 ? PSW_SW_NEG_INF_I32 : e_col[j] - ge_t[i - 1];
			if (e_open >= e_ext) {
				e = e_open;
			} else {
				e = e_ext;
				tr |= 0x04;
			}
			e_col[j] = e;

			f_open = h_cur[j - 1] <= PSW_SW_NEG_INF_I32 / 2 ? PSW_SW_NEG_INF_I32 : h_cur[j - 1] - go_ge_q;
			f_ext = f <= PSW_SW_NEG_INF_I32 / 2 ? PSW_SW_NEG_INF_I32 : f - ge_q;
			if (f_open >= f_ext) {
				f = f_open;
			} else {
				f = f_ext;
				tr |= 0x08;
			}

			aidx = (int)query[j - 1];
			s = tp[(size_t)aidx * tlen + (i - 1)];
			diag = h_prev[j - 1];
			diag = diag <= PSW_SW_NEG_INF_I32 / 2 ? PSW_SW_NEG_INF_I32 : diag + s;

			h = 0;
			if (diag > h) {
				h = diag;
				tr = (uint8_t)((tr & 0x0c) | 1);
			}
			if (e > h) {
				h = e;
				tr = (uint8_t)((tr & 0x0c) | 2);
			}
			if (f > h) {
				h = f;
				tr = (uint8_t)((tr & 0x0c) | 3);
			}
			if (h <= 0) {
				h = 0;
				tr = 0;
			}

			h_cur[j] = h;
			trace[(size_t)(i - 1) * qlen + (j - 1)] = tr;
			if (h > best) {
				best = h;
				best_i = i;
				best_j = j;
			}
		}
		for (j = 0; j <= qlen; ++j)
			h_prev[j] = h_cur[j];
	}

	if (m_cigar_ && n_cigar_ && cigar_) {
		*n_cigar_ = 0;
		if (best > 0) {
			*cigar_ = psw_sw_backtrack(km, qlen, best_i, best_j, trace, m_cigar_, n_cigar_, *cigar_);
		} else if (qlen > 0) {
			int n_cigar = 0;
			int m_cigar = *m_cigar_;
			uint32_t *cigar = *cigar_;
			cigar = psw_push_cigar(km, &n_cigar, &m_cigar, cigar, PSW_SW_CLIP_OP, qlen);
			*cigar_ = cigar;
			*m_cigar_ = m_cigar;
			*n_cigar_ = n_cigar;
		}
	}

	ret = (float)best / (float)scale;
	kfree(km, go_t); kfree(km, ge_t); kfree(km, go_ge_t);
	kfree(km, h_prev); kfree(km, h_cur); kfree(km, e_col);
	kfree(km, trace);
	kfree(km, tp); kfree(km, tbf);
	psw_free_norm_prof(km, &target_n);
	return ret;
}

