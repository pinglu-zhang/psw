#include <limits.h>
#include <stdint.h>
#include "psw.h"

typedef struct { int32_t h, e; } eh_t;

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

static inline int16_t psw_dot_scaled(const int16_t *x, const int16_t *y, int m, int8_t scale_shift)
{
	int32_t acc = 0;
	int b;

	if (scale_shift <= 0) return 0;
	for (b = 0; b < m; ++b)
		acc += (int32_t)x[b] * (int32_t)y[b];
	return psw_sat16(acc >> scale_shift);
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

static inline int16_t *psw_gen_qp_i16(void *km, int qlen, const psw_prof_t *query, int8_t m, const int8_t *mat)
{
	int a, b, j;
	int16_t *qp;

	qp = (int16_t*)kmalloc(km, (size_t)qlen * m * sizeof(int16_t));
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
	int16_t *tf;

	tf = (int16_t*)kmalloc(km, (size_t)tlen * m * sizeof(int16_t));
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

	tp = (int16_t*)kmalloc(km, (size_t)m * tlen * sizeof(int16_t));
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

void psw_extz_pp(void *km, int qlen, const psw_prof_t *query,
              int tlen, const psw_prof_t *target,
              int8_t m, const int8_t *mat,
              int8_t gapo, int8_t gape, int w, int zdrop, int flag, psw_extz_t *ez)
{
	eh_t *eh;
	int16_t *qp, *tf;
	int16_t *qbf, *tbf;
	int32_t *go_q, *ge_q, *go_t, *ge_t;
	psw_prof_t query_n = {0, 0, 0, 0}, target_n = {0, 0, 0, 0};
	const psw_prof_t *q_use, *t_use;
	int32_t i, j, max_j = 0, n_col, *off = 0;
	int32_t ht0 = 0;
	int32_t zdrop_i32, gape_i32;
	int with_cigar;
	uint8_t *z = 0;
	int16_t scale;
	int8_t scale_shift;

	if (ez == 0) return;
	psw_reset_extz(ez);
	with_cigar = !(flag & PSW_FLAG_SCORE_ONLY);

	if (gapo < 0 || gape < 0) return;
	if (query == 0 || target == 0 || mat == 0) return;
	if (query->prof == 0 || target->prof == 0) return;
	if (qlen < 0 || tlen < 0 || m <= 0) return;
	if (query->len < qlen || target->len < tlen) return;
	if (query->dim < m || target->dim < m) return;

	scale = psw_pick_safe_scale_pow2(m, mat, gapo, gape);
	if (scale < 2) return;
	scale_shift = psw_scale_to_shift(scale);
	if (scale_shift <= 0) return;
	if (!psw_make_norm_prof(km, query, &query_n, m, scale)) return;
	if (!psw_make_norm_prof(km, target, &target_n, m, scale)) {
		psw_free_norm_prof(km, &query_n);
		return;
	}
	q_use = &query_n;
	t_use = &target_n;

	if (w < 0) w = tlen > qlen ? tlen : qlen;
	n_col = qlen < 2 * w + 1 ? qlen : 2 * w + 1;
	zdrop_i32 = zdrop < 0 ? -1 : zdrop * (int32_t)scale;
	gape_i32 = (int32_t)gape * (int32_t)scale;

	qp = psw_gen_qp_i16(km, qlen, q_use, m, mat);
	if (qp == 0) {
		psw_free_norm_prof(km, &query_n);
		psw_free_norm_prof(km, &target_n);
		return;
	}
	tf = psw_gen_tf_i16(km, tlen, t_use, m);
	if (tf == 0) {
		kfree(km, qp);
		psw_free_norm_prof(km, &query_n);
		psw_free_norm_prof(km, &target_n);
		return;
	}
	qbf = psw_gen_base_freq_i16(km, qlen, q_use, m, scale);
	if (qbf == 0) {
		kfree(km, qp);
		kfree(km, tf);
		psw_free_norm_prof(km, &query_n);
		psw_free_norm_prof(km, &target_n);
		return;
	}
	tbf = psw_gen_base_freq_i16(km, tlen, t_use, m, scale);
	if (tbf == 0) {
		kfree(km, qp);
		kfree(km, tf);
		kfree(km, qbf);
		psw_free_norm_prof(km, &query_n);
		psw_free_norm_prof(km, &target_n);
		return;
	}

	go_q = (int32_t*)kmalloc(km, (size_t)qlen * sizeof(int32_t));
	ge_q = (int32_t*)kmalloc(km, (size_t)qlen * sizeof(int32_t));
	go_t = (int32_t*)kmalloc(km, (size_t)tlen * sizeof(int32_t));
	ge_t = (int32_t*)kmalloc(km, (size_t)tlen * sizeof(int32_t));
	eh = (eh_t*)kcalloc(km, qlen + 1, sizeof(eh_t));
	if (go_q == 0 || ge_q == 0 || go_t == 0 || ge_t == 0 || eh == 0) {
		if (go_q) kfree(km, go_q);
		if (ge_q) kfree(km, ge_q);
		if (go_t) kfree(km, go_t);
		if (ge_t) kfree(km, ge_t);
		if (eh) kfree(km, eh);
		kfree(km, qp);
		kfree(km, tf);
		kfree(km, qbf);
		kfree(km, tbf);
		psw_free_norm_prof(km, &query_n);
		psw_free_norm_prof(km, &target_n);
		return;
	}

	for (j = 0; j < qlen; ++j) {
		go_q[j] = (int32_t)gapo * (int32_t)qbf[j];
		ge_q[j] = (int32_t)gape * (int32_t)qbf[j];
	}
	for (i = 0; i < tlen; ++i) {
		go_t[i] = (int32_t)gapo * (int32_t)tbf[i];
		ge_t[i] = (int32_t)gape * (int32_t)tbf[i];
	}

	if (with_cigar) {
		z = (uint8_t*)kmalloc(km, (size_t)n_col * tlen);
		off = (int32_t*)kcalloc(km, tlen, sizeof(int32_t));
		if (z == 0 || off == 0) {
			if (z) kfree(km, z);
			if (off) kfree(km, off);
			kfree(km, go_q);
			kfree(km, ge_q);
			kfree(km, go_t);
			kfree(km, ge_t);
			kfree(km, eh);
			kfree(km, qp);
			kfree(km, tf);
			kfree(km, qbf);
			kfree(km, tbf);
			psw_free_norm_prof(km, &query_n);
			psw_free_norm_prof(km, &target_n);
			return;
		}
	}

	if (qlen == 0 || tlen == 0) {
		int32_t s32 = 0;
		if (qlen == 0) {
			for (i = 0; i < tlen; ++i)
				s32 -= i == 0 ? go_t[i] + ge_t[i] : ge_t[i];
		} else {
			for (j = 0; j < qlen; ++j)
				s32 -= j == 0 ? go_q[j] + ge_q[j] : ge_q[j];
		}
		ez->score = psw_scaled_to_int(s32, scale);
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
		kfree(km, go_q);
		kfree(km, ge_q);
		kfree(km, go_t);
		kfree(km, ge_t);
		kfree(km, eh);
		kfree(km, qp);
		kfree(km, tf);
		kfree(km, qbf);
		kfree(km, tbf);
		psw_free_norm_prof(km, &query_n);
		psw_free_norm_prof(km, &target_n);
		return;
	}

	{
		int32_t hqj = 0;
		int32_t go_t0 = go_t[0], ge_t0 = ge_t[0];
		eh[0].h = 0;
		eh[0].e = -(go_t0 + ge_t0);
		for (j = 1; j <= qlen && j <= w; ++j) {
			if (j == 1) hqj = -(go_q[j - 1] + ge_q[j - 1]);
			else        hqj -= ge_q[j - 1];
			eh[j].h = hqj;
			eh[j].e = hqj - (go_t0 + ge_t0);
		}
		for (; j <= qlen; ++j)
			eh[j].h = eh[j].e = PSW_NEG_INF;
	}

	for (i = 0; i < tlen; ++i) {
		int32_t f, h1, st, en, max = PSW_NEG_INF;
		const int16_t *tfi = tf + (size_t)i * m;
		int32_t go_e = go_t[i], ge_e = ge_t[i];

		if (i == 0) ht0 = -(go_e + ge_e);
		else        ht0 -= ge_e;

		st = i > w ? i - w : 0;
		en = i + w < qlen - 1 ? i + w : qlen - 1;
		h1 = st > 0 ? PSW_NEG_INF : ht0;
		f  = st > 0 ? PSW_NEG_INF : ht0 - (go_q[0] + ge_q[0]);

		if (!with_cigar) {
			for (j = st; j <= en; ++j) {
				eh_t *p = &eh[j];
				const int16_t *qpj = qp + (size_t)j * m;
				int32_t go_f = go_q[j], ge_f = ge_q[j];
				int32_t h = p->h, e = p->e;
				int32_t s = (int32_t)psw_dot_scaled(qpj, tfi, m, scale_shift);

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
			for (j = st; j <= en; ++j) {
				eh_t *p = &eh[j];
				const int16_t *qpj = qp + (size_t)j * m;
				int32_t go_f = go_q[j], ge_f = ge_q[j];
				int32_t h = p->h, e = p->e;
				int32_t s = (int32_t)psw_dot_scaled(qpj, tfi, m, scale_shift);
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
	if ((int32_t)ez->max > 0) ez->max = (uint32_t)psw_scaled_to_int((int32_t)ez->max, scale);
	else ez->max = 0;
	ez->mqe = ez->mqe == PSW_NEG_INF ? PSW_NEG_INF : psw_scaled_to_int(ez->mqe, scale);
	ez->mte = ez->mte == PSW_NEG_INF ? PSW_NEG_INF : psw_scaled_to_int(ez->mte, scale);
	ez->score = ez->score == PSW_NEG_INF ? PSW_NEG_INF : psw_scaled_to_int(ez->score, scale);

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

	if (z) kfree(km, z);
	if (off) kfree(km, off);
	kfree(km, go_q);
	kfree(km, ge_q);
	kfree(km, go_t);
	kfree(km, ge_t);
	kfree(km, eh);
	kfree(km, qp);
	kfree(km, tf);
	kfree(km, qbf);
	kfree(km, tbf);
	psw_free_norm_prof(km, &query_n);
	psw_free_norm_prof(km, &target_n);
}

void psw_extz_ps(void *km, int qlen, const uint8_t *query,
                 int tlen, const psw_prof_t *target,
                 int8_t m, const int8_t *mat,
                 int8_t gapo, int8_t gape, int w, int zdrop, int flag, psw_extz_t *ez)
{
	eh_t *eh;
	int16_t *tp, *tbf;
	int32_t *go_t, *ge_t;
	psw_prof_t target_n = {0, 0, 0, 0};
	const psw_prof_t *t_use;
	int32_t i, j, max_j = 0, n_col, *off = 0;
	int32_t ht0 = 0;
	int32_t zdrop_i32, gape_i32;
	int32_t go_q, ge_q;
	int with_cigar;
	uint8_t *z = 0;
	int16_t scale;

	if (ez == 0) return;
	psw_reset_extz(ez);
	with_cigar = !(flag & PSW_FLAG_SCORE_ONLY);

	if (gapo < 0 || gape < 0) return;
	if (query == 0 || target == 0 || mat == 0) return;
	if (target->prof == 0) return;
	if (qlen < 0 || tlen < 0 || m <= 0) return;
	if (target->len < tlen || target->dim < m) return;

	scale = psw_pick_safe_scale_pow2(m, mat, gapo, gape);
	if (scale < 2) return;
	go_q = (int32_t)gapo * scale;
	ge_q = (int32_t)gape * scale;

	for (j = 0; j < qlen; ++j)
		if ((int)query[j] < 0 || (int)query[j] >= m)
			return;

	if (!psw_make_norm_prof(km, target, &target_n, m, scale)) return;
	t_use = &target_n;

	if (w < 0) w = tlen > qlen ? tlen : qlen;
	n_col = qlen < 2 * w + 1 ? qlen : 2 * w + 1;
	zdrop_i32 = zdrop < 0 ? -1 : zdrop * (int32_t)scale;
	gape_i32 = (int32_t)gape * (int32_t)scale;

	tp = psw_gen_tp_i16(km, tlen, t_use, m, mat);
	if (tp == 0) {
		psw_free_norm_prof(km, &target_n);
		return;
	}
	tbf = psw_gen_base_freq_i16(km, tlen, t_use, m, scale);
	if (tbf == 0) {
		kfree(km, tp);
		psw_free_norm_prof(km, &target_n);
		return;
	}

	go_t = (int32_t*)kmalloc(km, (size_t)tlen * sizeof(int32_t));
	ge_t = (int32_t*)kmalloc(km, (size_t)tlen * sizeof(int32_t));
	eh = (eh_t*)kcalloc(km, qlen + 1, sizeof(eh_t));
	if (go_t == 0 || ge_t == 0 || eh == 0) {
		if (go_t) kfree(km, go_t);
		if (ge_t) kfree(km, ge_t);
		if (eh) kfree(km, eh);
		kfree(km, tp);
		kfree(km, tbf);
		psw_free_norm_prof(km, &target_n);
		return;
	}

	for (i = 0; i < tlen; ++i) {
		go_t[i] = (int32_t)gapo * (int32_t)tbf[i];
		ge_t[i] = (int32_t)gape * (int32_t)tbf[i];
	}

	if (with_cigar) {
		z = (uint8_t*)kmalloc(km, (size_t)n_col * tlen);
		off = (int32_t*)kcalloc(km, tlen, sizeof(int32_t));
		if (z == 0 || off == 0) {
			if (z) kfree(km, z);
			if (off) kfree(km, off);
			kfree(km, go_t);
			kfree(km, ge_t);
			kfree(km, eh);
			kfree(km, tp);
			kfree(km, tbf);
			psw_free_norm_prof(km, &target_n);
			return;
		}
	}

	if (qlen == 0 || tlen == 0) {
		int32_t s32 = 0;
		if (qlen == 0) {
			for (i = 0; i < tlen; ++i)
				s32 -= i == 0 ? go_t[i] + ge_t[i] : ge_t[i];
		} else {
			s32 = -((int32_t)go_q + (int32_t)qlen * ge_q);
		}
		ez->score = psw_scaled_to_int(s32, scale);
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
		kfree(km, go_t);
		kfree(km, ge_t);
		kfree(km, eh);
		kfree(km, tp);
		kfree(km, tbf);
		psw_free_norm_prof(km, &target_n);
		return;
	}

	{
		int32_t hqj = 0;
		int32_t go_t0 = go_t[0], ge_t0 = ge_t[0];
		eh[0].h = 0;
		eh[0].e = -(go_t0 + ge_t0);
		for (j = 1; j <= qlen && j <= w; ++j) {
			if (j == 1) hqj = -(go_q + ge_q);
			else        hqj -= ge_q;
			eh[j].h = hqj;
			eh[j].e = hqj - (go_t0 + ge_t0);
		}
		for (; j <= qlen; ++j)
			eh[j].h = eh[j].e = PSW_NEG_INF;
	}

	for (i = 0; i < tlen; ++i) {
		int32_t f, h1, st, en, max = PSW_NEG_INF;
		int32_t go_e = go_t[i], ge_e = ge_t[i];

		if (i == 0) ht0 = -(go_e + ge_e);
		else        ht0 -= ge_e;

		st = i > w ? i - w : 0;
		en = i + w < qlen - 1 ? i + w : qlen - 1;
		h1 = st > 0 ? PSW_NEG_INF : ht0;
		f  = st > 0 ? PSW_NEG_INF : ht0 - (go_q + ge_q);

		if (!with_cigar) {
			for (j = st; j <= en; ++j) {
				eh_t *p = &eh[j];
				int aidx = (int)query[j];
				int32_t go_f = go_q, ge_f = ge_q;
				int32_t h = p->h, e = p->e;
				int32_t s = tp[(size_t)aidx * tlen + i];

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
			for (j = st; j <= en; ++j) {
				eh_t *p = &eh[j];
				int aidx = (int)query[j];
				int32_t go_f = go_q, ge_f = ge_q;
				int32_t h = p->h, e = p->e;
				int32_t s = tp[(size_t)aidx * tlen + i];
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
	if ((int32_t)ez->max > 0) ez->max = (uint32_t)psw_scaled_to_int((int32_t)ez->max, scale);
	else ez->max = 0;
	ez->mqe = ez->mqe == PSW_NEG_INF ? PSW_NEG_INF : psw_scaled_to_int(ez->mqe, scale);
	ez->mte = ez->mte == PSW_NEG_INF ? PSW_NEG_INF : psw_scaled_to_int(ez->mte, scale);
	ez->score = ez->score == PSW_NEG_INF ? PSW_NEG_INF : psw_scaled_to_int(ez->score, scale);

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

	if (z) kfree(km, z);
	if (off) kfree(km, off);
	kfree(km, go_t);
	kfree(km, ge_t);
	kfree(km, eh);
	kfree(km, tp);
	kfree(km, tbf);
	psw_free_norm_prof(km, &target_n);
}

