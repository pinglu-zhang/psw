#include <stdint.h>
#include "psw.h"

#ifndef PSW_NEG_INF_F
#define PSW_NEG_INF_F (-1e30f)
#endif

typedef struct { int16_t u, v, x, y; } uvxy16_t;

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

static inline int16_t *psw_gen_tp_i16(void *km, int tlen, const psw_prof_t *target, int8_t m, const int8_t *mat)
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

static inline int32_t psw_gap_only_target_i16(const int16_t *go_t, const int16_t *ge_t, int tlen)
{
	int i;
	int32_t s = 0;
	for (i = 0; i < tlen; ++i)
		s -= (i == 0 ? (int32_t)go_t[i] + ge_t[i] : (int32_t)ge_t[i]);
	return s;
}

static inline int32_t psw_gap_only_query_i16(const int16_t *go_q, const int16_t *ge_q, int qlen)
{
	int j;
	int32_t s = 0;
	for (j = 0; j < qlen; ++j)
		s -= (j == 0 ? (int32_t)go_q[j] + ge_q[j] : (int32_t)ge_q[j]);
	return s;
}

static inline int32_t psw_gap_only_query_scalar_i16(int16_t go_q, int16_t ge_q, int qlen)
{
	if (qlen <= 0) return 0;
	return -((int32_t)go_q + (int32_t)qlen * ge_q);
}

static inline int32_t psw_unscale_i32(int32_t x, int16_t scale)
{
	if (x == PSW_NEG_INF) return PSW_NEG_INF;
	if (scale <= 1) return x;
	return x >= 0 ? (x + scale / 2) / scale : (x - scale / 2) / scale;
}

static inline int psw_apply_zdrop_raw(int32_t *max_raw, int *max_t, int *max_q,
                                      int32_t H, int r, int t,
                                      int32_t zdrop_scaled, int32_t e_scaled)
{
	int q = r - t;
	if (H > *max_raw) {
		*max_raw = H;
		*max_t = t;
		*max_q = q;
	} else if (t >= *max_t && q >= *max_q) {
		int tl = t - *max_t;
		int ql = q - *max_q;
		int l = tl > ql ? tl - ql : ql - tl;
		if (zdrop_scaled >= 0 && *max_raw - H > zdrop_scaled + (int32_t)l * e_scaled)
			return 1;
	}
	return 0;
}

static inline void psw_finalize_extz(psw_extz_t *ez, int16_t scale,
                                     int32_t max_raw, int max_t, int max_q,
                                     int32_t score_raw, int32_t mqe_raw, int mqe_t,
                                     int32_t mte_raw, int mte_q,
                                     int zdropped, int reach_end)
{
	ez->max = max_raw > 0 ? (uint32_t)psw_unscale_i32(max_raw, scale) : 0;
	ez->max_t = max_t;
	ez->max_q = max_q;
	ez->score = psw_unscale_i32(score_raw, scale);
	ez->mqe = psw_unscale_i32(mqe_raw, scale);
	ez->mte = psw_unscale_i32(mte_raw, scale);
	ez->mqe_t = mqe_t;
	ez->mte_q = mte_q;
	ez->zdropped = zdropped ? 1 : 0;
	ez->reach_end = reach_end ? 1 : 0;
}

void psw_extz_pp(void *km, int qlen, const psw_prof_t *query,
                 int tlen, const psw_prof_t *target,
                 int8_t m, const int8_t *mat,
                 int8_t gapo, int8_t gape, int w, int zdrop, int flag, psw_extz_t *ez)
{
	uvxy16_t *a = 0;
	int16_t *qp = 0, *tf = 0;
	int16_t *qbf = 0, *tbf = 0;
	int16_t *go_q = 0, *ge_q = 0, *go_t = 0, *ge_t = 0;
	psw_prof_t query_n = { 0, 0, 0, 0 }, target_n = { 0, 0, 0, 0 };
	const psw_prof_t *q_use, *t_use;
	int32_t r, t, n_col, *off = 0, *off_end = 0, *H = 0;
	uint8_t *z = 0;
	int16_t scale;
	int8_t scale_shift;
	int16_t igapo, igape;
	int with_cigar = !(flag & PSW_FLAG_SCORE_ONLY);
	int32_t max_raw = 0, score_raw = PSW_NEG_INF, mqe_raw = PSW_NEG_INF, mte_raw = PSW_NEG_INF;
	int max_t = -1, max_q = -1, mqe_t = -1, mte_q = -1;
	int32_t zdrop_scaled = zdrop < 0 ? -1 : (int32_t)zdrop;
	int zdropped = 0, reach_end = 0;
	int bt_t = -1, bt_q = -1;

	if (ez == 0) return;
	psw_reset_extz(ez);

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
	zdrop_scaled = zdrop < 0 ? -1 : zdrop_scaled * scale;
	if (!psw_make_norm_prof(km, query, &query_n, m, scale)) return;
	if (!psw_make_norm_prof(km, target, &target_n, m, scale)) {
		psw_free_norm_prof(km, &query_n);
		return;
	}
	q_use = &query_n;
	t_use = &target_n;

	do {
		if (w < 0) w = tlen > qlen ? tlen : qlen;
		n_col = w + 1 < tlen ? w + 1 : tlen;

		qp = psw_gen_qp_i16(km, qlen, q_use, m, mat);
		tf = psw_gen_tf_i16(km, tlen, t_use, m);
		qbf = psw_gen_base_freq_i16(km, qlen, q_use, m, scale);
		tbf = psw_gen_base_freq_i16(km, tlen, t_use, m, scale);
		if (qp == 0 || tf == 0 || qbf == 0 || tbf == 0) break;

		go_q = (int16_t *)kmalloc(km, (size_t)(qlen + 1) * sizeof(int16_t));
		ge_q = (int16_t *)kmalloc(km, (size_t)(qlen + 1) * sizeof(int16_t));
		go_t = (int16_t *)kmalloc(km, (size_t)(tlen + 1) * sizeof(int16_t));
		ge_t = (int16_t *)kmalloc(km, (size_t)(tlen + 1) * sizeof(int16_t));
		a = (uvxy16_t *)kmalloc(km, (size_t)(tlen + 1) * sizeof(uvxy16_t));
		H = (int32_t *)kmalloc(km, (size_t)(tlen > 0 ? tlen : 1) * sizeof(int32_t));
		if (go_q == 0 || ge_q == 0 || go_t == 0 || ge_t == 0 || a == 0 || H == 0) break;

		igapo = psw_sat16((int32_t)gapo * scale);
		igape = psw_sat16((int32_t)gape * scale);
		for (t = 0; t < qlen; ++t) {
			go_q[t] = psw_sat16((int32_t)gapo * qbf[t]);
			ge_q[t] = psw_sat16((int32_t)gape * qbf[t]);
		}
		go_q[qlen] = igapo;
		ge_q[qlen] = igape;
		for (t = 0; t < tlen; ++t) {
			go_t[t] = psw_sat16((int32_t)gapo * tbf[t]);
			ge_t[t] = psw_sat16((int32_t)gape * tbf[t]);
		}
		go_t[tlen] = igapo;
		ge_t[tlen] = igape;
		for (t = 0; t <= tlen; ++t)
			a[t].x = a[t].v = a[t].y = a[t].u = -(igapo + igape);
		for (t = 0; t < tlen; ++t) H[t] = PSW_NEG_INF;

		if (with_cigar) {
			z = (uint8_t *)kcalloc(km, (size_t)(qlen + tlen) * n_col, 1);
			off = (int32_t *)kmalloc(km, (size_t)(qlen + tlen) * sizeof(int32_t) * 2);
			if (z == 0 || off == 0) break;
			off_end = off + qlen + tlen;
		}

		if (qlen == 0 || tlen == 0) {
			int32_t s32 = qlen == 0 ? psw_gap_only_target_i16(go_t, ge_t, tlen)
				: psw_gap_only_query_i16(go_q, ge_q, qlen);
			score_raw = s32;
			if (qlen == 0) {
				mqe_raw = 0;
				reach_end = 1;
			}
			if (tlen == 0) mte_raw = 0;
			if (with_cigar) {
				ez->n_cigar = 0;
				if (qlen == 0 && tlen > 0)
					ez->cigar = psw_push_cigar(km, &ez->n_cigar, &ez->m_cigar, ez->cigar, PSW_CIGAR_DEL, tlen);
				else if (tlen == 0 && qlen > 0)
					ez->cigar = psw_push_cigar(km, &ez->n_cigar, &ez->m_cigar, ez->cigar, PSW_CIGAR_INS, qlen);
			}
			psw_finalize_extz(ez, scale, max_raw, max_t, max_q, score_raw, mqe_raw, mqe_t, mte_raw, mte_q, zdropped, reach_end);
			break;
		}

		for (r = 0; r < qlen + tlen - 1; ++r) {
			int32_t st = 0, en = tlen - 1;
			int16_t x1, v1;
			int32_t row_max = PSW_NEG_INF;
			int row_max_t = -1;

			if (st < r - qlen + 1) st = r - qlen + 1;
			if (en > r) en = r;
			if (st < (r - w + 1) >> 1) st = (r - w + 1) >> 1;
			if (en > (r + w) >> 1) en = (r + w) >> 1;
			if (st > en) {
				zdropped = 1;
				break;
			}

			if (st != 0) {
				if (r > st + st + w - 1) x1 = v1 = -(igapo + igape);
				else {
					x1 = a[st - 1].x;
					v1 = a[st - 1].v;
				}
			} else {
				x1 = -go_t[0] - ge_t[0];
				v1 = r ? -ge_q[r] : -go_q[r] - ge_q[r];
			}
			if (en != r) {
				if (r < en + en - w - 1)
					a[en].y = a[en].u = -(igapo + igape);
			} else {
				a[r].y = -go_q[0] - ge_q[0];
				a[r].u = r ? -ge_t[r] : -go_t[r] - ge_t[r];
			}

			if (with_cigar) {
				uint8_t *zr = z + (size_t)r * n_col;
				off[r] = st;
				off_end[r] = en;
				for (t = st; t <= en; ++t) {
					int32_t j = r - t;
					const int16_t *qpj = qp + (size_t)j * m;
					const int16_t *tfi = tf + (size_t)t * m;
					int16_t s = psw_dot_scaled(qpj, tfi, m, scale_shift);
					int16_t score0, ax, by, u1, q_open, q_ext, t_open, t_ext, z_after;
					uint8_t d;

					t_open = go_t[t + 1];
					t_ext = ge_t[t + 1];
					q_open = go_q[j + 1];
					q_ext = ge_q[j + 1];
					score0 = s;
					ax = x1 + v1;
					by = a[t].y + a[t].u;
					d = ax > score0 ? 1 : 0;
					score0 = ax > score0 ? ax : score0;
					d = by > score0 ? 2 : d;
					score0 = by > score0 ? by : score0;

					u1 = a[t].u;
					a[t].u = score0 - v1;
					v1 = a[t].v;
					a[t].v = score0 - u1;

					z_after = score0 - t_open;
					ax -= z_after;
					x1 = a[t].x;
					d |= ax > 0 ? 0x08 : 0;
					a[t].x = ax > 0 ? (int16_t)ax : 0;
					a[t].x -= t_open + t_ext;

					by -= score0 - q_open;
					d |= by > 0 ? 0x10 : 0;
					a[t].y = by > 0 ? (int16_t)by : 0;
					a[t].y -= q_open + q_ext;
					zr[t - st] = d;
				}
			} else {
				for (t = st; t <= en; ++t) {
					int32_t j = r - t;
					const int16_t *qpj = qp + (size_t)j * m;
					const int16_t *tfi = tf + (size_t)t * m;
					int16_t s = psw_dot_scaled(qpj, tfi, m, scale_shift);
					int16_t score0, ax, by, u1, q_open, q_ext, t_open, t_ext, z_after;

					t_open = go_t[t + 1];
					t_ext = ge_t[t + 1];
					q_open = go_q[j + 1];
					q_ext = ge_q[j + 1];
					score0 = s;
					ax = x1 + v1;
					by = a[t].y + a[t].u;
					score0 = ax > score0 ? ax : score0;
					score0 = by > score0 ? by : score0;

					u1 = a[t].u;
					a[t].u = score0 - v1;
					v1 = a[t].v;
					a[t].v = score0 - u1;

					z_after = score0 - t_open;
					ax -= z_after;
					x1 = a[t].x;
					a[t].x = ax > 0 ? (int16_t)ax : 0;
					a[t].x -= t_open + t_ext;

					by -= score0 - q_open;
					a[t].y = by > 0 ? (int16_t)by : 0;
					a[t].y -= q_open + q_ext;
				}
			}

			if (r == 0) {
				H[0] = (int32_t)a[0].v - ((int32_t)go_t[0] + ge_t[0]);
				row_max = H[0];
				row_max_t = 0;
			} else {
				for (t = st; t < en; ++t) {
					if (H[t] != PSW_NEG_INF)
						H[t] += (int32_t)a[t].v;
					if (H[t] > row_max)
						row_max = H[t], row_max_t = t;
				}
				if (en > 0 && H[en - 1] != PSW_NEG_INF)
					H[en] = H[en - 1] + (int32_t)a[en].u;
				else if (en == 0 && H[0] != PSW_NEG_INF)
					H[0] += (int32_t)a[0].v;
				else H[en] = PSW_NEG_INF;
				if (H[en] > row_max)
					row_max = H[en], row_max_t = en;
			}

			if (en == tlen - 1 && H[en] > mte_raw)
				mte_raw = H[en], mte_q = r - en;
			if (r - st == qlen - 1 && H[st] > mqe_raw)
				mqe_raw = H[st], mqe_t = st;
			if (row_max_t >= 0 && psw_apply_zdrop_raw(&max_raw, &max_t, &max_q, row_max, r, row_max_t, zdrop_scaled, igape)) {
				zdropped = 1;
				break;
			}
			if (r == qlen + tlen - 2 && en == tlen - 1)
				score_raw = H[tlen - 1];
		}

		if (with_cigar) {
			int rev_cigar = !!(flag & PSW_FLAG_REV_CIGAR);
			if (!zdropped && (flag & PSW_FLAG_GLOBAL) && score_raw != PSW_NEG_INF) {
				bt_t = tlen - 1; bt_q = qlen - 1;
			} else if (!zdropped && (flag & PSW_FLAG_SEMIGLOBAL) && mqe_raw != PSW_NEG_INF) {
				reach_end = 1;
				bt_t = mqe_t; bt_q = qlen - 1;
			} else if (!zdropped && !(flag & PSW_FLAG_LOCAL) && score_raw != PSW_NEG_INF) {
				bt_t = tlen - 1; bt_q = qlen - 1;
			} else if (!zdropped && !(flag & PSW_FLAG_GLOBAL) && mqe_raw != PSW_NEG_INF && mqe_raw >= max_raw) {
				reach_end = 1;
				bt_t = mqe_t; bt_q = qlen - 1;
			} else if (max_t >= 0 && max_q >= 0) {
				bt_t = max_t; bt_q = max_q;
			}
			if (bt_t >= 0 && bt_q >= 0)
				psw_backtrack(km, 1, rev_cigar, 0, z, off, off_end, n_col, bt_t, bt_q, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		}

		psw_finalize_extz(ez, scale, max_raw, max_t, max_q, score_raw, mqe_raw, mqe_t, mte_raw, mte_q, zdropped, reach_end);
	} while (0);

	if (z) kfree(km, z);
	if (off) kfree(km, off);
	if (H) kfree(km, H);
	if (go_q) kfree(km, go_q);
	if (ge_q) kfree(km, ge_q);
	if (go_t) kfree(km, go_t);
	if (ge_t) kfree(km, ge_t);
	if (a) kfree(km, a);
	if (qp) kfree(km, qp);
	if (tf) kfree(km, tf);
	if (qbf) kfree(km, qbf);
	if (tbf) kfree(km, tbf);
	psw_free_norm_prof(km, &query_n);
	psw_free_norm_prof(km, &target_n);
}

void psw_extz_ps(void *km, int qlen, const uint8_t *query,
                 int tlen, const psw_prof_t *target,
                 int8_t m, const int8_t *mat,
                 int8_t gapo, int8_t gape, int w, int zdrop, int flag, psw_extz_t *ez)
{
	uvxy16_t *a = 0;
	int16_t *tp = 0, *tbf = 0;
	int16_t *go_t = 0, *ge_t = 0;
	psw_prof_t target_n = { 0, 0, 0, 0 };
	const psw_prof_t *t_use;
	int32_t r, t, n_col, *off = 0, *off_end = 0, *H = 0;
	uint8_t *z = 0;
	int16_t scale;
	int8_t scale_shift;
	int16_t go_q, ge_q;
	int16_t igapo, igape;
	int with_cigar = !(flag & PSW_FLAG_SCORE_ONLY);
	int32_t max_raw = 0, score_raw = PSW_NEG_INF, mqe_raw = PSW_NEG_INF, mte_raw = PSW_NEG_INF;
	int max_t = -1, max_q = -1, mqe_t = -1, mte_q = -1;
	int32_t zdrop_scaled = zdrop < 0 ? -1 : (int32_t)zdrop;
	int zdropped = 0, reach_end = 0;
	int bt_t = -1, bt_q = -1;

	if (ez == 0) return;
	psw_reset_extz(ez);

	if (gapo < 0 || gape < 0) return;
	if (query == 0 || target == 0 || mat == 0) return;
	if (target->prof == 0) return;
	if (qlen < 0 || tlen < 0 || m <= 0) return;
	if (target->len < tlen || target->dim < m) return;

	scale = psw_pick_safe_scale_pow2(m, mat, gapo, gape);
	if (scale < 2) return;
	scale_shift = psw_scale_to_shift(scale);
	if (scale_shift <= 0) return;
	zdrop_scaled = zdrop < 0 ? -1 : zdrop_scaled * scale;
	go_q = psw_sat16((int32_t)gapo * scale);
	ge_q = psw_sat16((int32_t)gape * scale);
	if (!psw_make_norm_prof(km, target, &target_n, m, scale)) return;
	t_use = &target_n;

	do {
		int bad_query = 0;
		for (t = 0; t < qlen; ++t) {
			if ((int)query[t] < 0 || (int)query[t] >= m) {
				bad_query = 1;
				break;
			}
		}
		if (bad_query) break;

		if (w < 0) w = tlen > qlen ? tlen : qlen;
		n_col = w + 1 < tlen ? w + 1 : tlen;

		tp = psw_gen_tp_i16(km, tlen, t_use, m, mat);
		tbf = psw_gen_base_freq_i16(km, tlen, t_use, m, scale);
		if (tp == 0 || tbf == 0) break;

		go_t = (int16_t *)kmalloc(km, (size_t)(tlen + 1) * sizeof(int16_t));
		ge_t = (int16_t *)kmalloc(km, (size_t)(tlen + 1) * sizeof(int16_t));
		a = (uvxy16_t *)kmalloc(km, (size_t)(tlen + 1) * sizeof(uvxy16_t));
		H = (int32_t *)kmalloc(km, (size_t)(tlen > 0 ? tlen : 1) * sizeof(int32_t));
		if (go_t == 0 || ge_t == 0 || a == 0 || H == 0) break;

		igapo = psw_sat16((int32_t)gapo * scale);
		igape = psw_sat16((int32_t)gape * scale);
		for (t = 0; t < tlen; ++t) {
			go_t[t] = psw_sat16((int32_t)gapo * tbf[t]);
			ge_t[t] = psw_sat16((int32_t)gape * tbf[t]);
		}
		go_t[tlen] = igapo;
		ge_t[tlen] = igape;
		for (t = 0; t <= tlen; ++t)
			a[t].x = a[t].v = a[t].y = a[t].u = -(igapo + igape);
		for (t = 0; t < tlen; ++t) H[t] = PSW_NEG_INF;

		if (with_cigar) {
			z = (uint8_t *)kcalloc(km, (size_t)(qlen + tlen) * n_col, 1);
			off = (int32_t *)kmalloc(km, (size_t)(qlen + tlen) * sizeof(int32_t) * 2);
			if (z == 0 || off == 0) break;
			off_end = off + qlen + tlen;
		}

		if (qlen == 0 || tlen == 0) {
			int32_t s32 = qlen == 0 ? psw_gap_only_target_i16(go_t, ge_t, tlen)
				: psw_gap_only_query_scalar_i16(go_q, ge_q, qlen);
			score_raw = s32;
			if (qlen == 0) {
				mqe_raw = 0;
				reach_end = 1;
			}
			if (tlen == 0) mte_raw = 0;
			if (with_cigar) {
				ez->n_cigar = 0;
				if (qlen == 0 && tlen > 0)
					ez->cigar = psw_push_cigar(km, &ez->n_cigar, &ez->m_cigar, ez->cigar, PSW_CIGAR_DEL, tlen);
				else if (tlen == 0 && qlen > 0)
					ez->cigar = psw_push_cigar(km, &ez->n_cigar, &ez->m_cigar, ez->cigar, PSW_CIGAR_INS, qlen);
			}
			psw_finalize_extz(ez, scale, max_raw, max_t, max_q, score_raw, mqe_raw, mqe_t, mte_raw, mte_q, zdropped, reach_end);
			break;
		}

		for (r = 0; r < qlen + tlen - 1; ++r) {
			int32_t st = 0, en = tlen - 1;
			int16_t x1, v1;
			int32_t row_max = PSW_NEG_INF;
			int row_max_t = -1;

			if (st < r - qlen + 1) st = r - qlen + 1;
			if (en > r) en = r;
			if (st < (r - w + 1) >> 1) st = (r - w + 1) >> 1;
			if (en > (r + w) >> 1) en = (r + w) >> 1;
			if (st > en) {
				zdropped = 1;
				break;
			}

			if (st != 0) {
				if (r > st + st + w - 1) x1 = v1 = -(igapo + igape);
				else {
					x1 = a[st - 1].x;
					v1 = a[st - 1].v;
				}
			} else {
				x1 = -go_t[0] - ge_t[0];
				v1 = r ? -ge_q : -go_q - ge_q;
			}
			if (en != r) {
				if (r < en + en - w - 1)
					a[en].y = a[en].u = -(igapo + igape);
			} else {
				a[r].y = -go_q - ge_q;
				a[r].u = r ? -ge_t[r] : -go_t[r] - ge_t[r];
			}

			if (with_cigar) {
				uint8_t *zr = z + (size_t)r * n_col;
				off[r] = st;
				off_end[r] = en;
				for (t = st; t <= en; ++t) {
					int32_t j = r - t;
					int aidx = (int)query[j];
					int16_t s = tp[(size_t)aidx * tlen + t];
					int16_t score0, ax, by, u1, q_open, q_ext, t_open, t_ext, z_after;
					uint8_t d;

					t_open = go_t[t + 1];
					t_ext = ge_t[t + 1];
					q_open = go_q;
					q_ext = ge_q;
					score0 = s;
					ax = x1 + v1;
					by = a[t].y + a[t].u;
					d = ax > score0 ? 1 : 0;
					score0 = ax > score0 ? ax : score0;
					d = by > score0 ? 2 : d;
					score0 = by > score0 ? by : score0;

					u1 = a[t].u;
					a[t].u = score0 - v1;
					v1 = a[t].v;
					a[t].v = score0 - u1;

					z_after = score0 - t_open;
					ax -= z_after;
					x1 = a[t].x;
					d |= ax > 0 ? 0x08 : 0;
					a[t].x = ax > 0 ? (int16_t)ax : 0;
					a[t].x -= t_open + t_ext;

					by -= score0 - q_open;
					d |= by > 0 ? 0x10 : 0;
					a[t].y = by > 0 ? (int16_t)by : 0;
					a[t].y -= q_open + q_ext;
					zr[t - st] = d;
				}
			} else {
				for (t = st; t <= en; ++t) {
					int32_t j = r - t;
					int aidx = (int)query[j];
					int16_t s = tp[(size_t)aidx * tlen + t];
					int16_t score0, ax, by, u1, q_open, q_ext, t_open, t_ext, z_after;

					t_open = go_t[t + 1];
					t_ext = ge_t[t + 1];
					q_open = go_q;
					q_ext = ge_q;
					score0 = s;
					ax = x1 + v1;
					by = a[t].y + a[t].u;
					score0 = ax > score0 ? ax : score0;
					score0 = by > score0 ? by : score0;

					u1 = a[t].u;
					a[t].u = score0 - v1;
					v1 = a[t].v;
					a[t].v = score0 - u1;

					z_after = score0 - t_open;
					ax -= z_after;
					x1 = a[t].x;
					a[t].x = ax > 0 ? (int16_t)ax : 0;
					a[t].x -= t_open + t_ext;

					by -= score0 - q_open;
					a[t].y = by > 0 ? (int16_t)by : 0;
					a[t].y -= q_open + q_ext;
				}
			}

			if (r == 0) {
				H[0] = (int32_t)a[0].v - ((int32_t)go_t[0] + ge_t[0]);
				row_max = H[0];
				row_max_t = 0;
			} else {
				for (t = st; t < en; ++t) {
					if (H[t] != PSW_NEG_INF)
						H[t] += (int32_t)a[t].v;
					if (H[t] > row_max)
						row_max = H[t], row_max_t = t;
				}
				if (en > 0 && H[en - 1] != PSW_NEG_INF)
					H[en] = H[en - 1] + (int32_t)a[en].u;
				else if (en == 0 && H[0] != PSW_NEG_INF)
					H[0] += (int32_t)a[0].v;
				else H[en] = PSW_NEG_INF;
				if (H[en] > row_max)
					row_max = H[en], row_max_t = en;
			}

			if (en == tlen - 1 && H[en] > mte_raw)
				mte_raw = H[en], mte_q = r - en;
			if (r - st == qlen - 1 && H[st] > mqe_raw)
				mqe_raw = H[st], mqe_t = st;
			if (row_max_t >= 0 && psw_apply_zdrop_raw(&max_raw, &max_t, &max_q, row_max, r, row_max_t, zdrop_scaled, igape)) {
				zdropped = 1;
				break;
			}
			if (r == qlen + tlen - 2 && en == tlen - 1)
				score_raw = H[tlen - 1];
		}

		if (with_cigar) {
			int rev_cigar = !!(flag & PSW_FLAG_REV_CIGAR);
			if (!zdropped && (flag & PSW_FLAG_GLOBAL) && score_raw != PSW_NEG_INF) {
				bt_t = tlen - 1; bt_q = qlen - 1;
			} else if (!zdropped && (flag & PSW_FLAG_SEMIGLOBAL) && mqe_raw != PSW_NEG_INF) {
				reach_end = 1;
				bt_t = mqe_t; bt_q = qlen - 1;
			} else if (!zdropped && !(flag & PSW_FLAG_LOCAL) && score_raw != PSW_NEG_INF) {
				bt_t = tlen - 1; bt_q = qlen - 1;
			} else if (!zdropped && !(flag & PSW_FLAG_GLOBAL) && mqe_raw != PSW_NEG_INF && mqe_raw >= max_raw) {
				reach_end = 1;
				bt_t = mqe_t; bt_q = qlen - 1;
			} else if (max_t >= 0 && max_q >= 0) {
				bt_t = max_t; bt_q = max_q;
			}
			if (bt_t >= 0 && bt_q >= 0)
				psw_backtrack(km, 1, rev_cigar, 0, z, off, off_end, n_col, bt_t, bt_q, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		}

		psw_finalize_extz(ez, scale, max_raw, max_t, max_q, score_raw, mqe_raw, mqe_t, mte_raw, mte_q, zdropped, reach_end);
	} while (0);

	if (z) kfree(km, z);
	if (off) kfree(km, off);
	if (H) kfree(km, H);
	if (go_t) kfree(km, go_t);
	if (ge_t) kfree(km, ge_t);
	if (a) kfree(km, a);
	if (tp) kfree(km, tp);
	if (tbf) kfree(km, tbf);
	psw_free_norm_prof(km, &target_n);
}

