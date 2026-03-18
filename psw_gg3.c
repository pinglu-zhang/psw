#include <limits.h>
#include <stdint.h>
#include "psw.h"

#ifndef PSW_NEG_INF_F
#define PSW_NEG_INF_F (-1e30f)
#endif

#ifndef PSW_GG3_SCALE
#define PSW_GG3_SCALE 100
#endif

typedef struct { int16_t u, v, x, y; } uvxy16_t;

static inline int16_t psw_sat16(int32_t x)
{
	if (x > INT16_MAX) return INT16_MAX;
	if (x < INT16_MIN) return INT16_MIN;
	return (int16_t)x;
}

static inline int16_t psw_round_div_i32(int32_t num, int32_t den)
{
	if (den <= 0) den = 1;
	if (num >= 0) return psw_sat16((num + den / 2) / den);
	return psw_sat16((num - den / 2) / den);
}

static inline int16_t psw_to_scaled_ratio(int32_t num, int32_t den)
{
	if (den <= 0) den = 1;
	if (num >= 0) return psw_sat16((num * PSW_GG3_SCALE + den / 2) / den);
	return psw_sat16((num * PSW_GG3_SCALE - den / 2) / den);
}

static inline int16_t psw_mul_scaled(int16_t a, int16_t b)
{
	int32_t p = (int32_t)a * (int32_t)b;
	return psw_round_div_i32(p, PSW_GG3_SCALE);
}

static inline int16_t psw_div_scale_round_i32(int32_t x)
{
	if (x >= 0) return psw_sat16((x + PSW_GG3_SCALE / 2) / PSW_GG3_SCALE);
	return psw_sat16((x - PSW_GG3_SCALE / 2) / PSW_GG3_SCALE);
}

static inline int32_t psw_dot_scaled(const int16_t *x, const int16_t *y, int m, int use_m5, int use_m4)
{
	int32_t raw;
	if (use_m5) {
		raw = (int32_t)x[0] * y[0] + (int32_t)x[1] * y[1] + (int32_t)x[2] * y[2] +
		      (int32_t)x[3] * y[3] + (int32_t)x[4] * y[4];
	} else if (use_m4) {
		raw = (int32_t)x[0] * y[0] + (int32_t)x[1] * y[1] + (int32_t)x[2] * y[2] +
		      (int32_t)x[3] * y[3];
	} else {
		int b;
		raw = 0;
		for (b = 0; b < m; ++b) raw += (int32_t)x[b] * y[b];
	}
	return (int32_t)psw_div_scale_round_i32(raw);
}

static inline int16_t *psw_gen_base_freq_i16(void *km, int len, const psw_prof_t *p, int8_t m)
{
	int i, a;
	int32_t depth;
	int16_t *bf;

	bf = (int16_t*)kmalloc(km, (size_t)len * sizeof(int16_t));
	if (bf == 0) return 0;

	depth = p->depth > 0 ? (int32_t)p->depth : 1;
	for (i = 0; i < len; ++i) {
		const uint32_t *col = p->prof + (size_t)i * p->dim;
		int32_t sum = 0;
		for (a = 0; a < m; ++a)
			sum += (int32_t)col[a];
		bf[i] = psw_to_scaled_ratio(sum, depth);
		if (bf[i] < 0) bf[i] = 0;
		if (bf[i] > PSW_GG3_SCALE) bf[i] = PSW_GG3_SCALE;
	}
	return bf;
}

static inline int16_t *psw_gen_qp_i16(void *km, int qlen, const psw_prof_t *query, int8_t m, const int8_t *mat)
{
	int a, b, j;
	const int32_t depth = query->depth > 0 ? (int32_t)query->depth : 1;
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
			dst[b] = psw_to_scaled_ratio(s, depth);
		}
	}
	return qp;
}

static inline int16_t *psw_gen_tf_i16(void *km, int tlen, const psw_prof_t *target, int8_t m)
{
	int i, b;
	const int32_t depth = target->depth > 0 ? (int32_t)target->depth : 1;
	int16_t *tf = (int16_t*)kmalloc(km, (size_t)tlen * m * sizeof(int16_t));
	if (tf == 0) return 0;

	for (i = 0; i < tlen; ++i) {
		const uint32_t *tcol = target->prof + (size_t)i * target->dim;
		int16_t *dst = tf + (size_t)i * m;
		for (b = 0; b < m; ++b)
			dst[b] = psw_to_scaled_ratio((int32_t)tcol[b], depth);
	}
	return tf;
}

static inline int16_t *psw_gen_tp_i16(void *km, int tlen, const psw_prof_t *target,
                                      int8_t m, const int8_t *mat)
{
	int a, b, i;
	const int32_t depth = target->depth > 0 ? (int32_t)target->depth : 1;
	int16_t *tp;

	tp = (int16_t*)kmalloc(km, (size_t)m * tlen * sizeof(int16_t));
	if (tp == 0) return 0;

	for (i = 0; i < tlen; ++i) {
		const uint32_t *tcol = target->prof + (size_t)i * target->dim;
		for (a = 0; a < m; ++a) {
			int32_t s = 0;
			for (b = 0; b < m; ++b)
				s += (int32_t)mat[a * m + b] * (int32_t)tcol[b];
			tp[(size_t)a * tlen + i] = psw_to_scaled_ratio(s, depth);
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

float psw_gg3_pp(void *km, int qlen, const psw_prof_t *query,
                 int tlen, const psw_prof_t *target,
                 int8_t m, const int8_t *mat,
                 int8_t gapo, int8_t gape, int w,
                 int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{
	uvxy16_t *a;
	int16_t *qp, *tf;
	int16_t *qbf, *tbf;
	int16_t *go_q, *ge_q, *go_t, *ge_t;
	int16_t *go_ge_q, *go_ge_t;
	int32_t r, t, n_col, *off = 0;
	int16_t score16 = INT16_MIN;
	int32_t H0 = 0;
	int32_t last_H0_t = 0;
	uint8_t *z = 0;
	const int use_m5 = (m == 5);
	const int use_m4 = (m == 4);

	if (gapo < 0 || gape < 0) return PSW_NEG_INF_F;
	if (query == 0 || target == 0 || mat == 0) return PSW_NEG_INF_F;
	if (query->prof == 0 || target->prof == 0) return PSW_NEG_INF_F;
	if (qlen < 0 || tlen < 0 || m <= 0) return PSW_NEG_INF_F;
	if (query->len < qlen || target->len < tlen) return PSW_NEG_INF_F;
	if (query->dim < m || target->dim < m) return PSW_NEG_INF_F;

	if (w < 0) w = tlen > qlen ? tlen : qlen;
	n_col = w + 1 < tlen ? w + 1 : tlen;

	qp = psw_gen_qp_i16(km, qlen, query, m, mat);
	if (qp == 0) return PSW_NEG_INF_F;
	tf = psw_gen_tf_i16(km, tlen, target, m);
	if (tf == 0) {
		kfree(km, qp);
		return PSW_NEG_INF_F;
	}
	qbf = psw_gen_base_freq_i16(km, qlen, query, m);
	if (qbf == 0) {
		kfree(km, qp); kfree(km, tf);
		return PSW_NEG_INF_F;
	}
	tbf = psw_gen_base_freq_i16(km, tlen, target, m);
	if (tbf == 0) {
		kfree(km, qp); kfree(km, tf); kfree(km, qbf);
		return PSW_NEG_INF_F;
	}

	go_q = (int16_t*)kmalloc(km, (size_t)qlen * sizeof(int16_t));
	ge_q = (int16_t*)kmalloc(km, (size_t)qlen * sizeof(int16_t));
	go_t = (int16_t*)kmalloc(km, (size_t)tlen * sizeof(int16_t));
	ge_t = (int16_t*)kmalloc(km, (size_t)tlen * sizeof(int16_t));
	go_ge_q = (int16_t*)kmalloc(km, (size_t)qlen * sizeof(int16_t));
	go_ge_t = (int16_t*)kmalloc(km, (size_t)tlen * sizeof(int16_t));
	a = (uvxy16_t*)kcalloc(km, tlen + 1, sizeof(uvxy16_t));
	if (go_q == 0 || ge_q == 0 || go_t == 0 || ge_t == 0 || go_ge_q == 0 || go_ge_t == 0 || a == 0) {
		if (go_q) kfree(km, go_q);
		if (ge_q) kfree(km, ge_q);
		if (go_t) kfree(km, go_t);
		if (ge_t) kfree(km, ge_t);
		if (go_ge_q) kfree(km, go_ge_q);
		if (go_ge_t) kfree(km, go_ge_t);
		if (a) kfree(km, a);
		kfree(km, qp); kfree(km, tf); kfree(km, qbf); kfree(km, tbf);
		return PSW_NEG_INF_F;
	}

	for (t = 0; t < qlen; ++t) {
		go_q[t] = psw_sat16((int32_t)gapo * qbf[t]);
		ge_q[t] = psw_sat16((int32_t)gape * qbf[t]);
		go_ge_q[t] = (int16_t)(go_q[t] + ge_q[t]);
	}
	for (t = 0; t < tlen; ++t) {
		go_t[t] = psw_sat16((int32_t)gapo * tbf[t]);
		ge_t[t] = psw_sat16((int32_t)gape * tbf[t]);
		go_ge_t[t] = (int16_t)(go_t[t] + ge_t[t]);
	}

	if (m_cigar_ && n_cigar_ && cigar_) {
		*n_cigar_ = 0;
		z = (uint8_t*)kcalloc(km, (size_t)(qlen + tlen) * n_col, 1);
		off = (int32_t*)kmalloc(km, (size_t)(qlen + tlen) * sizeof(int32_t));
		if (z == 0 || off == 0) {
			if (z) kfree(km, z);
			if (off) kfree(km, off);
			kfree(km, go_q); kfree(km, ge_q); kfree(km, go_t); kfree(km, ge_t);
			kfree(km, go_ge_q); kfree(km, go_ge_t);
			kfree(km, a);
			kfree(km, qp); kfree(km, tf); kfree(km, qbf); kfree(km, tbf);
			return PSW_NEG_INF_F;
		}
	}

	if (qlen == 0 || tlen == 0) {
		int32_t s32 = qlen == 0 ? psw_gap_only_target_i16(go_t, ge_t, tlen)
		                       : psw_gap_only_query_i16(go_q, ge_q, qlen);
		score16 = psw_sat16(s32);
		if (m_cigar_ && n_cigar_ && cigar_) {
			*n_cigar_ = 0;
			if (qlen == 0 && tlen > 0)
				*cigar_ = psw_push_cigar(km, n_cigar_, m_cigar_, *cigar_, PSW_CIGAR_DEL, tlen);
			else if (tlen == 0 && qlen > 0)
				*cigar_ = psw_push_cigar(km, n_cigar_, m_cigar_, *cigar_, PSW_CIGAR_INS, qlen);
		}
		if (z) kfree(km, z);
		if (off) kfree(km, off);
		kfree(km, go_q); kfree(km, ge_q); kfree(km, go_t); kfree(km, ge_t);
		kfree(km, go_ge_q); kfree(km, go_ge_t);
		kfree(km, a);
		kfree(km, qp); kfree(km, tf); kfree(km, qbf); kfree(km, tbf);
		return (float)score16 / (float)PSW_GG3_SCALE;
	}

	for (r = 0; r < qlen + tlen - 1; ++r) {
		int32_t st = 0, en = tlen - 1;
		int16_t x1, v1;

		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r - w + 1) >> 1) st = (r - w + 1) >> 1;
		if (en > (r + w) >> 1) en = (r + w) >> 1;
		if (st > en) continue;

		if (st != 0) {
			if (r > st + st + w - 1) x1 = v1 = 0;
			else {
				x1 = a[st - 1].x;
				v1 = a[st - 1].v;
			}
		} else {
			x1 = 0;
			v1 = r ? go_q[r - 1] : 0;
		}
		if (en != r) {
			if (r < en + en - w - 1)
				a[en].y = a[en].u = 0;
		} else {
			a[r].y = 0;
			a[r].u = r ? go_t[r - 1] : 0;
		}

		if (z) {
			uint8_t *zr = z + (size_t)r * n_col;
			off[r] = st;
			for (t = st; t <= en; ++t) {
				int32_t j = r - t;
				const int16_t *qpj = qp + (size_t)j * m;
				const int16_t *tfi = tf + (size_t)t * m;
				int32_t s = psw_dot_scaled(qpj, tfi, m, use_m5, use_m4);
				int32_t score0, ax, by;
				int16_t score0_16, u1, q_open, t_open;
				int32_t z_after;
				uint8_t d;

				q_open = go_t[t];
				t_open = go_q[j];
				score0 = s + (int32_t)go_ge_t[t] + go_ge_q[j];
				ax = (int32_t)x1 + v1;
				by = (int32_t)a[t].y + a[t].u;
				d = ax > score0 ? 1 : 0;
				score0 = ax > score0 ? ax : score0;
				d = by > score0 ? 2 : d;
				score0 = by > score0 ? by : score0;
				score0_16 = (int16_t)score0;

				u1 = a[t].u;
				a[t].u = (int16_t)((int32_t)score0_16 - v1);
				v1 = a[t].v;
				a[t].v = (int16_t)((int32_t)score0_16 - u1);

				z_after = (int32_t)score0_16 - q_open;
				ax -= z_after;
				x1 = a[t].x;
				d |= ax > 0 ? 0x08 : 0;
				a[t].x = ax > 0 ? (int16_t)ax : 0;

				by -= ((int32_t)score0_16 - t_open);
				d |= by > 0 ? 0x10 : 0;
				a[t].y = by > 0 ? (int16_t)by : 0;

				zr[t - st] = d;
			}
		} else {
			for (t = st; t <= en; ++t) {
				int32_t j = r - t;
				const int16_t *qpj = qp + (size_t)j * m;
				const int16_t *tfi = tf + (size_t)t * m;
				int32_t s = psw_dot_scaled(qpj, tfi, m, use_m5, use_m4);
				int32_t score0, ax, by;
				int16_t score0_16, u1, q_open, t_open;

				q_open = go_t[t];
				t_open = go_q[j];
				score0 = s + (int32_t)go_ge_t[t] + go_ge_q[j];
				ax = (int32_t)x1 + v1;
				by = (int32_t)a[t].y + a[t].u;
				score0 = ax > score0 ? ax : score0;
				score0 = by > score0 ? by : score0;
				score0_16 = (int16_t)score0;

				u1 = a[t].u;
				a[t].u = (int16_t)((int32_t)score0_16 - v1);
				v1 = a[t].v;
				a[t].v = (int16_t)((int32_t)score0_16 - u1);

				ax -= ((int32_t)score0_16 - q_open);
				x1 = a[t].x;
				a[t].x = ax > 0 ? (int16_t)ax : 0;

				by -= ((int32_t)score0_16 - t_open);
				a[t].y = by > 0 ? (int16_t)by : 0;
			}
		}

		if (r > 0) {
			if (last_H0_t >= st && last_H0_t <= en) {
				int32_t jh = r - last_H0_t;
				H0 += (int32_t)a[last_H0_t].v - go_ge_q[jh];
			} else {
				++last_H0_t;
				H0 += (int32_t)a[last_H0_t].u - go_ge_t[last_H0_t];
			}
		} else {
			H0 = (int32_t)a[0].v - ((int32_t)go_ge_t[0] + go_ge_q[0]);
			last_H0_t = 0;
		}
	}

	score16 = psw_sat16(H0);

	if (z && off) {
		psw_backtrack(km, 1, 0, 0, z, off, 0, n_col, tlen - 1, qlen - 1,
		              m_cigar_, n_cigar_, cigar_);
	}

	if (z) kfree(km, z);
	if (off) kfree(km, off);
	kfree(km, go_q); kfree(km, ge_q); kfree(km, go_t); kfree(km, ge_t);
	kfree(km, go_ge_q); kfree(km, go_ge_t);
	kfree(km, a);
	kfree(km, qp); kfree(km, tf); kfree(km, qbf); kfree(km, tbf);
	return (float)score16 / (float)PSW_GG3_SCALE;
}

float psw_gg3_ps(void *km, int qlen, const uint8_t *query,
                 int tlen, const psw_prof_t *target,
                 int8_t m, const int8_t *mat,
                 int8_t gapo, int8_t gape, int w,
                 int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{
	uvxy16_t *a;
	int16_t *tp, *tbf;
	int16_t *go_t, *ge_t;
	int32_t r, t, n_col, *off = 0;
	int16_t score16 = INT16_MIN;
	int32_t H0 = 0;
	int32_t last_H0_t = 0;
	uint8_t *z = 0;
	const int16_t go_q = psw_sat16((int32_t)gapo * PSW_GG3_SCALE);
	const int16_t ge_q = psw_sat16((int32_t)gape * PSW_GG3_SCALE);

	if (gapo < 0 || gape < 0) return PSW_NEG_INF_F;
	if (query == 0 || target == 0 || mat == 0) return PSW_NEG_INF_F;
	if (target->prof == 0) return PSW_NEG_INF_F;
	if (qlen < 0 || tlen < 0 || m <= 0) return PSW_NEG_INF_F;
	if (target->len < tlen || target->dim < m) return PSW_NEG_INF_F;

	for (t = 0; t < qlen; ++t)
		if ((int)query[t] < 0 || (int)query[t] >= m) return PSW_NEG_INF_F;

	if (w < 0) w = tlen > qlen ? tlen : qlen;
	n_col = w + 1 < tlen ? w + 1 : tlen;

	tp = psw_gen_tp_i16(km, tlen, target, m, mat);
	if (tp == 0) return PSW_NEG_INF_F;
	tbf = psw_gen_base_freq_i16(km, tlen, target, m);
	if (tbf == 0) {
		kfree(km, tp);
		return PSW_NEG_INF_F;
	}
	go_t = (int16_t*)kmalloc(km, (size_t)tlen * sizeof(int16_t));
	ge_t = (int16_t*)kmalloc(km, (size_t)tlen * sizeof(int16_t));
	a = (uvxy16_t*)kcalloc(km, tlen + 1, sizeof(uvxy16_t));
	if (go_t == 0 || ge_t == 0 || a == 0) {
		if (go_t) kfree(km, go_t);
		if (ge_t) kfree(km, ge_t);
		if (a) kfree(km, a);
		kfree(km, tp); kfree(km, tbf);
		return PSW_NEG_INF_F;
	}
	for (t = 0; t < tlen; ++t) {
		go_t[t] = psw_sat16((int32_t)gapo * tbf[t]);
		ge_t[t] = psw_sat16((int32_t)gape * tbf[t]);
	}

	if (m_cigar_ && n_cigar_ && cigar_) {
		*n_cigar_ = 0;
		z = (uint8_t*)kcalloc(km, (size_t)(qlen + tlen) * n_col, 1);
		off = (int32_t*)kmalloc(km, (size_t)(qlen + tlen) * sizeof(int32_t));
		if (z == 0 || off == 0) {
			if (z) kfree(km, z);
			if (off) kfree(km, off);
			kfree(km, go_t); kfree(km, ge_t); kfree(km, a);
			kfree(km, tp); kfree(km, tbf);
			return PSW_NEG_INF_F;
		}
	}

	if (qlen == 0 || tlen == 0) {
		int32_t s32 = qlen == 0 ? psw_gap_only_target_i16(go_t, ge_t, tlen)
		                       : psw_gap_only_query_scalar_i16(go_q, ge_q, qlen);
		score16 = psw_sat16(s32);
		if (m_cigar_ && n_cigar_ && cigar_) {
			*n_cigar_ = 0;
			if (qlen == 0 && tlen > 0)
				*cigar_ = psw_push_cigar(km, n_cigar_, m_cigar_, *cigar_, PSW_CIGAR_DEL, tlen);
			else if (tlen == 0 && qlen > 0)
				*cigar_ = psw_push_cigar(km, n_cigar_, m_cigar_, *cigar_, PSW_CIGAR_INS, qlen);
		}
		if (z) kfree(km, z);
		if (off) kfree(km, off);
		kfree(km, go_t); kfree(km, ge_t); kfree(km, a);
		kfree(km, tp); kfree(km, tbf);
		return (float)score16 / (float)PSW_GG3_SCALE;
	}

	for (r = 0; r < qlen + tlen - 1; ++r) {
		int32_t st = 0, en = tlen - 1;
		int16_t x1, v1;

		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r - w + 1) >> 1) st = (r - w + 1) >> 1;
		if (en > (r + w) >> 1) en = (r + w) >> 1;
		if (st > en) continue;

		if (st != 0) {
			if (r > st + st + w - 1) x1 = v1 = 0;
			else {
				x1 = a[st - 1].x;
				v1 = a[st - 1].v;
			}
		} else {
			x1 = 0;
			v1 = r ? go_q : 0;
		}
		if (en != r) {
			if (r < en + en - w - 1)
				a[en].y = a[en].u = 0;
		} else {
			a[r].y = 0;
			a[r].u = r ? go_t[r - 1] : 0;
		}

		if (z) {
			uint8_t *zr = z + (size_t)r * n_col;
			off[r] = st;
			for (t = st; t <= en; ++t) {
				int32_t j = r - t;
				int aidx = (int)query[j];
				int32_t s = tp[(size_t)aidx * tlen + t];
				int32_t score0, ax, by;
				int16_t score0_16, u1, q_open, t_open;
				uint8_t d;

				q_open = go_t[t];
				t_open = go_q;
				score0 = s + q_open + ge_t[t] + t_open + ge_q;
				ax = (int32_t)x1 + v1;
				by = (int32_t)a[t].y + a[t].u;
				d = ax > score0 ? 1 : 0;
				score0 = ax > score0 ? ax : score0;
				d = by > score0 ? 2 : d;
				score0 = by > score0 ? by : score0;
				score0_16 = psw_sat16(score0);

				u1 = a[t].u;
				a[t].u = psw_sat16((int32_t)score0_16 - v1);
				v1 = a[t].v;
				a[t].v = psw_sat16((int32_t)score0_16 - u1);

				ax -= ((int32_t)score0_16 - q_open);
				x1 = a[t].x;
				d |= ax > 0 ? 0x08 : 0;
				a[t].x = ax > 0 ? psw_sat16(ax) : 0;

				by -= ((int32_t)score0_16 - t_open);
				d |= by > 0 ? 0x10 : 0;
				a[t].y = by > 0 ? psw_sat16(by) : 0;

				zr[t - st] = d;
			}
		} else {
			for (t = st; t <= en; ++t) {
				int32_t j = r - t;
				int aidx = (int)query[j];
				int32_t s = tp[(size_t)aidx * tlen + t];
				int32_t score0, ax, by;
				int16_t score0_16, u1, q_open, t_open;

				q_open = go_t[t];
				t_open = go_q;
				score0 = s + q_open + ge_t[t] + t_open + ge_q;
				ax = (int32_t)x1 + v1;
				by = (int32_t)a[t].y + a[t].u;
				score0 = ax > score0 ? ax : score0;
				score0 = by > score0 ? by : score0;
				score0_16 = psw_sat16(score0);

				u1 = a[t].u;
				a[t].u = psw_sat16((int32_t)score0_16 - v1);
				v1 = a[t].v;
				a[t].v = psw_sat16((int32_t)score0_16 - u1);

				ax -= ((int32_t)score0_16 - q_open);
				x1 = a[t].x;
				a[t].x = ax > 0 ? psw_sat16(ax) : 0;

				by -= ((int32_t)score0_16 - t_open);
				a[t].y = by > 0 ? psw_sat16(by) : 0;
			}
		}

		if (r > 0) {
			if (last_H0_t >= st && last_H0_t <= en)
				H0 += (int32_t)a[last_H0_t].v - ((int32_t)go_q + ge_q);
			else {
				++last_H0_t;
				H0 += (int32_t)a[last_H0_t].u - ((int32_t)go_t[last_H0_t] + ge_t[last_H0_t]);
			}
		} else {
			H0 = (int32_t)a[0].v - ((int32_t)go_t[0] + ge_t[0] + go_q + ge_q);
			last_H0_t = 0;
		}
	}

	score16 = psw_sat16(H0);

	if (z && off) {
		psw_backtrack(km, 1, 0, 0, z, off, 0, n_col, tlen - 1, qlen - 1,
		              m_cigar_, n_cigar_, cigar_);
	}

	if (z) kfree(km, z);
	if (off) kfree(km, off);
	kfree(km, go_t); kfree(km, ge_t); kfree(km, a);
	kfree(km, tp); kfree(km, tbf);
	return (float)score16 / (float)PSW_GG3_SCALE;
}
