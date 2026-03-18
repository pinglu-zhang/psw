#include <stdint.h>
#include "psw.h"

#ifndef PSW_NEG_INF_F
#define PSW_NEG_INF_F (-1e30f)
#endif

typedef struct { float u, v, x, y; } uvxy_t;

static inline float *psw_gen_base_freq(void *km, int len, const psw_prof_t *p, int8_t m)
{
	int i, a;
	float depth;
	float *bf;

	bf = (float*)kmalloc(km, (size_t)len * sizeof(float));
	if (bf == 0) return 0;

	depth = p->depth > 0 ? (float)p->depth : 1.0f;
	for (i = 0; i < len; ++i) {
		const uint32_t *col = p->prof + (size_t)i * p->dim;
		float sum = 0.0f;
		for (a = 0; a < m; ++a)
			sum += (float)col[a];
		bf[i] = sum / depth;
		if (bf[i] < 0.0f) bf[i] = 0.0f;
		if (bf[i] > 1.0f) bf[i] = 1.0f;
	}
	return bf;
}

static inline float *psw_gen_qp(void *km, int qlen, const psw_prof_t *query, int8_t m, const int8_t *mat)
{
	int a, b, j;
	const float inv_depth = 1.0f / (query->depth > 0 ? (float)query->depth : 1.0f);
	float *qp;

	qp = (float*)kmalloc(km, (size_t)qlen * m * sizeof(float));
	if (qp == 0) return 0;

	for (j = 0; j < qlen; ++j) {
		const uint32_t *qcol = query->prof + (size_t)j * query->dim;
		float *dst = qp + (size_t)j * m;
		for (b = 0; b < m; ++b) {
			float s = 0.0f;
			for (a = 0; a < m; ++a)
				s += (float)qcol[a] * (float)mat[a * m + b];
			dst[b] = s * inv_depth;
		}
	}
	return qp;
}

static inline float *psw_gen_tf(void *km, int tlen, const psw_prof_t *target, int8_t m)
{
	int i, b;
	const float inv_depth = 1.0f / (target->depth > 0 ? (float)target->depth : 1.0f);
	float *tf = (float*)kmalloc(km, (size_t)tlen * m * sizeof(float));
	if (tf == 0) return 0;

	for (i = 0; i < tlen; ++i) {
		const uint32_t *tcol = target->prof + (size_t)i * target->dim;
		float *dst = tf + (size_t)i * m;
		for (b = 0; b < m; ++b)
			dst[b] = (float)tcol[b] * inv_depth;
	}
	return tf;
}

static inline float *psw_gen_tp(void *km, int tlen, const psw_prof_t *target,
                                int8_t m, const int8_t *mat)
{
	int a, b, i;
	const float inv_depth = 1.0f / (target->depth > 0 ? (float)target->depth : 1.0f);
	float *tp;

	tp = (float*)kmalloc(km, (size_t)m * tlen * sizeof(float));
	if (tp == 0) return 0;

	for (i = 0; i < tlen; ++i) {
		const uint32_t *tcol = target->prof + (size_t)i * target->dim;
		for (a = 0; a < m; ++a) {
			float s = 0.0f;
			for (b = 0; b < m; ++b)
				s += (float)mat[a * m + b] * (float)tcol[b];
			tp[(size_t)a * tlen + i] = s * inv_depth;
		}
	}
	return tp;
}

static inline float psw_gap_only_target(const float *go_t, const float *ge_t, int tlen)
{
	int i;
	float s = 0.0f;
	for (i = 0; i < tlen; ++i)
		s -= (i == 0 ? (go_t[i] + ge_t[i]) : ge_t[i]);
	return s;
}

static inline float psw_gap_only_query(const float *go_q, const float *ge_q, int qlen)
{
	int j;
	float s = 0.0f;
	for (j = 0; j < qlen; ++j)
		s -= (j == 0 ? (go_q[j] + ge_q[j]) : ge_q[j]);
	return s;
}

float psw_gg2_pp(void *km, int qlen, const psw_prof_t *query,
                 int tlen, const psw_prof_t *target,
                 int8_t m, const int8_t *mat,
                 int8_t gapo, int8_t gape, int w,
                 int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{
	uvxy_t *a;
	float *qp, *tf;
	float *qbf, *tbf;
	float *go_q, *ge_q, *go_t, *ge_t;
	int32_t r, t, n_col, *off = 0;
	float score = PSW_NEG_INF_F, H0 = 0.0f;
	int32_t last_H0_t = 0;
	uint8_t *z = 0;
	const float fgapo = (float)gapo, fgape = (float)gape;

	if (query == 0 || target == 0 || mat == 0) return PSW_NEG_INF_F;
	if (query->prof == 0 || target->prof == 0) return PSW_NEG_INF_F;
	if (qlen < 0 || tlen < 0 || m <= 0) return PSW_NEG_INF_F;
	if (query->len < qlen || target->len < tlen) return PSW_NEG_INF_F;
	if (query->dim < m || target->dim < m) return PSW_NEG_INF_F;

	if (w < 0) w = tlen > qlen ? tlen : qlen;
	n_col = w + 1 < tlen ? w + 1 : tlen;

	qp = psw_gen_qp(km, qlen, query, m, mat);
	if (qp == 0) return PSW_NEG_INF_F;
	tf = psw_gen_tf(km, tlen, target, m);
	if (tf == 0) {
		kfree(km, qp);
		return PSW_NEG_INF_F;
	}
	qbf = psw_gen_base_freq(km, qlen, query, m);
	if (qbf == 0) {
		kfree(km, qp); kfree(km, tf);
		return PSW_NEG_INF_F;
	}
	tbf = psw_gen_base_freq(km, tlen, target, m);
	if (tbf == 0) {
		kfree(km, qp); kfree(km, tf); kfree(km, qbf);
		return PSW_NEG_INF_F;
	}

	go_q = (float*)kmalloc(km, (size_t)qlen * sizeof(float));
	ge_q = (float*)kmalloc(km, (size_t)qlen * sizeof(float));
	go_t = (float*)kmalloc(km, (size_t)tlen * sizeof(float));
	ge_t = (float*)kmalloc(km, (size_t)tlen * sizeof(float));
	a = (uvxy_t*)kcalloc(km, tlen + 1, sizeof(uvxy_t));
	if (go_q == 0 || ge_q == 0 || go_t == 0 || ge_t == 0 || a == 0) {
		if (go_q) kfree(km, go_q);
		if (ge_q) kfree(km, ge_q);
		if (go_t) kfree(km, go_t);
		if (ge_t) kfree(km, ge_t);
		if (a) kfree(km, a);
		kfree(km, qp); kfree(km, tf); kfree(km, qbf); kfree(km, tbf);
		return PSW_NEG_INF_F;
	}

	for (t = 0; t < qlen; ++t) {
		go_q[t] = fgapo * qbf[t];
		ge_q[t] = fgape * qbf[t];
	}
	for (t = 0; t < tlen; ++t) {
		go_t[t] = fgapo * tbf[t];
		ge_t[t] = fgape * tbf[t];
	}

	if (m_cigar_ && n_cigar_ && cigar_) {
		*n_cigar_ = 0;
		z = (uint8_t*)kcalloc(km, (size_t)(qlen + tlen) * n_col, 1);
		off = (int32_t*)kmalloc(km, (size_t)(qlen + tlen) * sizeof(int32_t));
		if (z == 0 || off == 0) {
			if (z) kfree(km, z);
			if (off) kfree(km, off);
			kfree(km, go_q); kfree(km, ge_q); kfree(km, go_t); kfree(km, ge_t);
			kfree(km, a);
			kfree(km, qp); kfree(km, tf); kfree(km, qbf); kfree(km, tbf);
			return PSW_NEG_INF_F;
		}
	}

	if (qlen == 0 || tlen == 0) {
		score = qlen == 0 ? psw_gap_only_target(go_t, ge_t, tlen) : psw_gap_only_query(go_q, ge_q, qlen);
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
		kfree(km, a);
		kfree(km, qp); kfree(km, tf); kfree(km, qbf); kfree(km, tbf);
		return score;
	}

	for (r = 0; r < qlen + tlen - 1; ++r) {
		int32_t st = 0, en = tlen - 1;
		float x1, v1;

		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r - w + 1) >> 1) st = (r - w + 1) >> 1;
		if (en > (r + w) >> 1) en = (r + w) >> 1;
		if (st > en) continue;

		if (st != 0) {
			if (r > st + st + w - 1) x1 = v1 = 0.0f;
			else {
				x1 = a[st - 1].x;
				v1 = a[st - 1].v;
			}
		} else {
			x1 = 0.0f;
			v1 = r ? go_q[r - 1] : 0.0f;
		}
		if (en != r) {
			if (r < en + en - w - 1)
				a[en].y = a[en].u = 0.0f;
		} else {
			a[r].y = 0.0f;
			a[r].u = r ? go_t[r - 1] : 0.0f;
		}

		if (z) {
			uint8_t *zr = z + (size_t)r * n_col;
			off[r] = st;
			for (t = st; t <= en; ++t) {
				int32_t j = r - t;
				const float *qpj = qp + (size_t)j * m;
				const float *tfi = tf + (size_t)t * m;
				float s = 0.0f, score0, ax, by;
				float u1, z_after;
				float q_open = go_t[t];
				float t_open = go_q[j];
				float bias = q_open + ge_t[t] + t_open + ge_q[j];
				uint8_t d;
				int b;

				for (b = 0; b < m; ++b) s += qpj[b] * tfi[b];

				score0 = s + bias;
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

				z_after = score0 - q_open;
				ax -= z_after;
				x1 = a[t].x;
				d |= ax > 0.0f ? 0x08 : 0;
				a[t].x = ax > 0.0f ? ax : 0.0f;

				by -= (score0 - t_open);
				d |= by > 0.0f ? 0x10 : 0;
				a[t].y = by > 0.0f ? by : 0.0f;

				zr[t - st] = d;
			}
		} else {
			for (t = st; t <= en; ++t) {
				int32_t j = r - t;
				const float *qpj = qp + (size_t)j * m;
				const float *tfi = tf + (size_t)t * m;
				float s = 0.0f, score0, ax, by;
				float u1;
				float q_open = go_t[t];
				float t_open = go_q[j];
				float bias = q_open + ge_t[t] + t_open + ge_q[j];
				int b;

				for (b = 0; b < m; ++b) s += qpj[b] * tfi[b];

				score0 = s + bias;
				ax = x1 + v1;
				by = a[t].y + a[t].u;
				score0 = ax > score0 ? ax : score0;
				score0 = by > score0 ? by : score0;

				u1 = a[t].u;
				a[t].u = score0 - v1;
				v1 = a[t].v;
				a[t].v = score0 - u1;

				ax -= (score0 - q_open);
				x1 = a[t].x;
				a[t].x = ax > 0.0f ? ax : 0.0f;

				by -= (score0 - t_open);
				a[t].y = by > 0.0f ? by : 0.0f;
			}
		}

		if (r > 0) {
			if (last_H0_t >= st && last_H0_t <= en) {
				int32_t jh = r - last_H0_t;
				H0 += a[last_H0_t].v - (go_q[jh] + ge_q[jh]);
			} else {
				++last_H0_t;
				H0 += a[last_H0_t].u - (go_t[last_H0_t] + ge_t[last_H0_t]);
			}
		} else {
			H0 = a[0].v - (go_t[0] + ge_t[0] + go_q[0] + ge_q[0]);
			last_H0_t = 0;
		}
	}

	score = H0;

	if (z && off) {
		psw_backtrack(km, 1, 0, 0, z, off, 0, n_col, tlen - 1, qlen - 1,
		              m_cigar_, n_cigar_, cigar_);
	}

	if (z) kfree(km, z);
	if (off) kfree(km, off);
	kfree(km, go_q); kfree(km, ge_q); kfree(km, go_t); kfree(km, ge_t);
	kfree(km, a);
	kfree(km, qp); kfree(km, tf); kfree(km, qbf); kfree(km, tbf);
	return score;
}

float psw_gg2_ps(void *km, int qlen, const uint8_t *query,
                 int tlen, const psw_prof_t *target,
                 int8_t m, const int8_t *mat,
                 int8_t gapo, int8_t gape, int w,
                 int *m_cigar_, int *n_cigar_, uint32_t **cigar_)
{
	uvxy_t *a;
	float *tp, *tbf;
	float *go_t, *ge_t;
	int32_t r, t, n_col, *off = 0;
	float score = PSW_NEG_INF_F, H0 = 0.0f;
	int32_t last_H0_t = 0;
	uint8_t *z = 0;
	const float fgapo = (float)gapo, fgape = (float)gape;
	const float go_q = (float)gapo, ge_q = (float)gape;

	if (query == 0 || target == 0 || mat == 0) return PSW_NEG_INF_F;
	if (target->prof == 0) return PSW_NEG_INF_F;
	if (qlen < 0 || tlen < 0 || m <= 0) return PSW_NEG_INF_F;
	if (target->len < tlen || target->dim < m) return PSW_NEG_INF_F;

	for (t = 0; t < qlen; ++t)
		if ((int)query[t] < 0 || (int)query[t] >= m) return PSW_NEG_INF_F;

	if (w < 0) w = tlen > qlen ? tlen : qlen;
	n_col = w + 1 < tlen ? w + 1 : tlen;

	tp = psw_gen_tp(km, tlen, target, m, mat);
	if (tp == 0) return PSW_NEG_INF_F;
	tbf = psw_gen_base_freq(km, tlen, target, m);
	if (tbf == 0) {
		kfree(km, tp);
		return PSW_NEG_INF_F;
	}
	go_t = (float*)kmalloc(km, (size_t)tlen * sizeof(float));
	ge_t = (float*)kmalloc(km, (size_t)tlen * sizeof(float));
	a = (uvxy_t*)kcalloc(km, tlen + 1, sizeof(uvxy_t));
	if (go_t == 0 || ge_t == 0 || a == 0) {
		if (go_t) kfree(km, go_t);
		if (ge_t) kfree(km, ge_t);
		if (a) kfree(km, a);
		kfree(km, tp); kfree(km, tbf);
		return PSW_NEG_INF_F;
	}
	for (t = 0; t < tlen; ++t) {
		go_t[t] = fgapo * tbf[t];
		ge_t[t] = fgape * tbf[t];
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
		score = qlen == 0 ? psw_gap_only_target(go_t, ge_t, tlen) : psw_gap_only_query(&go_q, &ge_q, qlen);
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
		return score;
	}

	for (r = 0; r < qlen + tlen - 1; ++r) {
		int32_t st = 0, en = tlen - 1;
		float x1, v1;

		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r - w + 1) >> 1) st = (r - w + 1) >> 1;
		if (en > (r + w) >> 1) en = (r + w) >> 1;
		if (st > en) continue;

		if (st != 0) {
			if (r > st + st + w - 1) x1 = v1 = 0.0f;
			else {
				x1 = a[st - 1].x;
				v1 = a[st - 1].v;
			}
		} else {
			x1 = 0.0f;
			v1 = r ? go_q : 0.0f;
		}
		if (en != r) {
			if (r < en + en - w - 1)
				a[en].y = a[en].u = 0.0f;
		} else {
			a[r].y = 0.0f;
			a[r].u = r ? go_t[r - 1] : 0.0f;
		}

		if (z) {
			uint8_t *zr = z + (size_t)r * n_col;
			off[r] = st;
			for (t = st; t <= en; ++t) {
				int32_t j = r - t;
				int aidx = (int)query[j];
				float s = tp[(size_t)aidx * tlen + t];
				float score0, ax, by;
				float u1, q_open, t_open;
				float bias;
				uint8_t d;

				q_open = go_t[t];
				t_open = go_q;
				bias = q_open + ge_t[t] + t_open + ge_q;
				score0 = s + bias;
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

				ax -= (score0 - q_open);
				x1 = a[t].x;
				d |= ax > 0.0f ? 0x08 : 0;
				a[t].x = ax > 0.0f ? ax : 0.0f;

				by -= (score0 - t_open);
				d |= by > 0.0f ? 0x10 : 0;
				a[t].y = by > 0.0f ? by : 0.0f;

				zr[t - st] = d;
			}
		} else {
			for (t = st; t <= en; ++t) {
				int32_t j = r - t;
				int aidx = (int)query[j];
				float s = tp[(size_t)aidx * tlen + t];
				float score0, ax, by;
				float u1, q_open, t_open;
				float bias;

				q_open = go_t[t];
				t_open = go_q;
				bias = q_open + ge_t[t] + t_open + ge_q;
				score0 = s + bias;
				ax = x1 + v1;
				by = a[t].y + a[t].u;
				score0 = ax > score0 ? ax : score0;
				score0 = by > score0 ? by : score0;

				u1 = a[t].u;
				a[t].u = score0 - v1;
				v1 = a[t].v;
				a[t].v = score0 - u1;

				ax -= (score0 - q_open);
				x1 = a[t].x;
				a[t].x = ax > 0.0f ? ax : 0.0f;

				by -= (score0 - t_open);
				a[t].y = by > 0.0f ? by : 0.0f;
			}
		}

		if (r > 0) {
			if (last_H0_t >= st && last_H0_t <= en)
				H0 += a[last_H0_t].v - (go_q + ge_q);
			else {
				++last_H0_t;
				H0 += a[last_H0_t].u - (go_t[last_H0_t] + ge_t[last_H0_t]);
			}
		} else {
			H0 = a[0].v - (go_t[0] + ge_t[0] + go_q + ge_q);
			last_H0_t = 0;
		}
	}

	score = H0;

	if (z && off) {
		psw_backtrack(km, 1, 0, 0, z, off, 0, n_col, tlen - 1, qlen - 1,
		              m_cigar_, n_cigar_, cigar_);
	}

	if (z) kfree(km, z);
	if (off) kfree(km, off);
	kfree(km, go_t); kfree(km, ge_t); kfree(km, a);
	kfree(km, tp); kfree(km, tbf);
	return score;
}

