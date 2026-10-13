/*
 * Reference C port of pvsim/mppt (see mppt.h).  Each function mirrors the
 * Python method named in its comment; keep the two in sync.
 */
#include "mppt.h"

#include <math.h>

#define R(x) ((mppt_real)(x))

static int imax(int a, int b) { return a > b ? a : b; }

/* MPPTAlgorithm.clip */
static mppt_real clip(const mppt_t *m, mppt_real d)
{
    if (d < m->p.d_min)
        return m->p.d_min;
    if (d > m->p.d_max)
        return m->p.d_max;
    return d;
}

/* MPPTAlgorithm._remember */
static void remember(mppt_t *m, mppt_real v, mppt_real i)
{
    m->s.v_prev = v;
    m->s.i_prev = i;
    m->s.p_prev = v * i;
    m->s.n += 1;
}

void mppt_default_params(mppt_params_t *p)
{
    p->d_init = R(0.5);
    p->d_min = R(0.0);
    p->d_max = R(0.95);
    p->step = R(0.005);
    p->n_gain = R(0.002);
    p->step_min = R(0.0005);
    p->step_max = R(0.02);
    p->eps_g = R(0.0);
    p->eps_v = R(1e-6);
    p->eps_i = R(1e-6);
    p->scan_points = 40;
    p->scan_period = 1000;
    p->dp_trigger = R(0.25);
    p->ref_alpha = R(0.02);
    p->scan_on_start = 1;
    p->duty = R(0.5);
}

/* ------------------------------------------------------------ global_scan.py */

/* GlobalScanPO._scan_duty */
static mppt_real scan_duty(const mppt_t *m, int k)
{
    int n = imax(m->p.scan_points, 2);
    return m->p.d_min + k * (m->p.d_max - m->p.d_min) / (n - 1);
}

/* GlobalScanPO._start_scan */
static void start_scan(mppt_t *m)
{
    mppt_state_t *s = &m->s;
    s->mode = MPPT_SCAN;
    s->scan_idx = 0;
    s->best_p = R(-1.0);
    s->best_d = s->d;
    s->n_scans += 1;
    s->d = clip(m, scan_duty(m, 0));
}

/* GlobalScanPO.step */
static mppt_real global_scan_step(mppt_t *m, mppt_real v, mppt_real i)
{
    mppt_state_t *s = &m->s;
    const mppt_params_t *p = &m->p;
    mppt_real pw = v * i;
    int rescan;
    if (s->mode == MPPT_SCAN) {
        if (pw > s->best_p) {
            s->best_p = pw;
            s->best_d = s->d;
        }
        s->scan_idx += 1;
        if (s->scan_idx < imax(p->scan_points, 2)) {
            s->d = clip(m, scan_duty(m, s->scan_idx));
        } else {
            s->mode = MPPT_TRACK;
            s->d = s->best_d;
            s->since_scan = 0;
            s->p_ref = s->best_p;
            s->dir = 1;
            pw = s->best_p; /* next comparison is against the scan optimum */
        }
        remember(m, v, i);
        s->p_prev = pw;
        return s->d;
    }

    /* TRACK: local P&O plus rescan triggers */
    s->since_scan += 1;
    rescan = p->scan_period > 0 && s->since_scan >= p->scan_period;
    if (p->dp_trigger > 0 && s->p_ref > 0 && fabs(pw - s->p_ref) > p->dp_trigger * s->p_ref)
        rescan = 1;
    if (rescan) {
        start_scan(m);
        remember(m, v, i);
        return s->d;
    }
    s->p_ref += p->ref_alpha * (pw - s->p_ref);
    if (pw - s->p_prev < R(0.0))
        s->dir = -s->dir;
    s->d = clip(m, s->d + s->dir * p->step);
    remember(m, v, i);
    return s->d;
}

/* --------------------------------------------------------------------- po.py */

/* PerturbObserve._step_size / PerturbObserveVariable._step_size */
static mppt_real po_step_size(const mppt_t *m, mppt_real dv, mppt_real dp)
{
    const mppt_params_t *p = &m->p;
    mppt_real st;
    if (m->algo != MPPT_PO_VARIABLE)
        return p->step;
    if (fabs(dv) < R(1e-6))
        return p->step_min;
    st = p->n_gain * R(fabs(dp / dv));
    if (st < p->step_min)
        st = p->step_min;
    if (st > p->step_max)
        st = p->step_max;
    return st;
}

/* PerturbObserve.step */
static mppt_real po_step(mppt_t *m, mppt_real v, mppt_real i)
{
    mppt_state_t *s = &m->s;
    mppt_real pw = v * i;
    if (s->n > 0) {
        mppt_real dp = pw - s->p_prev;
        if (dp < R(0.0))
            s->dir = -s->dir;
        s->d = clip(m, s->d + s->dir * po_step_size(m, v - s->v_prev, dp));
    } else {
        s->d = clip(m, s->d + s->dir * m->p.step);
    }
    remember(m, v, i);
    return s->d;
}

/* ---------------------------------------------------------------- inccond.py */

/* IncrementalConductance._step_size / IncrementalConductanceVariable._step_size */
static mppt_real ic_step_size(const mppt_t *m, mppt_real dv, mppt_real di, mppt_real v, mppt_real i)
{
    const mppt_params_t *p = &m->p;
    mppt_real dp_dv, st;
    if (m->algo != MPPT_INCCOND_VARIABLE)
        return p->step;
    if (fabs(dv) < p->eps_v)
        return p->step_min;
    dp_dv = i + v * di / dv;
    st = p->n_gain * R(fabs(dp_dv));
    if (st < p->step_min)
        st = p->step_min;
    if (st > p->step_max)
        st = p->step_max;
    return st;
}

/* IncrementalConductance.step */
static mppt_real inccond_step(mppt_t *m, mppt_real v, mppt_real i)
{
    mppt_state_t *s = &m->s;
    const mppt_params_t *p = &m->p;
    mppt_real dv, di, g;
    int direction;
    if (v <= p->eps_v) {
        /* input collapsed: back off the load */
        s->d = clip(m, s->d - R(4.0) * p->step);
        remember(m, v, i);
        return s->d;
    }
    if (s->n == 0) {
        s->d = clip(m, s->d + p->step);
        remember(m, v, i);
        return s->d;
    }

    dv = v - s->v_prev;
    di = i - s->i_prev;
    direction = 0; /* +1 raise V, -1 lower V */
    if (fabs(dv) < p->eps_v) {
        if (di > p->eps_i)
            direction = 1;
        else if (di < -p->eps_i)
            direction = -1;
    } else {
        g = di / dv + i / v;
        if (g > p->eps_g)
            direction = 1;
        else if (g < -p->eps_g)
            direction = -1;
    }
    if (direction != 0)
        s->d = clip(m, s->d - direction * ic_step_size(m, dv, di, v, i));
    remember(m, v, i);
    return s->d;
}

/* ------------------------------------------------------------ manual_duty.py */

/* ManualDuty.step */
static mppt_real manual_step(mppt_t *m, mppt_real v, mppt_real i)
{
    remember(m, v, i);
    return m->s.d;
}

void mppt_set_duty(mppt_t *m, mppt_real d)
{
    m->p.duty = d;
    m->s.d = clip(m, d);
}

/* ------------------------------------------------------------------- base.py */

void mppt_reset(mppt_t *m)
{
    mppt_state_t zero = {0};
    m->s = zero;
    m->s.dir = 1;
    m->s.mode = MPPT_TRACK;
    m->s.d = clip(m, m->p.d_init);
    if (m->algo == MPPT_MANUAL)
        m->s.d = clip(m, m->p.duty);
    if (m->algo == MPPT_GLOBAL_SCAN && m->p.scan_on_start)
        start_scan(m);
}

int mppt_init(mppt_t *m, int algo, const mppt_params_t *p)
{
    if (algo < MPPT_PO || algo > MPPT_MANUAL)
        return -1;
    m->algo = algo;
    m->p = *p;
    mppt_reset(m);
    return 0;
}

mppt_real mppt_step(mppt_t *m, mppt_real v, mppt_real i)
{
    switch (m->algo) {
    case MPPT_PO:
    case MPPT_PO_VARIABLE:
        return po_step(m, v, i);
    case MPPT_INCCOND:
    case MPPT_INCCOND_VARIABLE:
        return inccond_step(m, v, i);
    case MPPT_GLOBAL_SCAN:
        return global_scan_step(m, v, i);
    default:
        return manual_step(m, v, i);
    }
}

int mppt_sizeof_params(void) { return (int)sizeof(mppt_params_t); }
int mppt_sizeof(void) { return (int)sizeof(mppt_t); }
