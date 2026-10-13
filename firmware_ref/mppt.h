/*
 * Reference C port of the pvsim MPPT algorithms (pvsim/mppt/*.py).
 *
 * Every function here is a line-for-line translation of the Python step()
 * so firmware and simulator take the same duty decisions for the same
 * measurements.  tests/test_firmware_ref.py compiles this file (with
 * MPPT_USE_DOUBLE) and checks that the duty sequences match the Python ones.
 *
 * Usage (once per MPPT period, with the measured panel voltage and current):
 *
 *     mppt_t m;
 *     mppt_params_t p;
 *     mppt_default_params(&p);
 *     p.d_init = 0.3f;
 *     mppt_init(&m, MPPT_PO, &p);
 *     ...
 *     duty = mppt_step(&m, v_meas, i_meas);   // apply duty until next period
 *
 * Parameters and state are flat structs holding the union of the fields of
 * all algorithms (each algorithm reads only its own); the field names match
 * the Python Params / State dataclasses.
 *
 * No dynamic memory, no libc beyond fabs - fits an ESP32 / any MCU.
 */
#ifndef PVSIM_MPPT_H
#define PVSIM_MPPT_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MPPT_USE_DOUBLE
typedef double mppt_real;
#else
typedef float mppt_real; /* ESP32 has a single-precision FPU */
#endif

#if defined(_WIN32) && defined(MPPT_BUILD_DLL)
#define MPPT_API __declspec(dllexport)
#else
#define MPPT_API
#endif

typedef enum {
    MPPT_PO = 0,               /* "po"               */
    MPPT_PO_VARIABLE = 1,      /* "po_variable"      */
    MPPT_INCCOND = 2,          /* "inccond"          */
    MPPT_INCCOND_VARIABLE = 3, /* "inccond_variable" */
    MPPT_GLOBAL_SCAN = 4,      /* "global_scan"      */
    MPPT_MANUAL = 5            /* "manual"           */
} mppt_algo_t;

enum { MPPT_TRACK = 0, MPPT_SCAN = 1 }; /* global_scan modes */

typedef struct {
    /* BaseParams */
    mppt_real d_init;
    mppt_real d_min;
    mppt_real d_max;
    mppt_real step;
    /* POVarParams / IncCondVarParams */
    mppt_real n_gain;
    mppt_real step_min;
    mppt_real step_max;
    /* IncCondParams */
    mppt_real eps_g;
    mppt_real eps_v;
    mppt_real eps_i;
    /* GlobalScanParams */
    int scan_points;
    int scan_period;
    mppt_real dp_trigger;
    mppt_real ref_alpha;
    int scan_on_start;
    /* ManualParams */
    mppt_real duty;
} mppt_params_t;

typedef struct {
    /* BaseState */
    mppt_real d;
    mppt_real v_prev;
    mppt_real i_prev;
    mppt_real p_prev;
    int n;
    /* POState / GlobalScanState */
    int dir;
    /* GlobalScanState */
    int mode;
    int scan_idx;
    mppt_real best_p;
    mppt_real best_d;
    int since_scan;
    mppt_real p_ref;
    int n_scans;
} mppt_state_t;

typedef struct {
    int algo; /* mppt_algo_t */
    mppt_params_t p;
    mppt_state_t s;
} mppt_t;

/* Python dataclass defaults (identical for every algorithm). */
MPPT_API void mppt_default_params(mppt_params_t *p);
/* Copy the parameters and reset; returns 0, or -1 for an unknown algorithm. */
MPPT_API int mppt_init(mppt_t *m, int algo, const mppt_params_t *p);
/* MPPTAlgorithm.reset(): fresh state, duty = clip(d_init) (manual: clip(duty)). */
MPPT_API void mppt_reset(mppt_t *m);
/* One MPPT period: measured V [V] and I [A] in, next duty out. */
MPPT_API mppt_real mppt_step(mppt_t *m, mppt_real v, mppt_real i);
/* manual only: ManualDuty.set_duty(). */
MPPT_API void mppt_set_duty(mppt_t *m, mppt_real d);

/* sizes, so a host (ctypes) can check its struct mirror */
MPPT_API int mppt_sizeof_params(void);
MPPT_API int mppt_sizeof(void);

#ifdef __cplusplus
}
#endif

#endif /* PVSIM_MPPT_H */
