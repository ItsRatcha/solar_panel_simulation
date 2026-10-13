/*
 * ESP32 hardware-in-the-loop MPPT for pvsim (Arduino core for ESP32).
 *
 * The simulator plays panel + converter + load + INA226; the ESP32 runs the
 * real firmware MPPT (firmware_ref/mppt.c) and answers over USB serial with
 * the line protocol of pvsim/mppt/serial_esp32.py:
 *
 *     host -> "V,I\n"      measured panel voltage [V] and current [A]
 *     ESP  -> "D\n"        next duty cycle (0..1)
 *     host -> "R\n"        reset the algorithm      ESP -> "D\n" (initial duty)
 *     host -> "A,<n>\n"    select algorithm n (mppt_algo_t) and reset  ESP -> "D\n"
 *     ESP  -> "# ..."      comment / debug line (ignored by the host)
 *
 * Setup:
 *   1. Copy ../mppt.h and ../mppt.c next to this .ino (the Arduino IDE only
 *      compiles files in the sketch folder).
 *   2. Flash, then on the PC:
 *        from pvsim.mppt import create_mppt
 *        algo = create_mppt("serial_esp32", port="COM5", d_init=0.3)
 *      and use `algo` like any other MPPT (harness, LiveChannel, scenarios).
 *
 * The duty is also written to an LEDC PWM pin so the gate signal can be
 * checked on a scope; on a real board the same mppt_step() would be fed from
 * the INA226 instead of the serial line.
 */
#include "mppt.h"

static const long BAUD = 115200;
static const int PWM_PIN = 25;
static const int PWM_FREQ = 40000;  // matches BuckParams.f_sw
static const int PWM_BITS = 10;

static mppt_t m;
static char line[64];
static size_t len = 0;

static void pwm_begin() {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(PWM_PIN, PWM_FREQ, PWM_BITS);
#else
  ledcSetup(0, PWM_FREQ, PWM_BITS);
  ledcAttachPin(PWM_PIN, 0);
#endif
}

static void pwm_write(mppt_real d) {
  uint32_t counts = (uint32_t)(d * ((1 << PWM_BITS) - 1) + 0.5f);
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(PWM_PIN, counts);
#else
  ledcWrite(0, counts);
#endif
}

static void reply(mppt_real d) {
  pwm_write(d);
  Serial.println((double)d, 6);
}

static void select_algo(int algo) {
  mppt_params_t p;
  mppt_default_params(&p);
  p.d_init = 0.3f;  // keep in sync with the host's d_init
  if (mppt_init(&m, algo, &p) != 0) {
    Serial.println("# unknown algorithm, using P&O");
    mppt_init(&m, MPPT_PO, &p);
  }
}

static void handle(char *s) {
  if (s[0] == 'R') {
    mppt_reset(&m);
    reply(m.s.d);
  } else if (s[0] == 'A') {
    char *comma = strchr(s, ',');
    select_algo(comma ? atoi(comma + 1) : MPPT_PO);
    reply(m.s.d);
  } else {
    char *comma = strchr(s, ',');
    if (!comma) {
      Serial.println("# expected V,I");
      return;
    }
    *comma = '\0';
    mppt_real v = (mppt_real)atof(s);
    mppt_real i = (mppt_real)atof(comma + 1);
    reply(mppt_step(&m, v, i));
  }
}

void setup() {
  Serial.begin(BAUD);
  pwm_begin();
  select_algo(MPPT_PO);
  pwm_write(m.s.d);
  Serial.println("# pvsim ESP32 MPPT HIL ready");
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r')
      continue;
    if (c == '\n') {
      line[len] = '\0';
      if (len > 0)
        handle(line);
      len = 0;
    } else if (len < sizeof(line) - 1) {
      line[len++] = c;
    }
  }
}
