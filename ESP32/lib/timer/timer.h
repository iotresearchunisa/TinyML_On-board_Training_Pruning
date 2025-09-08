#ifndef TIMER_H
#define TIMER_H

#include <cstdint>
#include <inttypes.h>

#define PIN_START 14

typedef struct {
    int64_t start_time;
    int64_t end_time;
} Timer;

#define START_TIMER(t) \
do { \
    t.start_time = esp_timer_get_time(); \
    digitalWrite(PIN_START, HIGH); \
} while(0)

#define STOP_TIMER(t, fmt, ...) \
do { \
    digitalWrite(PIN_START, LOW); \
    t.end_time = esp_timer_get_time(); \
    LOG_INFO(fmt " - Time: %.7f s ", ##__VA_ARGS__, (t.end_time - t.start_time) / 1e6); \
} while(0)

#endif //TIMER_H
