#ifndef LOG_H
#define LOG_H

#include <FS.h>
#include <stdio.h>
#include <LiquidCrystal_I2C.h>

typedef enum {
    LOG_NONE = 0,
    LOG_FILE_ONLY,
    LOG_ERROR_ONLY,
    LOG_WARNINGS,
    LOG_VERBOSE_NO_FILE,
    LOG_VERBOSE,
} LogLevel;

extern LogLevel CURRENT_LOG_LEVEL;
extern File log_file;
extern LiquidCrystal_I2C *lcd;

bool initLogFile(const char *path);

static char buf[81];

#define LOG_ERROR(fmt, ...) do { \
  if (CURRENT_LOG_LEVEL >= LOG_ERROR_ONLY) { \
    fprintf(stdout, "[ERROR] " fmt "\n", ##__VA_ARGS__); \
    fflush(stdout); \
    if (lcd) { \
      snprintf(buf, sizeof(buf), fmt, ##__VA_ARGS__); \
      lcd->clear(); \
      lcd->setCursor(0, 0); \
      lcd->print("[ERROR]"); \
      for (int i = 0; i < 4; ++i) { \
        lcd->setCursor(0, i); \
        for (int j = 0; j < 20; ++j) { \
          char c = buf[i * 20 + j]; \
          if (c == '\0') break; \
          lcd->print(c); \
        } \
      } \
    } \
  } \
  if ((CURRENT_LOG_LEVEL == LOG_FILE_ONLY || CURRENT_LOG_LEVEL == LOG_VERBOSE) && \
      log_file) { \
    log_file.printf("[ERROR] " fmt "\n", ##__VA_ARGS__); \
    log_file.flush(); \
  } \
} while (0);

#define LOG_WARN(fmt, ...) do { \
  if (CURRENT_LOG_LEVEL >= LOG_WARNINGS) { \
    fprintf(stdout, "[WARN] " fmt "\n", ##__VA_ARGS__); \
    fflush(stdout); \
    if (lcd) { \
      snprintf(buf, sizeof(buf), fmt, ##__VA_ARGS__); \
      lcd->clear(); \
      lcd->setCursor(0, 0); \
      lcd->print("[WARN]"); \
      for (int i = 0; i < 4; ++i) { \
        lcd->setCursor(0, i); \
        for (int j = 0; j < 20; ++j) { \
          char c = buf[i * 20 + j]; \
          if (c == '\0') break; \
          lcd->print(c); \
        } \
      } \
    } \
  } \
  if ((CURRENT_LOG_LEVEL == LOG_FILE_ONLY || CURRENT_LOG_LEVEL == LOG_VERBOSE) && \
      log_file) { \
    log_file.printf("[WARN] " fmt "\n", ##__VA_ARGS__); \
    log_file.flush(); \
  } \
} while (0);

#define LOG_INFO(fmt, ...) do { \
  if (CURRENT_LOG_LEVEL >= LOG_VERBOSE_NO_FILE) { \
    fprintf(stdout, "[INFO] " fmt "\n", ##__VA_ARGS__); \
    fflush(stdout); \
    if (lcd) { \
      snprintf(buf, sizeof(buf), fmt, ##__VA_ARGS__); \
      lcd->clear(); \
      lcd->setCursor(0, 0); \
      lcd->print("[INFO]"); \
      for (int i = 0; i < 4; ++i) { \
        lcd->setCursor(0, i); \
        for (int j = 0; j < 20; ++j) { \
          char c = buf[i * 20 + j]; \
          if (c == '\0') break; \
          lcd->print(c); \
        } \
      } \
    } \
  } \
  if ((CURRENT_LOG_LEVEL == LOG_FILE_ONLY || CURRENT_LOG_LEVEL == LOG_VERBOSE) && \
      log_file) { \
    log_file.printf("[INFO] " fmt "\n", ##__VA_ARGS__); \
    log_file.flush(); \
  } \
} while (0);

  #define FAILURE(msg, ...) do { \
    LOG_ERROR(msg, ##__VA_ARGS__); \
    while (1) delay(3600000); \
  } while (0);

#endif //LOG_H
