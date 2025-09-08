#include "log.h"

#include <FS.h>
#include <SD.h>

File log_file;
LiquidCrystal_I2C * lcd;

bool initLogFile(const char *path) {
    if (CURRENT_LOG_LEVEL == LOG_FILE_ONLY || CURRENT_LOG_LEVEL == LOG_VERBOSE) {
        File dir = SD.open(path);
        if (!dir || !dir.isDirectory()) return false;

        int maxIndex = 0;
        while (true) {
            File entry = dir.openNextFile();
            if (!entry) break;

            const char *name = entry.name();
            if (strncmp(name, "log_", 4) == 0 && strstr(name, ".txt")) {
                int index = atoi(name + 4);
                if (index > maxIndex) maxIndex = index;
            }

            entry.close();
        }

        dir.close();

        char filename[32];
        snprintf(filename, sizeof(filename), "%s/log_%d.txt", path, maxIndex + 1);
        log_file = SD.open(filename, FILE_WRITE);
        if (!log_file) return false;
    }
    return true;
}
