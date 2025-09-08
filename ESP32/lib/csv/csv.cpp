#include "csv.h"
#include <FS.h>
#include <SD.h>
#include <stdarg.h>

bool open_csv(File &f, const char *base, const char *name, const char *mode) {
    char path[32];
    snprintf(path, sizeof(path), "%s/%s", base, name);
    f = SD.open(path, mode);
    return f ;
}

bool csv_read(float arr[], int len, File &f) {
    if (!f || len <= 0) return false;

    char value_buf[64];
    int buf_pos = 0;
    int count = 0;

    while (count < len) {
        if (!f.available()) {
            f.seek(0);
            if (!f.available()) break;
        }

        const char c = f.read();

        if (c == ',' || c == '\n' || c == '\r') {
            if (buf_pos > 0) {
                value_buf[buf_pos] = '\0';
                arr[count] = atof(value_buf);
                buf_pos = 0;
                count++;
            }

            if (c == '\r' && f.available() && f.peek() == '\n') {
                f.read();
            }
        } else if (buf_pos < sizeof(value_buf) - 1) {
            value_buf[buf_pos++] = c;
        }
    }

    return count == len;
}

bool csv_write(const float arr[], int len, File &f) {
    if (!arr || len <= 0 || !f) return false;

    for (int i = 0; i < len; i++) {
        f.print(arr[i], 7);
        if (i < len - 1) {
            f.print(",");
        } else {
            f.println();
        }
    }

    return true;
}


void close_all_files(uint32_t count, ...) {
    va_list args;
    va_start(args, count);

    for (int i = 0; i < count; i++) {
        File *f = va_arg(args, File *);
        if (f) f->close();
    }

    va_end(args);
}

void reset_files(int count, ...) {
    va_list args;
    va_start(args, count);

    for (int i = 0; i < count; i++) {
        File *f = va_arg(args, File *);
        if (f) {
            f->seek(0);
        }
    }

    va_end(args);
}
