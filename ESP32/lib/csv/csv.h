#ifndef CSVPARSER_H
#define CSVPARSER_H
#include <FS.h>

#define NUM_ARGS(...)  NUM_ARGS_IMPL(__VA_ARGS__, RSEQ_N())
#define NUM_ARGS_IMPL(...)  ARG_N(__VA_ARGS__)
#define ARG_N( \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, \
    _11,_12,_13,_14,_15,_16,_17,_18,_19,_20, \
    _21,_22,_23,_24,_25,_26,_27,_28,_29,_30, \
    _31,_32,_33,_34,_35,_36,_37,_38,_39,_40, \
    _41,_42,_43,_44,_45,_46,_47,_48,_49,_50, \
    N, ...) N
#define RSEQ_N() \
50,49,48,47,46,45,44,43,42,41, \
40,39,38,37,36,35,34,33,32,31, \
30,29,28,27,26,25,24,23,22,21, \
20,19,18,17,16,15,14,13,12,11, \
10,9,8,7,6,5,4,3,2,1,0

#define CLOSE_ALL_FILES(...) close_all_files(NUM_ARGS(__VA_ARGS__), __VA_ARGS__)
#define RESET_ALL_FILES(...) reset_files(NUM_ARGS(__VA_ARGS__), __VA_ARGS__)

bool open_csv(File &f, const char *base, const char *name, const char *mode);

bool csv_read(float arr[], int len, File &f);

bool csv_write(const float arr[], int len, File &f);

void close_all_files(uint32_t count, ...);

void reset_files(int count, ...);

#endif //CSVPARSER_H
