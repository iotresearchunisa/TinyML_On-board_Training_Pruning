#include <Arduino.h>
#include <SD.h>
#include <FS.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#include "csv.h"
#include "log.h"
#include "pietro.h"

#define PIN_START 14

LogLevel CURRENT_LOG_LEVEL = LOG_VERBOSE;

const int chipSelect = 5;

void train_task(void *params) {
    int layers[] = {35, 28, 128, 28, 9};
    int prune_epochs[] = {0};
    MLP *mlp = create_mlp(5, layers, (REAL) 0.0001);

    train_with_pruning_stream(
        mlp,
        35, 1, // x_cols, y_cols
        144000, // numero campioni train
        5, // epoche
        36000, // numero campioni test
        30.0f, // prune %
        prune_epochs,
        0.99f, // lr decay
        1, // quante epoche di pruning
        1 // log ogni N epoche
    );

    free_mlp(mlp);
    vTaskDelete(NULL);
}

void setup() {
    pinMode(PIN_START, OUTPUT);
    digitalWrite(PIN_START, LOW);
    Wire.begin(18, 17);
    SPI.begin(4, 7, 6);

    uint32_t seed = 758057339;
    srand(seed);

    Serial.begin(115200);
    while (!Serial) { ; }
    Serial.println("");
    Serial.flush();

    lcd = new LiquidCrystal_I2C(0x27, 20, 4);
    lcd->init();
    lcd->backlight();

    if (!SD.begin(chipSelect)) {
        FAILURE("MicroSD Card Mount Failed");
    }

    if (!initLogFile(BASE_DIR)) {
        FAILURE("Init Log Failed");
    }

    xTaskCreatePinnedToCore(
        train_task, // funzione
        "TrainTask", // nome task
        32768, // stack size in byte
        NULL, // parametri
        1, // priorità
        NULL, // handle
        1 // core 1
    );
}

void loop() {
    vTaskDelete(NULL);
}
