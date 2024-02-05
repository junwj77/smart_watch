#include <Wire.h>
#include <SoftwareSerial.h>
#define LOG_OUT 1
#define FFT_N 256
#include <FFT.h>
#include <avr/io.h>
#include <avr/interrupt.h>

void setup() {
  Serial.begin(115200);
  TIMSK0 = 0;
  ADCSRA = 0xe5;
  ADMUX = 0x40;
  DIDR0 = 0x01;
}

void loop() {
  Serial.flush();
  while(1) {
    cli();
    for (int i = 0 ; i < 512 ; i += 2) {
      while(!(ADCSRA & 0x10));
      ADCSRA = 0xf5;
      byte m = ADCL;
      byte j = ADCH;
      int k = (j << 8) | m;
      k -= 0x0200;
      k <<= 6;
      fft_input[i] = k;
      fft_input[i+1] = 0;
    }

    fft_window();
    fft_reorder();
    fft_run();
    fft_mag_log();
    sei();

    for (int i = 0; i < FFT_N / 2; i++) {
      int magnitude = fft_log_out[i];
      Serial.print(magnitude);
      Serial.print(" ");
    }
    Serial.println();
  }
}
