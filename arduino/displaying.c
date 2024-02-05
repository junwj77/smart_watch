#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SoftwareSerial.h>
SoftwareSerial serial_connection(3, 2);
#define BUFFER_SIZE 64
char inData[BUFFER_SIZE];
char inChar=-1;
int count=0;
int i=0;
const int vibrationPin = 9;
LiquidCrystal_I2C lcd(0x27, 16, 2);
typedef int MyVariableType;

void setup() {
  Serial.begin(9600);
  serial_connection.begin(9600);
  serial_connection.println("Ready");


  pinMode(vibrationPin, OUTPUT);

  lcd.begin(16, 2);
  lcd.setBacklight(HIGH);
  Serial.println("Ready..");
}

void loop() {
  byte byte_count=serial_connection.available();
  if(byte_count)
  {
    Serial.println("Incoming Data");
    int first_bytes=byte_count;
    int remaining_bytes=0;
    if(first_bytes>=BUFFER_SIZE-1)
    {
      remaining_bytes=byte_count-(BUFFER_SIZE-1);
    }
    for(i=0;i<first_bytes;i++)
    {
      inChar=serial_connection.read();
      inData[i]=inChar;
    }
inData[i]='\0';
    for(i=0;i<remaining_bytes;i++)
    {
      inChar=serial_connection.read();
    }

    Serial.println(String(inData));
    String dataString = String(inData);
    int result = dataString.toInt();

    sendDataToLCD(result);
    delay(100);

    switch (result) {
      case 0:
        stopVibration();
        break;
      case 1:
        vibratePattern1();
        break;
      case 2:
        vibratePattern2();
        break;
      case 3:
        vibratePattern3();
        break;
      case 4:
        vibratePattern4();
        break;
      default:
stopVibration();
        break;
    serial_connection.println("Hello from Blue "+String(count));
    }
    delay(100);
  }
}

void sendDataToLCD(int value) {
  char soundType[20];

  switch (value) {
    case 0:
      lcd.clear();
      lcd.noBacklight();
      lcd.noDisplay();
      break;
    case 1:
      lcd.clear();
      lcd.setBacklight(HIGH);
      strcpy(soundType, "  Fire Alarm");
      break;
    case 2:
      lcd.clear();
      lcd.setBacklight(HIGH);
      strcpy(soundType, "     Scream");
      break;
    case 3:
      lcd.clear();
      lcd.setBacklight(HIGH);
      strcpy(soundType, "  Dog Barking");
      break;
    case 4:
      lcd.clear();
      lcd.setBacklight(HIGH);
      strcpy(soundType, "  Broken Glass");
      break;
    default:
lcd.clear();
      lcd.noBacklight();
      lcd.noDisplay();
      break;
  }
  lcd.setCursor(0, 0);
  lcd.print(soundType);
  lcd.display();
  delay(1000);
}

void stopVibration() {
  digitalWrite(vibrationPin, LOW);
  Serial.println("무음");
}
void vibratePattern1() {
  vibrate(3000, 500);
  Serial.println("진동 패턴 1이 실행되었습니다.");
}
void vibratePattern2() {
  for (int i = 0; i < 3; i++) {
    vibrate(1000, 500);
  }
  Serial.println("진동 패턴 2가 실행되었습니다.");
}
void vibratePattern3() {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      vibrate(100, 50);
    }
    delay(700);
  }
  Serial.println("진동 패턴 3이 실행되었습니다.");
}
void vibratePattern4() {
  for (int i = 0; i < 15; i++) {
    vibrate(100, 50);
  }
  Serial.println("진동 패턴 4가 실행되었습니다.");
}
void vibrate(int onDuration, int offDuration) {
  digitalWrite(vibrationPin, HIGH);
  delay(onDuration);
  digitalWrite(vibrationPin, LOW);
  delay(offDuration);
}

