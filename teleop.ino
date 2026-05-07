// 4 pots + 1 button ESP32 teleop sketch.
// Sends: yaw,shoulder,elbow,endlink,button

int yawPot = 36;       // VP
int shoulderPot = 39;  // VN
int elbowPot = 34;     // D34
int endPot = 35;       // D35

int buttonPin = 12;    // Button to GND when pressed.

int zeroYaw = 0;
int zeroShoulder = 0;
int zeroElbow = 0;
int zeroEnd = 0;
 
unsigned long t0 = 0;

int readAvg(int pin) {
  long sum = 0;
  for (int i = 0; i < 20; i++) {
    sum += analogRead(pin);
    delay(2);
  }
  return (int)(sum / 20);
}

int toDeg300(int raw) {
  raw = constrain(raw, 0, 4095);
  return map(raw, 0, 4095, 0, 300);
}

void setup() {
  Serial.begin(115200);

  analogSetAttenuation(ADC_11db);  // 0-3.3V
  analogReadResolution(12);        // 0-4095

  pinMode(buttonPin, INPUT_PULLUP);

  t0 = millis();
  Serial.println("CALIBRATE 5s: hold arm at ZERO position (VP,VN,D34,D35)...");
}

void loop() {
  unsigned long now = millis();

  int btn = (digitalRead(buttonPin) == LOW) ? 1 : 0;

  // First 5 seconds: keep updating zero offsets.
  if (now - t0 < 5000) {
    zeroYaw = toDeg300(readAvg(yawPot));
    zeroShoulder = toDeg300(readAvg(shoulderPot));
    zeroElbow = toDeg300(readAvg(elbowPot));
    zeroEnd = toDeg300(readAvg(endPot));

    Serial.print("CAL ");
    Serial.print((5000 - (now - t0)) / 1000);
    Serial.print("  ");
    Serial.print(zeroYaw);
    Serial.print(",");
    Serial.print(zeroShoulder);
    Serial.print(",");
    Serial.print(zeroElbow);
    Serial.print(",");
    Serial.print(zeroEnd);
    Serial.print(",");
    Serial.println(btn);

    delay(50);
    return;
  }

  // After calibration: signed deltas around zero.
  int yaw = toDeg300(readAvg(yawPot)) - zeroYaw;
  int shoulder = toDeg300(readAvg(shoulderPot)) - zeroShoulder;
  int elbow = toDeg300(readAvg(elbowPot)) - zeroElbow;
  int endlink = toDeg300(readAvg(endPot)) - zeroEnd;

  Serial.print(yaw);
  Serial.print(",");
  Serial.print(shoulder);
  Serial.print(",");
  Serial.print(elbow);
  Serial.print(",");
  Serial.print(endlink);
  Serial.print(",");
  Serial.println(btn);

  delay(50);
}
