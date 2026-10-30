void setup() {
    pinMode(7, OUTPUT);  
    Serial.begin(9600);  
}

void loop() {
    if (Serial.available()) {
        int state = Serial.read();  
        if (state == '1') {
            digitalWrite(7, HIGH);  // Light ON
        } else if (state == '0') {
            digitalWrite(7, LOW);   // Light OFF
        }
    }
}

