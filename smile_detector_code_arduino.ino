int ledPin = 8;  // You can change this to any digital pin

void setup() {
  Serial.begin(9600);      // Start serial communication
  pinMode(ledPin, OUTPUT); // Set LED pin as output
  digitalWrite(ledPin, LOW); // Ensure LED is off initially
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');  // Read until newline
    input.trim();  // Remove any extra whitespace

    if (input == "smile") {
      digitalWrite(ledPin, LOW);   // Turn OFF LED when smiling 😊
    } else if (input == "nosmile") {
      digitalWrite(ledPin, HIGH);  // Turn ON LED when not smiling 😐
    }
  }
}
