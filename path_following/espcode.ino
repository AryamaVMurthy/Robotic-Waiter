/*
 * ESP32 Actuator for Explicit Waypoint Following with Visual Confirmation
 *
 * Compatible with explicit_waypoint_server.py using coap-simple.h
 * Receives simple commands (TURN_LEFT/RIGHT, MOVE_FORWARD, STOP) via CoAP
 * Executes the corresponding motor action.
 * Sends periodic status updates (including IP) back to the server.
 */

#include <WiFi.h>
#include <WiFiUdp.h>
#include <coap-simple.h> // Using coap-simple library
#include <ArduinoJson.h>

// --- WiFi Configuration ---
const char* ssid = "AryVM";         // <<< SET YOUR WIFI SSID
const char* password = "Ary+2005";    // <<< SET YOUR WIFI PASSWORD

// --- CoAP Configuration ---
const char* serverIpString = "192.168.214.81"; // <<< SET THE IP ADDRESS OF THE PC RUNNING PYTHON
IPAddress serverIP;                        // Will be resolved from serverIpString
const int serverPort = 5683;               // Default CoAP port
const char* statusUri = "status";          // URI on the server to send status updates
const char* commandUri = "command";        // URI on this ESP32 to listen for commands

// --- Motor Pin Configuration (Using 4-pin control: ENA, ENB, IN1, IN2, IN3, IN4) ---
// Adjust these pins to match your hardware wiring
#define ENA 13   // Enable pin for Motor A (PWM output)
#define ENB 14   // Enable pin for Motor B (PWM output)
#define IN1 12   // Motor A input 1
#define IN2 27   // Motor A input 2
#define IN3 25   // Motor B input 1
#define IN4 33   // Motor B input 2

// --- CoAP Client/Server ---
WiFiUDP udp;
Coap coap(udp);

// --- State & Communication Timing ---
enum RobotState { IDLE, EXECUTING_TURN_LEFT, EXECUTING_TURN_RIGHT, EXECUTING_MOVE_FORWARD };
RobotState currentState = IDLE;
unsigned long lastStatusTime = 0;
const unsigned long STATUS_INTERVAL = 2000; // Send status every 2 seconds
int lastReceivedSeq = -1;
String myIP = "192.168.137.35"; // Store own IP address

// --- Function Declarations ---
void connectWiFi();
void setupCoap();
void callbackCommand(CoapPacket &packet, IPAddress ip, int port);
void setMotors(int left, int right);
void stopMotors();
void sendStatus();

void setup() {
  Serial.begin(115200);
  while (!Serial); // Wait for serial connection
  Serial.println("\nESP32 Explicit Waypoint Actuator (coap-simple) Starting...");

  // Initialize Motor Pins using ENA, ENB, IN1, IN2, IN3, and IN4
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  stopMotors(); // Ensure motors are off initially
  Serial.println("Motor pins configured using ENA, ENB, IN1, IN2, IN3, and IN4.");

  connectWiFi();

  if (WiFi.status() == WL_CONNECTED) {
    myIP = WiFi.localIP().toString(); // Store IP address
    if (!serverIP.fromString(serverIpString)) { // Convert server IP string to IPAddress
       Serial.println("Error: Invalid Server IP Address string.");
       while(1) delay(1000);
    }
    setupCoap();
    Serial.println("CoAP server started.");
    delay(500); // Allow time for network stabilization
    sendStatus(); // Send initial status ("IDLE" and IP)
    lastStatusTime = millis();
  } else {
    Serial.println("WiFi connection failed. Cannot start CoAP.");
    while(1) delay(1000);
  }
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    coap.loop(); // Process incoming CoAP messages

    // Send periodic status update
    if (millis() - lastStatusTime > STATUS_INTERVAL) {
      sendStatus();
      lastStatusTime = millis();
    }
  } else {
    // Attempt to reconnect if WiFi is lost
    Serial.println("WiFi disconnected. Attempting reconnect...");
    stopMotors(); // Stop motors if connection lost
    currentState = IDLE;
    connectWiFi();
    if(WiFi.status() == WL_CONNECTED) {
        myIP = WiFi.localIP().toString(); // Update IP if reconnected
        sendStatus(); // Optionally resend status immediately after reconnecting
        lastStatusTime = millis();
    } else {
        delay(5000); // Wait before next reconnect attempt
    }
  }
  delay(10); // Small delay to yield
}

// --- WiFi Connection ---
void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.mode(WIFI_STA); // Ensure Station mode
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi connection failed.");
  }
}

// --- CoAP Setup ---
void setupCoap() {
  // Define endpoint for receiving commands from the server
  coap.server(callbackCommand, commandUri); // Listen on coap://<esp_ip>/command
  Serial.printf("Listening for commands on /%s\n", commandUri);
  coap.start();
}

// --- CoAP Command Callback ---
void callbackCommand(CoapPacket &packet, IPAddress ip, int port) {
  Serial.print("Received CoAP command from ");
  Serial.print(ip);
  Serial.print(":");
  Serial.println(port);

  // Extract payload using packet.payload and packet.payloadlen
  char payload[100]; // Buffer for payload
  int payloadLen = packet.payloadlen;
  if (payloadLen > 0 && payloadLen < sizeof(payload)) {
    memcpy(payload, packet.payload, payloadLen);
    payload[payloadLen] = '\0'; // Null-terminate
    Serial.print("Payload: ");
    Serial.println(payload);

    // Parse JSON command
    StaticJsonDocument<128> doc; // Adjust size if needed
    DeserializationError error = deserializeJson(doc, payload);

    if (!error) {
      const char* command = doc["cmd"];
      int seq = doc["seq"] | -1; // Get sequence number, default to -1 if missing

      // Check if command and sequence number are present
      if (command && seq != -1) {
        // --- Get Speed Parameters (with defaults) ---
        int forwardSpeed = doc["speed"] | 100; // Default forward speed if missing
        int turnSpeed = doc["turn_speed"] | 80; // Default turn speed if missing
        forwardSpeed = constrain(forwardSpeed, 0, 255); // Ensure valid range
        turnSpeed = constrain(turnSpeed, 0, 255);     // Ensure valid range
        // --- End Speed Parameters ---

        lastReceivedSeq = seq; // Store the latest sequence number processed

        // Execute the command
        if (strcmp(command, "TURN_LEFT") == 0) {
          currentState = EXECUTING_TURN_LEFT;
          setMotors(-turnSpeed, turnSpeed); // Use parsed/default turn speed
          Serial.println("Action: Turning Left");
        } else if (strcmp(command, "TURN_RIGHT") == 0) {
          currentState = EXECUTING_TURN_RIGHT;
          setMotors(turnSpeed, -turnSpeed); // Use parsed/default turn speed
          Serial.println("Action: Turning Right");
        } else if (strcmp(command, "MOVE_FORWARD") == 0) {
          currentState = EXECUTING_MOVE_FORWARD;
          setMotors(forwardSpeed, forwardSpeed); // Use parsed/default forward speed
          Serial.println("Action: Moving Forward");
        } else if (strcmp(command, "STOP") == 0) {
          currentState = IDLE;
          stopMotors();
          Serial.println("Action: Stopping");
        } else {
          Serial.println("JSON parsing error: Missing 'cmd' or 'seq'."); // Speed missing is ok (defaults used)
          coap.sendResponse(ip, port, packet.messageid, "BAD_JSON", 8, COAP_BAD_REQUEST, COAP_TEXT_PLAIN, NULL, 0);
        }

        coap.sendResponse(ip, port, packet.messageid, NULL, 0, COAP_CHANGED, COAP_TEXT_PLAIN, NULL, 0);
      } else {
        Serial.println("JSON parsing error: Missing 'cmd' or 'seq'.");
        coap.sendResponse(ip, port, packet.messageid, "BAD_JSON", 8, COAP_BAD_REQUEST, COAP_TEXT_PLAIN, NULL, 0);
      }
    } else {
      Serial.print("JSON deserialization failed: ");
      Serial.println(error.c_str());
      coap.sendResponse(ip, port, packet.messageid, "JSON_ERROR", 10, COAP_BAD_REQUEST, COAP_TEXT_PLAIN, NULL, 0);
    }
  } else if (payloadLen == 0) {
      Serial.println("Empty payload received.");
      coap.sendResponse(ip, port, packet.messageid, NULL, 0, COAP_CHANGED, COAP_TEXT_PLAIN, NULL, 0);
  }
  else {
     Serial.println("Payload too large.");
     coap.sendResponse(ip, port, packet.messageid, "PAYLOAD_TOO_LARGE", 17, COAP_BAD_REQUEST, COAP_TEXT_PLAIN, NULL, 0);
  }
}

// --- Motor Control using ENA/ENB and IN1, IN2, IN3, IN4 ---
// 'left' and 'right' are speed values ranging from -255 (full reverse) to +255 (full forward)
void setMotors(int left, int right) {
  // Ensure speed values are within limits
  left = constrain(left, -255, 255);
  right = constrain(right, -255, 255);

  // --- Left Motor Control (Motor A) using ENA, IN1, IN2 ---
  if (left > 0) { // Forward
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, left);
  } else if (left < 0) { // Backward
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    analogWrite(ENA, -left);
  } else { // Stop
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, 0);
  }

  // --- Right Motor Control (Motor B) using ENB, IN3, IN4 ---
  if (right > 0) { // Forward
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
    analogWrite(ENB, right);
  } else if (right < 0) { // Backward
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    analogWrite(ENB, -right);
  } else { // Stop
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
    analogWrite(ENB, 0);
  }
}

// Stop both motors
void stopMotors() {
  setMotors(0, 0);
}

// --- Status Sending ---
void sendStatus() {
  if (WiFi.status() != WL_CONNECTED || serverIP[0] == 0) {
    return;
  }

  // Determine status string based on current state
  const char* statusStr;
  switch(currentState) {
    case IDLE:                    statusStr = "READY"; break;
    case EXECUTING_MOVE_FORWARD:  statusStr = "EXECUTING_MOVE_FORWARD"; break;
    case EXECUTING_TURN_LEFT:     statusStr = "EXECUTING_TURN_LEFT"; break;
    case EXECUTING_TURN_RIGHT:    statusStr = "EXECUTING_TURN_RIGHT"; break;
    default:                      statusStr = "UNKNOWN"; break;
  }

  // Create JSON payload: {"status": "...", "ip": "..."}
  StaticJsonDocument<128> doc;
  doc["status"] = statusStr;
  doc["ip"] = myIP; // Include ESP32's IP address

  char buffer[128];
  size_t n = serializeJson(doc, buffer);

  if (n > 0) {
    Serial.print("Sending status to server: ");
    Serial.println(buffer);
    // Send CoAP PUT request to the server's /status endpoint
    coap.put(serverIP, serverPort, statusUri, buffer);
  } else {
    Serial.println("Error serializing status JSON.");
  }
}
