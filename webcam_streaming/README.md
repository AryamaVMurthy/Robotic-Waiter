# Low-Latency WebRTC Video Streaming (Mobile to Laptop)

This application demonstrates low-latency video streaming from a mobile device's camera to a laptop browser using WebRTC and PeerJS. It simplifies the connection process using QR codes.

## Features

*   Streams video directly between devices (peer-to-peer) over the local network (when possible) or internet.
*   Uses PeerJS for simplified WebRTC signaling.
*   Utilizes QR codes for easy connection setup (scan code on laptop with mobile).
*   Allows selecting between available cameras (front, back, wide-angle - based on browser detection) on the sender device.
*   Responsive design for usability on both desktop and mobile.

## Prerequisites

1.  **Modern Web Browser:** A browser supporting WebRTC (e.g., Chrome, Firefox, Safari, Brave) on both the sending (mobile) and receiving (laptop) devices. (Note: Brave Browser worked best on mobile during testing).
2.  **Python 3:** Required to run the simple local web server. Download from [python.org](https://www.python.org/downloads/) if you don't have it. Ensure it's added to your system's PATH during installation.
3.  **ngrok:** Required for easily creating a secure (HTTPS) tunnel to your local server, which is necessary for browsers to grant camera permissions.
    *   Download from [ngrok.com/download](https://ngrok.com/download).
    *   Alternatively, install via package managers like Chocolatey (`choco install ngrok`) or Homebrew (`brew install ngrok`).
4.  **Internet Connection:** Both devices need internet access for the initial PeerJS signaling and potentially for the WebRTC connection if a direct local link fails. For lowest latency, both devices should ideally be on the same *unrestricted* Wi-Fi network.

## Setup

1.  **Get the Code:** Download or clone this project folder (`webcam_streaming`) to your laptop.
2.  **Configure ngrok:**
    *   Sign up for a free account at [ngrok.com](https://ngrok.com/signup).
    *   Get your authtoken from the ngrok dashboard ([Your Authtoken section](https://dashboard.ngrok.com/get-started/your-authtoken)).
    *   Open a terminal or command prompt and add your authtoken:
        ```bash
        ngrok config add-authtoken <YOUR_AUTHTOKEN>
        ```
        (Replace `<YOUR_AUTHTOKEN>` with the actual token).

## Running the Application (Recommended Method: Using ngrok)

1.  **Start Local Server:**
    *   Open a terminal or command prompt.
    *   Navigate (`cd`) into the `webcam_streaming` project directory.
    *   Run the Python HTTP server:
        ```bash
        python -m http.server 8000
        ```
    *   Leave this terminal running.

2.  **Start ngrok Tunnel:**
    *   Open a **new** terminal or command prompt.
    *   Run ngrok to create a tunnel to port 8000:
        ```bash
        ngrok http 8000
        ```
    *   Leave this terminal running. ngrok will display a `Forwarding` address like `https://<random-string>.ngrok-free.app`. **This is your public URL.**

3.  **Open Receiver (Laptop):**
    *   On your laptop's browser, go to the `https://<random-string>.ngrok-free.app` URL provided by ngrok and add `/receiver.html` to the end.
        *Example: `https://abc-123.ngrok-free.app/receiver.html`*
    *   Wait a few seconds. The page should display "Your Peer ID:" and then generate a QR code below it.

4.  **Open Sender (Mobile):**
    *   On your mobile device, ensure you are connected to a network that allows the connection (Mobile Data often works best if local Wi-Fi has restrictions).
    *   Use your phone's camera app or a QR scanner app to **scan the QR code** displayed on the laptop screen.
    *   Your phone should prompt you to open the scanned URL in a browser. **Open the link.** (Brave Browser recommended if Chrome has issues).
    *   The sender page will load. The "Receiver Peer ID" field should be automatically filled.

5.  **Start Streaming:**
    *   The browser will likely ask for camera permission (to list cameras). **Allow** it.
    *   The "Select Camera" dropdown should appear. Choose the desired camera (it attempts to default to the back camera if labeled correctly).
    *   Click the "**Start & Stream Camera**" button.
    *   If prompted again for permission for the specific camera, **Allow** it.
    *   The camera preview should appear on your phone.

6.  **View Stream:**
    *   Look at the `receiver.html` page on your laptop. The live video should appear shortly.
    *   To switch cameras, simply select a different option from the dropdown on the mobile sender page.

## Alternative Method: Local Network with Self-Signed Certificate (No ngrok)

This avoids ngrok but requires bypassing browser security warnings.

1.  **Generate Certificate:**
    *   You need OpenSSL (often included with Git for Windows, macOS, Linux).
    *   In the `webcam_streaming` directory, run:
        ```bash
        openssl req -new -x509 -keyout server.pem -out server.pem -days 365 -nodes
        ```
    *   Accept defaults or enter dummy info when prompted. This creates `server.pem`.
2.  **Run HTTPS Server:**
    *   Ensure no other server is running on port 8000.
    *   In the terminal (in the project directory), run:
        ```bash
        python serve_https.py
        ```
3.  **Find Laptop IP:** Use `ipconfig` (Windows) or `ifconfig`/`ip addr` (Mac/Linux) to find your laptop's local IP address (e.g., `192.168.1.10`).
4.  **Access Pages:**
    *   **Laptop:** Open `https://localhost:8000/receiver.html`. **Accept the browser security warning** (click "Advanced", "Proceed anyway", etc.). The QR code should appear.
    *   **Mobile:** Scan the QR code. It will point to `https://<LAPTOP_IP>:8000/sender.html`. Open this link. **Accept the browser security warning** on the mobile browser too.
5.  Proceed to start the stream as in the ngrok method.

## Troubleshooting

*   **No Video Stream / ICE Failed:** Usually a firewall issue.
    *   Ensure both devices are on the same Wi-Fi network (if attempting local connection). Routers with "AP Isolation" can block this.
    *   Temporarily disable firewalls (Windows Defender, third-party antivirus) on the *laptop* to test if they block incoming WebRTC UDP ports. **Remember to re-enable them!**
    *   Try using Mobile Data on the sender device - if this works, the issue is likely your local Wi-Fi configuration.
*   **"Error accessing camera: NotReadableError" (Sender):**
    *   Close ALL other apps/tabs that might be using the camera.
    *   Restart the mobile device.
    *   Try a different browser on mobile (e.g., Brave).
    *   Check OS-level camera permissions for the browser app.
*   **Peer ID / QR Code Doesn't Appear (Receiver):**
    *   Check the browser's Developer Console (F12 -> Console) on the receiver page for errors.
    *   Ensure `python -m http.server 8000` and `ngrok http 8000` are running correctly.
    *   Verify the `<script>` tags are present at the end of `receiver.html`.
*   **Camera Selection Not Working:** Camera labels ("back", "front") are not standardized. Check the console logs on the sender page to see the actual device labels found and experiment with the dropdown options. 