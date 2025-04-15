import http.server
import ssl
import os

PORT = 8000
CERT_FILE = 'server.pem' # Make sure server.pem is in the same directory

# Check if certificate file exists
if not os.path.exists(CERT_FILE):
    print(f"Error: Certificate file '{CERT_FILE}' not found.")
    print("Please generate it using OpenSSL first:")
    print("openssl req -new -x509 -keyout server.pem -out server.pem -days 365 -nodes")
    exit(1)

# Create a standard request handler
handler = http.server.SimpleHTTPRequestHandler

# Wrap the socket with SSL
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
try:
    context.load_cert_chain(certfile=CERT_FILE)
except ssl.SSLError as e:
    print(f"Error loading certificate: {e}")
    print(f"Ensure '{CERT_FILE}' contains both a private key and a certificate.")
    exit(1)
except FileNotFoundError:
     print(f"Error: Certificate file '{CERT_FILE}' not found during context loading.")
     exit(1)


# Create the HTTPS server
with http.server.HTTPServer(("", PORT), handler) as httpd:
    print(f"Serving HTTPS on port {PORT}...")
    print(f"Open receiver on laptop: https://localhost:{PORT}/receiver.html")
    print(f"Open sender on mobile: https://<YOUR-LAPTOP-IP>:{PORT}/sender.html")
    print("*** You WILL see browser security warnings - you must accept/proceed. ***")
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.") 