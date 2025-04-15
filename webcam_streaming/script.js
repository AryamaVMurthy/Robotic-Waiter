document.addEventListener('DOMContentLoaded', () => {
    const myPeerIdElement = document.getElementById('myPeerId');
    const statusElement = document.getElementById('status');
    const localVideo = document.getElementById('localVideo'); // Sender only
    const remoteVideo = document.getElementById('remoteVideo'); // Receiver only
    const receiverIdInput = document.getElementById('receiverIdInput'); // Sender only
    const startStreamButton = document.getElementById('startStreamButton'); // Sender only
    // New elements for camera selection (Sender only)
    const cameraSelectArea = document.querySelector('.camera-select-area');
    const cameraSelect = document.getElementById('cameraSelect');
    // New elements for receiver link
    const connectionInfoDiv = document.getElementById('connectionInfo'); // Receiver only
    const connectionLink = document.getElementById('connectionLink'); // Receiver only

    let peer = null;
    let localStream = null;
    let currentCall = null;
    let videoDevices = []; // To store available video devices

    function initializePeer() {
        // ... (PeerJS initialization code remains the same) ...
        peer = new Peer(undefined, {
            // debug: 3 // Uncomment for detailed PeerJS logs
        });

        peer.on('open', (id) => {
            console.log('script.js: PeerJS open event fired!'); // Log 5: Inside open event
            if (myPeerIdElement) {
                myPeerIdElement.textContent = id;
            }
            updateStatus('Peer connection open. Ready.');

            // --- Receiver Link Generation / QR Code --- 
            // Check if qrcode element and QRCode library exist (Receiver page)
            if (document.getElementById('qrcode') && typeof QRCode !== 'undefined') { 
                const qrcodeContainer = document.getElementById('qrcode');
                const connectionInfoDiv = document.getElementById('connectionInfo');

                // Construct the sender URL with the receiver's ID as a parameter
                const senderUrl = `${window.location.origin}/sender.html?receiverId=${id}`;
                console.log('Generating QR code for URL:', senderUrl);

                // Clear previous QR code if any
                qrcodeContainer.innerHTML = ''; 
                
                // Generate new QR code
                try {
                    new QRCode(qrcodeContainer, {
                        text: senderUrl,
                        width: 180, // Slightly smaller than container for padding
                        height: 180,
                        colorDark : "#000000",
                        colorLight : "#ffffff",
                        correctLevel : QRCode.CorrectLevel.H // High correction level
                    });
                    connectionInfoDiv.style.display = 'block'; // Show the QR code area
                    updateStatus('Peer ready. Scan the QR code with the sender device.');
                } catch (e) {
                    console.error("QR Code generation failed:", e);
                    updateStatus("Error generating QR code.");
                    connectionInfoDiv.style.display = 'none';
                }
                
            } else if (connectionLink) {
                 // Fallback for older browsers or if QR code library failed to load (uses link)
                 const senderUrl = `${window.location.origin}/sender.html?receiverId=${id}`;
                 connectionLink.href = senderUrl;
                 connectionLink.textContent = senderUrl;
                 connectionInfoDiv.style.display = 'block';
                 updateStatus('Peer ready. Share the link below with the sender.');
            }

            // --- Sender Actions (remain the same) ---
            if (startStreamButton) { 
                 const urlParams = new URLSearchParams(window.location.search);
                 const receiverIdFromUrl = urlParams.get('receiverId');
                 if (receiverIdFromUrl) {
                     console.log('Receiver ID found in URL:', receiverIdFromUrl);
                     receiverIdInput.value = receiverIdFromUrl;
                     updateStatus('Receiver ID loaded from link/scan. Ready to start camera.');
                     getCameras();
                 } else {
                    // Still might want to populate cameras even if no ID
                    // getCameras(); 
                 }
            }
        });

        peer.on('error', (err) => {
            console.error('PeerJS error:', err);
            updateStatus(`Error: ${err.message} (Type: ${err.type})`);
            // Attempt to reconnect on certain errors
            if (err.type === 'network' || err.type === 'server-error' || err.type === 'socket-error' || err.type === 'disconnected') {
                console.log('Attempting to reconnect PeerJS...');
                setTimeout(() => {
                    if (peer && !peer.destroyed && !peer.disconnected) {
                        peer.reconnect();
                    } else {
                        initializePeer(); // Re-initialize if destroyed or disconnected
                    }
                }, 3000); // Wait 3 seconds before reconnecting
            }
        });

        peer.on('disconnected', () => {
            updateStatus('Peer disconnected from signaling server. Attempting reconnect...');
            console.log('Peer disconnected. Attempting reconnect...');
            // PeerJS will automatically attempt to reconnect in this case
            // If it fails, the 'error' event might be triggered
        });

        peer.on('close', () => {
            updateStatus('Peer connection closed.');
            console.log('Peer connection closed.');
            // Maybe try to re-initialize after a delay
        });

        // --- Receiver Logic (remains the same) ---
        if (remoteVideo) { // We are on the receiver page
            peer.on('call', (call) => {
                console.log('Incoming call...');
                updateStatus('Incoming call...');

                // Answer the call without sending our own stream (we just want to receive)
                call.answer();

                currentCall = call;
                updateStatus('Call answered. Waiting for stream...');

                call.on('stream', (remoteStream) => {
                    console.log('Stream received');
                    updateStatus('Streaming...');
                    remoteVideo.srcObject = remoteStream;
                    // Add explicit play just in case autoplay doesn't fire for remote stream
                    remoteVideo.play().catch(e => {
                        console.error('Error attempting to play remote video:', e);
                        updateStatus(`Error playing stream: ${e.message}. Click video?`);
                    });
                });

                call.on('close', () => {
                    console.log('Call closed by sender.');
                    updateStatus('Call finished.');
                    if(remoteVideo) remoteVideo.srcObject = null; // Clear the video
                    currentCall = null;
                });

                call.on('error', (err) => {
                    console.error('Call error:', err);
                    updateStatus(`Call error: ${err.message}`);
                    if(remoteVideo) remoteVideo.srcObject = null;
                    currentCall = null;
                });
            });
        }
    }

    function updateStatus(message) {
        if (statusElement) {
            statusElement.textContent = message;
        }
        console.log(`Status: ${message}`);
    }

    // --- Sender Logic Modifications ---

    // Function to get and list available video devices
    async function getCameras() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
            updateStatus('Camera enumeration not supported on this browser.');
            console.error('enumerateDevices() not supported.');
            return false;
        }

        try {
            const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            videoDevices = await navigator.mediaDevices.enumerateDevices();
            videoDevices = videoDevices.filter(device => device.kind === 'videoinput');
            tempStream.getTracks().forEach(track => track.stop());

            if (videoDevices.length === 0) {
                updateStatus('No video cameras found.');
                return false;
            }

            // Log available devices for debugging
            console.log('Available video devices:', JSON.stringify(videoDevices.map(d => ({ id: d.deviceId, label: d.label, kind: d.kind })), null, 2));

            cameraSelect.innerHTML = ''; // Clear previous options
            let foundBackCameraId = null;

            videoDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Camera ${index + 1}`;

                if (device.label.toLowerCase().includes('back')) {
                    option.text += ' (Back)';
                    if (!foundBackCameraId) { // Prioritize the first one labeled 'back'
                        foundBackCameraId = device.deviceId;
                    }
                } else if (device.label.toLowerCase().includes('front')) {
                    option.text += ' (Front)';
                } else if (device.label.toLowerCase().includes('wide') || device.label.toLowerCase().includes('ultra')) {
                    option.text += ' (Wide)';
                    // Optionally prioritize back wide camera if desired
                    // if (!foundBackCameraId && (device.label.toLowerCase().includes('back'))) {
                    //     foundBackCameraId = device.deviceId;
                    // }
                }
                cameraSelect.appendChild(option);
            });

            // Default to the back camera if one was identified by label
            if (foundBackCameraId) {
                console.log('Defaulting to back camera:', foundBackCameraId);
                cameraSelect.value = foundBackCameraId;
            } else {
                 console.log('No back camera identified by label, defaulting to first camera in list.');
                 // The first option added will be selected by default otherwise
            }

            cameraSelectArea.style.display = 'block'; // Show the dropdown
            return true; // Indicate success

        } catch (err) {
            updateStatus(`Error accessing media devices: ${err.message}. Please grant permission.`);
            console.error('Error getting media devices:', err);
             cameraSelectArea.style.display = 'none'; // Hide dropdown on error
            return false;
        }
    }

    // Function to start the stream with a specific camera
    async function startStream(deviceId) {
        stopStream(); // Stop any existing stream/call first

        const receiverId = receiverIdInput.value.trim();
        if (!receiverId) {
            updateStatus('Please enter the Receiver Peer ID.');
            return;
        }
        if (!peer || peer.disconnected) {
            updateStatus('Peer connection not ready. Please wait or refresh.');
            return;
        }
        if (currentCall) {
            console.warn('Attempted to start stream while call active. Should have been stopped.');
            // stopStream should handle this, but as a safeguard:
            currentCall.close();
            currentCall = null;
        }

        updateStatus('Attempting to start camera...');
        // Log the exact deviceId being requested
        console.log('Requesting getUserMedia with deviceId:', deviceId);

        const constraints = {
            video: { deviceId: deviceId ? { exact: deviceId } : undefined },
            audio: false // Keep audio off for now
        };

        try {
            localStream = await navigator.mediaDevices.getUserMedia(constraints);
            console.log('Camera access granted for selected device.');
            localVideo.srcObject = localStream;
            localVideo.play().catch(e => console.error('Local video play failed:', e));

            updateStatus(`Calling Peer: ${receiverId}...`);
            console.log(`Calling ${receiverId}`);

            const call = peer.call(receiverId, localStream);
            currentCall = call;

            call.on('close', () => {
                console.log('Call closed (sender side).');
                 // Don't stop stream here automatically, user might want to switch camera
                updateStatus('Call finished or closed.');
                 // We nullify currentCall but keep localStream running for preview
                 currentCall = null; 
            });

            call.on('error', (err) => {
                console.error('Call error:', err);
                updateStatus(`Call error: ${err.message}. Check Receiver ID and network.`);
                stopStream(); // Stop stream fully on call error
                currentCall = null;
            });

        } catch (err) {
            // Log the FULL error object for more details
            console.error('Failed to get local stream for selected device. Full Error:', err);
            updateStatus(`Error accessing camera: ${err.name} - ${err.message}. Please grant permission or try another camera.`);
            stopStream(); // Stop if getUserMedia fails
        }
    }

    // Modified Button Listener
    if (startStreamButton) { // We are on the sender page
        startStreamButton.addEventListener('click', async () => {
            // Only populate cameras if not already done or if failed previously
            if (videoDevices.length === 0) {
                 const success = await getCameras();
                 if (!success) return; // Stop if getting cameras failed
            }
            // Start stream with the currently selected camera in the dropdown
            startStream(cameraSelect.value);
        });

        // Add listener for camera selection changes
        cameraSelect.addEventListener('change', () => {
            console.log('Camera selection changed to:', cameraSelect.value);
            // Start a new stream with the newly selected camera
            // This will implicitly stop the old stream via startStream()
            startStream(cameraSelect.value);
        });
    }

    // Modified Stop Function
    function stopStream() {
        if (localStream) {
            console.log('Stopping local stream tracks.');
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
            if (localVideo) {
                localVideo.srcObject = null;
            }
        }
        if (currentCall) {
            console.log('Closing active call.');
            // Don't necessarily set status here, might be called during camera switch
            currentCall.close();
            currentCall = null;
        }
        // Keep camera select visible if it was populated
        // updateStatus('Stream stopped.'); // Avoid overwriting specific statuses
    }

    // Initialize PeerJS connection when the page loads
    initializePeer();

    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
        stopStream();
        if (peer && !peer.destroyed) {
            peer.destroy();
        }
    });
}); 