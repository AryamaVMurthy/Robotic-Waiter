console.log('script.js: File loading...'); // Log 1: Right at the start

document.addEventListener('DOMContentLoaded', () => {
    console.log('script.js: DOMContentLoaded event fired.'); // Log 2: Inside listener

    const myPeerIdElement = document.getElementById('myPeerId');
    // ... rest of the code ...

    function initializePeer() {
        console.log('script.js: Initializing PeerJS...'); // Log 3: Before new Peer()
        peer = new Peer(undefined, {
            // debug: 3
        });
        console.log('script.js: Peer object created (waiting for open event).'); // Log 4: After new Peer()

        peer.on('open', (id) => {
            console.log('script.js: PeerJS open event fired!'); // Log 5: Inside open event
            // ... rest of open event handler ...
        });
        // ... rest of initializePeer ...
    }

    // ... rest of the code ...

     // Initialize PeerJS connection when the page loads
     console.log('script.js: Calling initializePeer()...'); // Log 6: Before calling init
    initializePeer();
    console.log('script.js: initializePeer() called.'); // Log 7: After calling init

    // ... rest of the code ...
}); 