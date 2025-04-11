
import asyncio
import aiocoap
import json
import time
import tkinter as tk
from functools import partial
import threading

# Change this to your ESP32's IP address
ESP_IP = "192.168.137.120"

class RobotControlGUI:
    def __init__(self, root, coap_loop):
        self.root = root
        self.coap_loop = coap_loop
        self.protocol = None
        self.sequence = 1
        self.last_command_time = 0
        self.command_cooldown = 0.2  # Seconds between commands to avoid flooding
        
        # Speed control variables
        self.forward_speed = 255  # Default forward speed (0-255)
        self.turn_speed = 255      # Default turn speed (0-255)
        
        # Set up the GUI
        self.root.title("Robot Controller")
        self.root.geometry("500x450")  # Increased height for sliders
        self.root.configure(bg='#f0f0f0')
        
        # Setup the control frame
        control_frame = tk.Frame(root, bg='#f0f0f0')
        control_frame.pack(pady=20)
        
        # Command status display
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = tk.Label(root, textvariable=self.status_var, bg='#f0f0f0', font=('Arial', 12))
        status_label.pack(pady=10)
        
        # Create directional control buttons
        button_config = {
            'width': 10,
            'height': 2,
            'font': ('Arial', 12, 'bold'),
        }
        
        # Forward button (top)
        self.forward_btn = tk.Button(control_frame, text="Forward", bg='#92d36e', **button_config,
                                     command=lambda: self.send_command("MOVE_FORWARD"))
        self.forward_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Left button (middle left)
        self.left_btn = tk.Button(control_frame, text="Left", bg='#6eb3d3', **button_config,
                                 command=lambda: self.send_command("TURN_LEFT"))
        self.left_btn.grid(row=1, column=0, padx=5, pady=5)
        
        # Stop button (middle)
        self.stop_btn = tk.Button(control_frame, text="Stop", bg='#d36e6e', **button_config,
                                 command=lambda: self.send_command("STOP"))
        self.stop_btn.grid(row=1, column=1, padx=5, pady=5)
        
        # Right button (middle right)
        self.right_btn = tk.Button(control_frame, text="Right", bg='#6eb3d3', **button_config,
                                  command=lambda: self.send_command("TURN_RIGHT"))
        self.right_btn.grid(row=1, column=2, padx=5, pady=5)
        
        # Add speed control frame
        speed_frame = tk.Frame(root, bg='#f0f0f0', bd=2, relief=tk.GROOVE)
        speed_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Forward speed slider
        tk.Label(speed_frame, text="Forward Speed:", bg='#f0f0f0').grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.forward_speed_slider = tk.Scale(
            speed_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
            length=200, bg='#f0f0f0', command=self.update_forward_speed
        )
        self.forward_speed_slider.set(self.forward_speed)
        self.forward_speed_slider.grid(row=0, column=1, padx=10, pady=5)
        self.forward_speed_label = tk.Label(speed_frame, text=f"{self.forward_speed}", width=3, bg='#f0f0f0')
        self.forward_speed_label.grid(row=0, column=2, padx=10, pady=5)
        
        # Turn speed slider
        tk.Label(speed_frame, text="Turn Speed:", bg='#f0f0f0').grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        self.turn_speed_slider = tk.Scale(
            speed_frame, from_=0, to=255, orient=tk.HORIZONTAL,
            length=200, bg='#f0f0f0', command=self.update_turn_speed
        )
        self.turn_speed_slider.set(self.turn_speed)
        self.turn_speed_slider.grid(row=1, column=1, padx=10, pady=5)
        self.turn_speed_label = tk.Label(speed_frame, text=f"{self.turn_speed}", width=3, bg='#f0f0f0')
        self.turn_speed_label.grid(row=1, column=2, padx=10, pady=5)
        
        # Instructions
        instructions = """
        Control with:
        - Arrow Up: Forward
        - Arrow Left: Turn Left
        - Arrow Right: Turn Right
        - Space/Down: Stop
        
        Use the sliders to adjust speed:
        - Forward Speed: Speed for forward motion (0-255)
        - Turn Speed: Speed for turning left/right (0-255)
        """
        instruction_label = tk.Label(root, text=instructions, bg='#f0f0f0', justify=tk.LEFT)
        instruction_label.pack(pady=10)
        
        # Key bindings
        self.root.bind("<Up>", lambda event: self.send_command("MOVE_FORWARD"))
        self.root.bind("<Left>", lambda event: self.send_command("TURN_LEFT"))
        self.root.bind("<Right>", lambda event: self.send_command("TURN_RIGHT"))
        self.root.bind("<Down>", lambda event: self.send_command("STOP"))
        self.root.bind("<space>", lambda event: self.send_command("STOP"))
        
        # Initialize CoAP client
        self.init_coap()
    
    def update_forward_speed(self, value):
        self.forward_speed = int(value)
        self.forward_speed_label.config(text=f"{self.forward_speed}")
    
    def update_turn_speed(self, value):
        self.turn_speed = int(value)
        self.turn_speed_label.config(text=f"{self.turn_speed}")
    
    def init_coap(self):
        # Schedule the CoAP context creation in the asyncio loop
        asyncio.run_coroutine_threadsafe(self.create_protocol(), self.coap_loop)
    
    async def create_protocol(self):
        self.protocol = await aiocoap.Context.create_client_context()
        self.status_var.set("Connected to CoAP")
    
    def send_command(self, cmd):
        # Check if we're sending commands too quickly
        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return
            
        self.last_command_time = current_time
        
        # Highlight the pressed button
        self.highlight_button(cmd)
        
        # Update status
        self.status_var.set(f"Sending: {cmd}")
        
        # Schedule the command in the asyncio loop
        asyncio.run_coroutine_threadsafe(self.async_send_command(cmd), self.coap_loop)
    
    def highlight_button(self, cmd):
        # Reset all buttons
        self.forward_btn.config(relief=tk.RAISED)
        self.left_btn.config(relief=tk.RAISED)
        self.right_btn.config(relief=tk.RAISED)
        self.stop_btn.config(relief=tk.RAISED)
        
        # Highlight the active button
        if cmd == "MOVE_FORWARD":
            self.forward_btn.config(relief=tk.SUNKEN)
        elif cmd == "TURN_LEFT":
            self.left_btn.config(relief=tk.SUNKEN)
        elif cmd == "TURN_RIGHT":
            self.right_btn.config(relief=tk.SUNKEN)
        elif cmd == "STOP":
            self.stop_btn.config(relief=tk.SUNKEN)
        
        # Schedule to reset the button after a short delay
        self.root.after(200, self.reset_buttons)
    
    def reset_buttons(self):
        self.forward_btn.config(relief=tk.RAISED)
        self.left_btn.config(relief=tk.RAISED)
        self.right_btn.config(relief=tk.RAISED)
        self.stop_btn.config(relief=tk.RAISED)
    
    async def async_send_command(self, cmd):
        if not self.protocol:
            self.status_var.set("Error: CoAP not initialized")
            return
            
        # Construct the URI
        uri = f"coap://{ESP_IP}/command"
        
        # Create the payload with speed parameters
        payload_dict = {
            "cmd": cmd,
            "seq": self.sequence,
            "speed": self.forward_speed,
            "turn_speed": self.turn_speed
        }
        payload = json.dumps(payload_dict).encode('utf-8')
        self.sequence += 1
        
        # Build the CoAP request message
        request = aiocoap.Message(
            code=aiocoap.PUT,
            uri=uri,
            payload=payload
        )
        
        try:
            response = await self.protocol.request(request).response
            response_text = f"Response: {response.payload.decode('utf-8')}"
            print(f"Command: {cmd}, Speed: {self.forward_speed}/{self.turn_speed}, {response_text}")
            self.root.after(0, self.status_var.set, f"Sent: {cmd}")
        except Exception as e:
            error_msg = f"Error sending {cmd}: {str(e)}"
            print(error_msg)
            self.root.after(0, self.status_var.set, error_msg)

def run_asyncio_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

def main():
    # Create the asyncio event loop
    asyncio_loop = asyncio.new_event_loop()
    
    # Create and start the asyncio thread
    threading.Thread(target=run_asyncio_loop, args=(asyncio_loop,), daemon=True).start()
    
    # Create the Tkinter window
    root = tk.Tk()
    app = RobotControlGUI(root, asyncio_loop)
    
    try:
        # Start the Tkinter main loop
        root.mainloop()
    finally:
        # Cleanup
        asyncio_loop.call_soon_threadsafe(asyncio_loop.stop)

if __name__ == '__main__':
    main()
