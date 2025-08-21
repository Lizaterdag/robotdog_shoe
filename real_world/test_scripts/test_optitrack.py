import natnet
from natnet.protocol.MocapFrameMessage import LabelledMarker, RigidBody
from natnet.comms import TimestampAndLatency
import logging
import time

# Set up logging for natnet library
logging.basicConfig(level=logging.INFO)
natnet_logger = logging.getLogger('natnet')
natnet_logger.setLevel(logging.DEBUG) # Set to DEBUG for maximum verbosity

# OptiTrack server IP address
server_ip = '192.168.1.132'
# local_ip = '192.168.1.108' # Your Ethernet IP
try:
    # Connect to the NatNet server
    client = natnet.Client.connect(server_ip, timeout=10)
    print("NatNet client connected successfully.")

    # Define a callback function to process received data
    def natnet_callback(rigid_bodies: list[RigidBody], markers: list[LabelledMarker], timing: TimestampAndLatency):
        if rigid_bodies:
            for rb in rigid_bodies:
                print(f"time: {timing.timestamp}")

    # Set the callback
    client.set_callback(natnet_callback)
    print("Callback set. Waiting for data...")
    client.spin()

    # Keep the client spinning to receive data
    # This will block until the client is stopped or an error occurs
    # We'll run it for a short period for testing
    start_time = time.time()
    while time.time() - start_time < 15: # Run for 15 seconds
        client.spin()
        # time.sleep(0.1) # Small delay to prevent busy-waiting
    
    print("Test complete. Stopping client.")
    client.stop()

except natnet.DiscoveryError as e:
    print(f"Error: Failed to connect to {server_ip}: {e}")
    print("Ensure Motive/OptiTrack software is running, NatNet streaming is enabled, and IP/firewall settings are correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Stopping client.")
    client.stop()
finally:
    print("NatNet client test finished.")