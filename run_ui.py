"""Run the Uvicorn server for the FLL Scheduler user interface."""

import threading
import time
import webbrowser

import uvicorn
from fll_scheduler_ga.api.main import app

HOST = "127.0.0.1"
PORT = 8000


def run_server() -> None:
    """Run the Uvicorn server."""
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    # We run the server in a separate thread so that the main thread can
    # continue and open the web browser.
    # The 'daemon=True' flag means the server thread will exit when the main
    # script is closed.
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # print(f"Server starting in background on http://{HOST}:{PORT}")

    # Give the server a moment to start up before opening the browser
    time.sleep(2)

    # print("Opening web browser to the user interface...")
    webbrowser.open(f"http://{HOST}:{PORT}")

    # print("\n----------------------------------------------------")
    # print("FLL Scheduler is now running.")
    # print("Close this window or press CTRL+C to stop the server.")
    # print("----------------------------------------------------\n")

    # Keep the main thread alive, waiting for a KeyboardInterrupt (Ctrl+C)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # print("Stopping server...")
        pass
