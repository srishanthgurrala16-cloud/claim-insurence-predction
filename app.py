import http.server
import socketserver
import webbrowser
import os

# Port configuration
PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class SafeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom handler to route request paths to templates/index.html and Static/style.css
    based on the project structure.
    """
    def do_GET(self):
        # Route default page requests to the templates folder
        if self.path == '/' or self.path == '/index.html':
            self.path = '/templates/index.html'
        # Route stylesheet requests to the Static folder
        elif self.path == '/style.css':
            self.path = '/Static/style.css'
            
        return super().do_GET()

def run_server():
    global PORT
    # Set the working directory to the directory of this file
    os.chdir(DIRECTORY)
    
    # Configure the TCP server to allow reusing the address
    socketserver.TCPServer.allow_reuse_address = True
    
    while PORT < 8100:
        try:
            with socketserver.TCPServer(("", PORT), SafeHTTPRequestHandler) as httpd:
                url = f"http://localhost:{PORT}/index.html"
                print("==================================================")
                print("  CreditFair Loan Default Risk Prediction Server  ")
                print("==================================================")
                print(f" Script Directory: {DIRECTORY}")
                print(f" Working Directory: {os.getcwd()}")
                print(f" Serving site locally at: {url}")
                print(" Press Ctrl+C to stop the server.")
                print("==================================================")
                
                # Automatically open the user's default browser window
                webbrowser.open(url)
                
                # Keep serving requests until interrupted
                httpd.serve_forever()
                break
        except OSError as e:
            print(f"Port {PORT} is busy, trying port {PORT + 1}...")
            PORT += 1
        except KeyboardInterrupt:
            print("\n[Server stopped successfully]")
            break

if __name__ == "__main__":
    run_server()
