'''
How this works:
monitor_bins runs continuously, checking fill levels every 5 seconds.

If any bin crosses the threshold, it adds an alert.

Alerts are served via the /alerts API endpoint.

The homepage / shows live alerts refreshing every 5 seconds with JavaScript.

To run:
Save this as app.py.

Install Flask if not installed: pip install flask.

Run: python app.py.

Open http://127.0.0.1:5000/ in your browser to see alerts liv
'''


from flask import Flask, jsonify, render_template_string
import threading
import time

app = Flask(__name__)

# Sample bin data (in real case, this would come from sensors or a DB)
bins = {
    'bin_1': {'fill_level': 65, 'location': (12.97, 77.59)},
    'bin_2': {'fill_level': 82, 'location': (12.98, 77.60)},
    'bin_3': {'fill_level': 45, 'location': (12.96, 77.58)},
}

ALERT_THRESHOLD = 80
alerts = []

def monitor_bins():
    while True:
        alerts.clear()
        for bin_id, data in bins.items():
            if data['fill_level'] >= ALERT_THRESHOLD:
                alert_msg = f"Alert: {bin_id} is {data['fill_level']}% full!"
                alerts.append(alert_msg)
                print(alert_msg)  # Console alert or replace with logging
        time.sleep(5)  # Check every 5 seconds

@app.route('/alerts')
def get_alerts():
    return jsonify(alerts)

@app.route('/')
def home():
    # Simple web page displaying current alerts
    html = '''
    <!doctype html>
    <html>
      <head><title>Bin Alerts</title></head>
      <body>
        <h1>Garbage Bin Alerts</h1>
        <ul id="alert-list"></ul>
        <script>
          async function fetchAlerts() {
            const response = await fetch('/alerts');
            const data = await response.json();
            const list = document.getElementById('alert-list');
            list.innerHTML = '';
            data.forEach(alert => {
              const li = document.createElement('li');
              li.textContent = alert;
              list.appendChild(li);
            });
          }
          setInterval(fetchAlerts, 5000);
          fetchAlerts();
        </script>
      </body>
    </html>
    '''
    return render_template_string(html)

if __name__ == '__main__':
    # Run the bin monitoring in a background thread
    threading.Thread(target=monitor_bins, daemon=True).start()
    app.run(debug=True)
