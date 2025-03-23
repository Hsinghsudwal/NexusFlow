6. Model Monitoring (Prometheus & Grafana)
For model monitoring, we use Prometheus and Grafana to track performance over time. Here’s a simplified approach:

Flask Metrics: Use the prometheus_client library to expose metrics.
Prometheus: Scrapes the metrics endpoint.
Grafana: Visualizes the data.
Install prometheus_client:

pip install prometheus_client
Modify app.py to include Prometheus metrics:

from prometheus_client import start_http_server, Summary

# Create a metric to track prediction latency
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

@app.route('/predict', methods=['POST'])
@REQUEST_TIME.time()  # This will measure the time taken to serve the request
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
   
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    start_http_server(8000)  # Start Prometheus metrics server on port 8000
    app.run(debug=True)
Now, Prometheus will scrape metrics from http://<host>:8000/metrics, and Grafana can be used to visualize those metrics.