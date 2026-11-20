"""Load test: sustained /predict traffic to trigger the HPA.

    locust -f load/locustfile.py --headless -u 50 -r 10 -t 4m \
        --host http://localhost --csv load/results
Send the Host header via the Ingress: add `--host http://btc-forecast.local` after
adding `127.0.0.1 btc-forecast.local` to /etc/hosts.
"""

from locust import HttpUser, between, task


class ForecastUser(HttpUser):
    wait_time = between(0.0, 0.2)

    @task(8)
    def predict(self):
        self.client.post("/predict", json={"forecast_days": 7})

    @task(1)
    def health(self):
        self.client.get("/health")

    @task(1)
    def metrics(self):
        self.client.get("/metrics")
