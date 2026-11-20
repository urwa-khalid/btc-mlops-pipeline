# Screenshot checklist (save each as the file name shown, inside this folder)

| File name | What it must show | Report section |
| --- | --- | --- |
| s01_fetch_train.png | `python src/etl_fetch.py` and `python src/train_forecast.py` output with metrics | Part A / C |
| s02_pytest.png | `pytest -v` all tests passing | A3 |
| s03_api_curl.png | curl of /health, /predict, /metrics and one 400 error | A1 / A2 |
| s04_docker_build.png | `docker build` finishing, then `docker images btc-forecast-api` (size) and `docker run --rm btc-forecast-api id` (non-root) | B1 |
| s05_actions_green.png | GitHub Actions run with both jobs (test, build-and-push) green | B3 / B4 |
| s06_dockerhub_tags.png | Docker Hub tags page showing the commit SHA tag and `latest` | B4 |
| s07_compose_ps.png | `docker compose ps` with api, postgres, airflow running | B2 |
| s08_airflow_graph.png | Airflow Graph view of a manual run, all tasks green | C1 / E2 |
| s09_airflow_logs.png | train_and_evaluate log (metrics) and reload_api_model log (`reloaded`) | C1 / E2 |
| s10_kfp_compile.png | `python kubeflow/btc_kfp_pipeline.py` plus the top of the YAML | C2 |
| s11_k8s_resources.png | `kubectl get pods,svc,pvc,configmap,secret,ingress` | D1-D4, D6 |
| s12_k8s_describe.png | `kubectl describe deployment btc-forecast` (replicas, probes, resources) | D3 |
| s13_live_predict.png | curl through the Ingress (Host: btc-forecast.local) returning 7 predictions | D6 / E2 |
| s14_hpa_watch.png | `kubectl get hpa --watch` with replicas rising | D5 |
| s15_pods_scaled.png | `kubectl get pods` during load showing extra pods | D5 |
| s16_locust_summary.png | Locust end-of-run statistics table | D5 |
| s17_chaos.png | `kubectl delete pod` and the replacement pod starting | E1 |
