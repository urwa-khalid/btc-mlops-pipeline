## C2: Kubeflow Pipeline

The Kubeflow pipeline performs three main steps: ingesting Bitcoin data, generating features, and training the forecasting model. Each step uses the output of the previous step as its input. We successfully compiled the pipeline into a YAML file but did not run it because execution requires a configured Kubeflow environment and cluster.

## E1: Kubernetes Self-Healing

We deployed the BTC forecasting application using a Kubernetes Deployment configured with three replicas. Initially, all three pods were running with a READY status of 1/1. We manually deleted the pod btc-forecast-c975d96f8-6d989 to test self-healing. Kubernetes automatically created a replacement pod named btc-forecast-c975d96f8-vktr7, maintaining the required three replicas.
