# Kubernetes manifests — gateway-only (external model endpoint)

**Architecture:** the model runs on an **external cloud platform**; we have only its
API endpoint. Kubernetes runs **only the FastAPI gateway**, which calls out to that
endpoint and returns responses to users. There is no engine in this cluster.

```
[user] → [K8s: FastAPI gateway] ──HTTP──▶ [external cloud model endpoint /v1]
              probes · auth · limits            (managed vLLM — we just call it)
```

Deploy everything:
```bash
kubectl apply -k k8s/
```

| File | What it is | Why it matters |
|---|---|---|
| `00-namespace` | isolates resources | blast-radius containment |
| `10-configmap` | non-secret knobs (revision, timeout) | tune without rebuild |
| `15-secret` | **external endpoint URL** + its token + gateway key (template) | endpoint/secrets never in Git |
| `20-gateway-deployment` | the gateway pods + **probes** | liveness restarts, readiness gates traffic |
| `25-gateway-service` | stable name + load-balance | one address for clients |
| `40-hpa` | autoscale the gateway on CPU | handle load spikes |
| `45-pdb` | keep ≥1 pod during disruptions | no full outage on node drain |

**Probes here:**
- `livenessProbe → /healthz` — process alive? fail = **restart**
- `readinessProbe → /readyz` — can we reach the **external endpoint**? fail = **stop routing traffic**

So if the external model endpoint goes down, the gateway pods go *NotReady* (traffic held)
but are *not* restarted — correct behavior for an upstream dependency being unavailable.

**Rollback:**
```bash
kubectl -n domainbot rollout undo deployment/domainbot-gateway     # roll back gateway code
# switch which model version you report / request:
kubectl -n domainbot patch configmap domainbot-config --type merge \
  -p '{"data":{"DOMAINBOT_REVISION":"v1.0.0"}}'
kubectl -n domainbot rollout restart deployment/domainbot-gateway
```
