# Terraform — declarative cluster base

Provisions the namespace + Secret so the cluster's base state is version-controlled
and reproducible, instead of hand-run `kubectl create` commands.

```bash
cp terraform.tfvars.example terraform.tfvars   # fill in real values (gitignored)
terraform init
terraform plan                                  # preview
terraform apply                                 # create namespace + secret
```

Why alongside Argo CD? Terraform owns **infra** (cluster, namespace, secrets, cloud
resources); Argo CD owns **app deployment** (the manifests in `k8s/`). Common split:
Terraform lays the foundation once, Argo continuously deploys the app on top.
