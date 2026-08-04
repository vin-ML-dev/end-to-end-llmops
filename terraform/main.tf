# Terraform — infrastructure as code. Here it provisions the Kubernetes
# namespace + the Secret/ConfigMap scaffolding declaratively, so the cluster's
# base resources are version-controlled and reproducible (not hand-created).
#
# In a real cloud setup this same file would also provision the cluster itself
# (EKS/GKE), node pools, and IAM. Kept minimal here: namespace + config objects.

terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.30"
    }
  }
}

provider "kubernetes" {
  config_path = "~/.kube/config"     # uses your current kubectl context
}

variable "namespace" {
  default = "domainbot"
}

variable "engine_url" {
  description = "External model endpoint (OpenAI-compatible /v1)"
  type        = string
  sensitive   = true
}

variable "engine_api_key" {
  description = "Token for the external endpoint"
  type        = string
  sensitive   = true
  default     = ""
}

variable "gateway_api_key" {
  description = "The gateway's own client-facing API key"
  type        = string
  sensitive   = true
}

resource "kubernetes_namespace" "domainbot" {
  metadata {
    name   = var.namespace
    labels = { app = "domainbot" }
  }
}

# secrets managed by Terraform (values come from tfvars / CI secrets, never committed)
resource "kubernetes_secret" "domainbot" {
  metadata {
    name      = "domainbot-secrets"
    namespace = kubernetes_namespace.domainbot.metadata[0].name
  }
  data = {
    DOMAINBOT_ENGINE_URL     = var.engine_url
    DOMAINBOT_ENGINE_API_KEY = var.engine_api_key
    DOMAINBOT_API_KEY        = var.gateway_api_key
  }
  type = "Opaque"
}

output "namespace" {
  value = kubernetes_namespace.domainbot.metadata[0].name
}
