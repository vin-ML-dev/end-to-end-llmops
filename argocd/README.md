# GitOps with Argo CD

**The model:** Git is the single source of truth. Argo CD runs *in* the cluster,
watches this repo, and continuously makes the cluster match `k8s/`. You never
`kubectl apply` by hand.

```
push to main → CI: test → build+push image (git-<sha>) → bump kustomization.yaml
                                                                    ↓ (git commit)
                                            Argo CD sees the change → syncs cluster
```

- **Deploy** = merge to main. CI builds the image and writes the new tag into
  `k8s/kustomization.yaml`; Argo rolls it out.
- **Rollback** = `git revert` the bump commit. Argo syncs back to the old image.
  The cluster's history *is* the Git history — fully auditable.
- **selfHeal** = if someone hand-edits the cluster, Argo reverts it to match Git.
- **prune** = if you delete a manifest from Git, Argo deletes it from the cluster.

Install Argo CD + this app: see `COMMANDS-day6.txt` Block 4.
