# Required Git workflow

The assignment requires at least one feature branch that is committed at least twice and merged back into main.

Example:

```bash
git init
git add .
git commit -m "Initial Zepto platform"

git checkout -b feature/data-pipeline
git add data_pipeline/
git commit -m "Add data pipeline"
git add analytics/ support_assistant/
git commit -m "Add analytics and support assistant"

git checkout main
git merge --no-ff feature/data-pipeline -m "Merge feature/data-pipeline"

git log --graph --all --oneline
```

Push the final repository to GitHub as a public repository.
