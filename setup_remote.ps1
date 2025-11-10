# Setup git remote and push
# Usage: .\setup_remote.ps1 <repository-url>
# Example: .\setup_remote.ps1 https://github.com/username/repo-name.git

param(
    [Parameter(Mandatory=$true)]
    [string]$RepositoryUrl
)

Write-Host "Adding remote repository: $RepositoryUrl" -ForegroundColor Yellow
git remote add origin $RepositoryUrl

Write-Host "Pushing to remote repository..." -ForegroundColor Yellow
git branch -M main
git push -u origin main

Write-Host "✓ Successfully pushed to remote repository!" -ForegroundColor Green

