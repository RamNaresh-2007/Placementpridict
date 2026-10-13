# PowerShell Script to Build and Push Placement Prediction App to Docker Hub
# Docker Hub Profile: https://app.docker.com/accounts/ram032007naresh

param (
    [string]$ImageName = "ram032007naresh/placement-predict",
    [string]$Tag = "latest"
)

$FullTag = "$ImageName`:$Tag"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Docker Build & Push: $FullTag" -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Check if Docker is installed and running
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Docker CLI not found in PATH." -ForegroundColor Red
    Write-Host "Please install Docker Desktop for Windows or run via WSL / GitHub Actions." -ForegroundColor Yellow
    exit 1
}

$daemonRunning = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker daemon is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# 2. Log in to Docker Hub
Write-Host "`n[1/3] Logging in to Docker Hub for user 'ram032007naresh'..." -ForegroundColor Green
docker login -u ram032007naresh

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker login failed." -ForegroundColor Red
    exit 1
}

# 3. Build Docker image
Write-Host "`n[2/3] Building Docker image '$FullTag'..." -ForegroundColor Green
docker build -t $FullTag .

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Docker build failed." -ForegroundColor Red
    exit 1
}

# 4. Push to Docker Hub
Write-Host "`n[3/3] Pushing image '$FullTag' to Docker Hub..." -ForegroundColor Green
docker push $FullTag

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==================================================" -ForegroundColor Green
    Write-Host " SUCCESS! Image pushed to Docker Hub:" -ForegroundColor Green
    Write-Host " https://hub.docker.com/r/$ImageName" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Green
    Write-Host "To run container locally:" -ForegroundColor Yellow
    Write-Host "  docker run -d -p 8000:8000 $FullTag" -ForegroundColor White
} else {
    Write-Host "[ERROR] Docker push failed." -ForegroundColor Red
    exit 1
}
