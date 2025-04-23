# build-docker.ps1

param (
    [string]$ImageName = "mle-project-challenge",
    [string]$Tag = "latest",
    [string]$DockerfilePath = "."
)

Write-Host "Building Docker image: ${ImageName}:${Tag} from ${DockerfilePath}"

docker build -t "${ImageName}:${Tag}" ${DockerfilePath}

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful: ${ImageName}:${Tag}"
} else {
    Write-Error "Build failed"
    exit 1
}