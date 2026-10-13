# 🐳 Docker Deployment Guide — Placement Prediction Suite

Docker Hub Account: [ram032007naresh](https://hub.docker.com/u/ram032007naresh)  
Docker Image: `ram032007naresh/placement-predict:latest`

---

## 🚀 Quick Start: Pull and Run

Once pushed to Docker Hub, anyone can run the application anywhere with a single command:

```bash
docker run -d -p 8000:8000 --name placement-app ram032007naresh/placement-predict:latest
```

Open your browser at: **[http://localhost:8000](http://localhost:8000)**

To run the Streamlit Analytics Dashboard instead:
```bash
docker run -d -p 8501:8501 --name placement-dashboard ram032007naresh/placement-predict:latest streamlit run placement/Data/Src/05_Web_Application/dashboard.py --server.port=8501 --server.address=0.0.0.0
```
Open your browser at: **[http://localhost:8501](http://localhost:8501)**

---

## 🛠️ Option 1: Automated Push via GitHub Actions (Recommended)

Since the code is hosted on GitHub (`RamNaresh-2007/Placementpridict`), GitHub Actions can build and push the image automatically using cloud runners without needing local Docker.

1. Go to your GitHub repository:
   **GitHub Repository Settings ➔ Secrets and variables ➔ Actions**
2. Click **New repository secret**:
   - Name: `DOCKERHUB_TOKEN`
   - Value: Your Docker Hub Personal Access Token (generate at [Docker Hub Account Settings ➔ Security](https://app.docker.com/settings/personal-access-tokens))
3. Push your commits to GitHub:
   ```bash
   git add .
   git commit -m "Add Docker containerization and CI/CD workflow"
   git push origin master
   ```
4. GitHub Actions will automatically:
   - Build the Docker image
   - Tag it as `ram032007naresh/placement-predict:latest`
   - Push it directly to your Docker Hub repository!

---

## 💻 Option 2: Local Build & Push (PowerShell / Windows)

If you have Docker Desktop installed on your Windows machine:

1. **Log in to Docker Hub:**
   ```powershell
   docker login -u ram032007naresh
   ```
   *(Enter your Docker Hub password or Personal Access Token)*

2. **Run the automated script:**
   ```powershell
   .\push_docker.ps1
   ```

   *Or manually execute:*
   ```powershell
   docker build -t ram032007naresh/placement-predict:latest .
   docker push ram032007naresh/placement-predict:latest
   ```

---

## 🐧 Option 3: Local Build & Push (WSL / Linux)

If using WSL Ubuntu or Linux:

```bash
chmod +x push_docker.sh
./push_docker.sh
```

---

## 📦 Using Docker Compose (Local Testing)

Run both the Flask Web App and Streamlit Dashboard concurrently:

```bash
docker compose up -d
```

- Flask Web App: `http://localhost:8000`
- Streamlit Analytics: `http://localhost:8501`

To stop:
```bash
docker compose down
```
