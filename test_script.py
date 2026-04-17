import uuid
import sys
from fastapi.testclient import TestClient

try:
    from app.main import app
except Exception as e:
    print(f"Error: app.main could not be imported: {e}")
    sys.exit(1)

def run_test():
    with TestClient(app) as client:
        # 1. GET /health
        health_res = client.get("/health")
        print(f"GET /health: {health_res.status_code}")

        # 2. POST /api/auth/register
        email = f"user_{uuid.uuid4().hex[:8]}@example.com"
        password = "Password123!"
        reg_data = {
            "email": email,
            "fullName": "Test User",
            "password": password,
            "confirmPassword": password
        }
        reg_res = client.post("/api/auth/register", json=reg_data)
        print(f"POST /api/auth/register: {reg_res.status_code}")
        reg_json = reg_res.json()
        token = reg_json.get("access_token")
        status = reg_json.get("status")
        print(f"Register status: {status}, Token present: {token is not None}")

        # 3. POST /api/auth/login
        login_res = client.post("/api/auth/login", json={"email": email, "password": password})
        print(f"POST /api/auth/login: {login_res.status_code}")
        
        # 4. GET /api/users/me
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        me_res = client.get("/api/users/me", headers=headers)
        print(f"GET /api/users/me: {me_res.status_code}")
        if me_res.status_code == 200:
            me_json = me_res.json()
            print(f"Me: {me_json.get('email')}, {me_json.get('fullName')}")
        else:
            print(f"Me error: {me_res.text}")

if __name__ == "__main__":
    if "." not in sys.path:
        sys.path.append(".")
    run_test()
