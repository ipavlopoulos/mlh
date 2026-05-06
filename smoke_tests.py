from app import app


def assert_ok(path):
    with app.test_client() as client:
        response = client.get(path)
    assert response.status_code == 200, f"{path} returned {response.status_code}"


def main():
    for path in ["/", "/alzheimer", "/llm", "/health"]:
        assert_ok(path)
    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
