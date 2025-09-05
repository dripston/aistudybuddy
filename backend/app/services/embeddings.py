import httpx
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

# Config
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://api.sambanova.ai/v1")
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "E5-Mistral-7B-Instruct")
EMBED_DIM = int(os.getenv("EMBED_DIM", "4096"))


async def get_embedding(text: str):
    """
    Calls Sambanova embeddings API and returns the embedding vector.
    Handles empty input and invalid responses gracefully.
    """
    if not text or not text.strip():
        print("⚠️ Skipping empty input text for embeddings.")
        return None

    # Debug API credentials (partial for security)
    print(f"\n=== DEBUG: API Configuration ===")
    print(f"API Key: {EMBED_API_KEY[:10]}...")  # Show first 10 chars only
    print(f"Model: {EMBED_MODEL}")
    print(f"Base URL: {EMBED_BASE_URL}")

    url = f"{EMBED_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {EMBED_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": EMBED_MODEL, "input": text}
    
    # Debug request details
    print(f"\n=== DEBUG: Request Details ===")
    print(f"URL: {url}")
    print(f"Text length: {len(text)}")
    print(f"Text preview: {text[:100]}...")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, headers=headers, json=payload)

            # Debug response details
            print(f"\n=== DEBUG: Response Details ===")
            print(f"Status code: {resp.status_code}")
            print(f"Response headers: {dict(resp.headers)}")
            print(f"Response text: {resp.text[:500]}")  # First 500 chars

            try:
                data = resp.json()
            except Exception as e:
                print(f"❌ JSON parsing error: {e}")
                print(f"Full response: {resp.text[:1000]}...")
                return None

            # Check for API error messages
            if "error" in data:
                print(f"❌ API Error: {data['error']}")
                if "message" in data.get("error", {}):
                    print(f"Error message: {data['error']['message']}")
                if "code" in data.get("error", {}):
                    print(f"Error code: {data['error']['code']}")
                return None

            # Check for rate limiting headers
            rate_limit_remaining = resp.headers.get("x-ratelimit-remaining")
            rate_limit_reset = resp.headers.get("x-ratelimit-reset")
            if rate_limit_remaining:
                print(f"Rate limit remaining: {rate_limit_remaining}")
            if rate_limit_reset:
                print(f"Rate limit reset: {rate_limit_reset}")

            # Debug info
            print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'None'}")

            if not data or "data" not in data:
                print(f"⚠️ Invalid response (no 'data'): {data}")
                return None

            if len(data["data"]) == 0:
                print(f"⚠️ Empty 'data' array: {data}")
                return None

            item = data["data"][0]
            embedding = item.get("embedding") or item.get("vector") or item.get("values")

            if embedding is None:
                print(f"❌ No embedding field found in response: {item}")
                return None
            try:
                embedding=[float(value) for value in embedding]
            except (TypeError, ValueError) as e:
                print(f"❌ Invalid embedding format: {e}")
                return None
            if len(embedding) != EMBED_DIM:
                print(f"⚠️ Dimension mismatch: expected {EMBED_DIM}, got {len(embedding)}")

            return embedding

    except httpx.TimeoutException:
        print("❌ Request timeout")
        return None
    except httpx.RequestError as e:
        print(f"❌ Request error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None