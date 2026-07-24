import urllib.request, hashlib, os, sys

url = "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
dest = "/root/.cache/whisper/large-v3.pt"
expected = "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb"

os.makedirs(os.path.dirname(dest), exist_ok=True)

for attempt in range(3):
    print(f"Download attempt {attempt+1}/3...")
    urllib.request.urlretrieve(url, dest)
    sha = hashlib.sha256(open(dest, "rb").read()).hexdigest()
    if sha == expected:
        print("Checksum OK")
        sys.exit(0)
    print(f"Checksum mismatch: {sha}")
    os.remove(dest)

print("All 3 attempts failed")
sys.exit(1)