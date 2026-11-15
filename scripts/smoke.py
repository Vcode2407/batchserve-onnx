"""Send one prediction to a running BatchServe instance using the standard library."""

import argparse
import json
import math
import struct
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    with urlopen(base + "/readyz", timeout=3) as response:
        if json.load(response) != {"status": "ready"}:
            raise RuntimeError("Server is not ready")
    features = [((index % 7) - 3) / 3 for index in range(128)]
    request = Request(base + "/v1/predict", data=struct.pack("<128f", *features),
                      headers={"Content-Type": "application/octet-stream", "X-Timeout-Ms": "3000"})
    with urlopen(request, timeout=5) as response:
        output = struct.unpack("<32f", response.read())
        if response.headers.get_content_type() != "application/octet-stream" or not all(map(math.isfinite, output)):
            raise RuntimeError("Unexpected prediction response")
    with urlopen(base + "/metrics", timeout=3) as response:
        if b"batchserve_batch_size_count " not in response.read():
            raise RuntimeError("Runtime metrics are missing")
    print(json.dumps({"outputs": len(output), "preview": output[:4]}))


if __name__ == "__main__":
    main()
