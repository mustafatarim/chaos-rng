#!/usr/bin/env python3
"""Optional cryptography helper example."""

from chaos_rng import ThreeBodyRNG
from chaos_rng.security import CryptoPostProcessor, SecureSeed


def main() -> None:
    try:
        seed = SecureSeed().generate_seed()
        rng = ThreeBodyRNG(seed=seed)
        raw = rng.bytes(64)
        processed = CryptoPostProcessor(method="hash").process(raw)
    except Exception as exc:  # pragma: no cover - demo script UX
        print("crypto example unavailable:", exc)
        print("Install optional dependency: pip install 'chaos-rng[crypto]'")
        return

    print("seed length:", len(seed))
    print("raw length:", len(raw))
    print("processed length:", len(processed))

    print()
    print("These helpers are optional utilities, not a substitute for audited CSPRNGs.")


if __name__ == "__main__":
    main()
