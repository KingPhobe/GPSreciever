"""Run a minimal static GNSS twin demo."""

from gnss_twin.config import SimConfig


def main() -> None:
    config = SimConfig()
    print("Static demo configuration:")
    print(config)


if __name__ == "__main__":
    main()
