"""Entry point for `python -m stonks`."""

from . import __version__


def main() -> None:
    print(f"stonks {__version__}")
    print("Simple, exploratory algorithms for stocks and investments.")


if __name__ == "__main__":
    main()
