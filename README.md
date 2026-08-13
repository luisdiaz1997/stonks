# stonks

Simple, exploratory algorithms for stocks and investments.

A learning playground for experimenting with market-related ideas — backtests,
signals, portfolio heuristics, and anything else worth trying out. **No financial
advice**; just code to read, run, and tinker with.

## Installation

From source (editable, recommended while developing):

```bash
git clone https://github.com/luisdiaz1997/stonks.git
cd stonks
pip install -e ".[dev]"
```

Minimal install (runtime deps only):

```bash
pip install -e .
```

## Usage

```python
import stonks

print(stonks.__version__)
```

Or from the command line:

```bash
python -m stonks
```

## Development

Run tests:

```bash
pytest
```

Format and check types:

```bash
black .
isort .
mypy stonks
```

## License

[MIT](LICENSE)
