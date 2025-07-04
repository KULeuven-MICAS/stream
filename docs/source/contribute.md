# Contribute

## Contributing Guidelines

When contributing to the framework, please follow these guidelines:

- Use Google's [Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Use Google-style docstrings for classes, functions, and methods.  
  See examples [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Update the documentation when you change or add public functionality

---

## Quick Setup

1. **Clone & create a branch**

   ```bash
   git clone <your-fork-url>
   cd stream
   git checkout -b <feature-or-fix>
   ```

2. **Install the dev tools (one-time only)**

   ```bash
   # Python ≥ 3.11
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip ruff pre-commit pytest
   pre-commit install  # hooks run on every commit
   ```

3. **Use VS Code with Ruff**

   - Install the **Ruff** extension (`charliermarsh.ruff`)
   - Enable **Format on Save**
   - Disable other formatters/linters to avoid conflicts

4. **Run the full check suite**

   ```bash
   ruff check .       # lint + auto-fix suggestions
   ruff format .      # apply formatting
   pytest             # run tests
   ```

---

## Coding Style

- **Style guide** – [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- **Line length** – 120 characters (enforced by Ruff)
- **Docstrings** – Google-style  
  (see [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html))
- **Type hints** – Required for all public functions, classes, and methods

---

## Submitting a Pull Request

1. Ensure all checks pass:

   ```bash
   ruff check .
   ruff format --check .
   pytest
   ```

2. Add or update unit tests  
3. Update documentation if public APIs change  
4. Open a pull request and fill out the PR template

---

Thanks for contributing to **Stream**!
