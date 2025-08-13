# -*- coding: utf-8 -*-
"""
Entry point script for the trading bot.

This script simply delegates to the command line interface defined in
``trading_bot.cli``. It exists as a convenient top‑level executable so that
end‑users can run the bot with ``python app.py`` rather than invoking the
package directly via ``python -m trading_bot.cli``.
"""

from trading_bot.cli import main


if __name__ == "__main__":
    main()