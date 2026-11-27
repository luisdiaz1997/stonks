# Running the take-profit sell job on a server

`stonks sell` checks every open Robinhood position and sells each one whose
total return has reached a threshold. This is how to run it unattended on a
Linux server with systemd: a timer every 15 minutes through the US session, a
dry run first, live selling as a separate, explicit step, and an optional
weekly cycle (step 7) that sells everything and buys it back.

Tested on Debian 12 with Python 3.11. While it runs, the job needs about 85 MB
of memory and a few seconds, and it does not need `torch`. It makes outbound
HTTPS calls to `api.robinhood.com` and listens on no port, so nginx and firewall
rules are not involved. The unit files assume the code lives in
`/root/gitclones/stonks`; if you clone somewhere else, change that path in them.

**Nothing here sells until step 6, and step 6 is one deliberate command.**

## 1. Install the code

```bash
git clone https://github.com/luisdiaz1997/stonks.git /root/gitclones/stonks
cd /root/gitclones/stonks
git checkout master                     # any branch or tag that has `stonks sell`
python3 -m venv .venv
.venv/bin/pip install --no-deps -e .
.venv/bin/pip install pandas robin_stocks click tqdm
.venv/bin/stonks sell --help
```

`--no-deps` is deliberate. The package declares `torch` and `yfinance`, which
the sell job never imports, and `torch` alone is a very large install. The
second command adds only what the job needs.

## 2. Add the credentials

The job signs in with your Robinhood username and password, kept in a file only
root can read. Together with the saved session from step 3, this file is enough
to trade the account. Never commit it or paste it into chat.

```bash
install -d -m 700 /etc/stonks
(umask 077; cat > /etc/stonks/robinhood.env <<'EOF'
ROBINHOOD_USERNAME='you@example.com'
ROBINHOOD_PASSWORD='your password'
EOF
)
```

Keep the single quotes. With them, spaces and characters such as `#` or `$`
mean the same thing to systemd as to a shell; without them a value with a space
breaks step 3.

## 3. Sign in once by hand

Robinhood approves each new device through its app, so the first sign-in needs
you. Approve the prompt when it arrives:

```bash
set -a; . /etc/stonks/robinhood.env; set +a
/root/gitclones/stonks/.venv/bin/stonks sell
```

This saves a session to `~/.tokens/robinhood.pickle`, which the timer reuses.
The command ends with a dry run: a table of every position, its total return,
and what the rule would do to it.

## 4. Check the numbers

Compare the table with the Robinhood app before going further. `total_return_pct`
is the app's total return, `day_change_pct` its today's return, and
`avg_buy_price` its average cost.

## 5. Install the timer (still a dry run)

```bash
cd /root/gitclones/stonks
install -m 644 deploy/systemd/stonks-sell.service deploy/systemd/stonks-sell.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now stonks-sell.timer
systemctl list-timers | grep stonks
```

The timer starts the job every 15 minutes through the US session: 06:32 to
12:47 Pacific, 26 runs, on every day including weekends. On a closed day the
job asks Robinhood's calendar and does nothing. The first run is at 06:32 rather
than 06:30:00 because quotes can still show the previous close in the opening
seconds. The timer is enabled, so it comes back after a reboot. A run missed
while the server was off is skipped, not replayed.

Each run writes its table to the journal. Start one now to confirm the whole
setup works under systemd (still a dry run):

```bash
systemctl start stonks-sell.service
journalctl -u stonks-sell -n 40
```

## 6. Go live

From here on the job sells without asking. It sells only during regular US
market hours (9:30 AM-4:00 PM ET), so doing this after the close is safe: the
first live action is at 06:32 Pacific.

```bash
install -d /etc/systemd/system/stonks-sell.service.d
install -m 644 /root/gitclones/stonks/deploy/systemd/live/override.conf \
  /etc/systemd/system/stonks-sell.service.d/override.conf
systemctl daemon-reload
systemctl show -p ExecStart --value stonks-sell.service    # must show --execute --yes
```

The override replaces the service's command with the live one; the unit file
itself keeps shipping a dry run. The threshold, 10 percent, is the
`--take-profit` value in both files.

To return to the dry run, delete the override and reload:

```bash
rm /etc/systemd/system/stonks-sell.service.d/override.conf
systemctl daemon-reload
```

## 7. The weekly cycle: sell everything on Monday, buy it back on Tuesday

On top of the take-profit rule, the same box can run the weekly rhythm the
backtests use: every Monday, ten minutes before the close, sell every
position; on Tuesday, just after the open, buy the optimizer's recommendation
again. The sell and the buy are dry runs until step 7c.

The amount re-bought is whatever the sell runs brought in since the last
re-buy: Monday's sale, and any take-profit sales of the week. Each `stonks sell`
run puts its filled orders on a ledger, and `stonks rebuy` spends that, capped
at the buying power Robinhood reports. The cycle moves the same money around
and borrows nothing more. Money you move by hand is not on the ledger and is
never re-bought.

### 7a. A second install for the optimizer

The recommendation needs `torch` and `yfinance`, too heavy for the live install
on a small box. Keep them apart: a second checkout with its own virtual
environment. On the 1 GB box this was tested on, a run with a week-old price
cache takes about fourteen minutes and 315 MB, which leaves the box with very
little spare, so it must never overlap the take-profit job.

```bash
git clone https://github.com/luisdiaz1997/stonks.git /root/gitclones/stonks-opt
cd /root/gitclones/stonks-opt
git checkout master
python3 -m venv .venv
.venv/bin/pip install --no-deps -e .
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install pandas numpy pyarrow yfinance
OMP_NUM_THREADS=1 .venv/bin/stonks portfolio --top-n 500 --months 3 --gamma 40 -k 25
```

The last command downloads and ranks the whole universe on first use and then
prints a recommendation. It places no orders and needs no credentials. Run it
outside the session: it uses nearly all the spare memory for a few minutes.

### 7b. The three timers (the sell and the buy still dry runs)

```bash
cd /root/gitclones/stonks
install -m 644 deploy/systemd/stonks-weekly-sell.service deploy/systemd/stonks-weekly-sell.timer \
                deploy/systemd/stonks-weekly-pick.service deploy/systemd/stonks-weekly-pick.timer \
                deploy/systemd/stonks-weekly-buy.service deploy/systemd/stonks-weekly-buy.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now stonks-weekly-pick.timer stonks-weekly-sell.timer stonks-weekly-buy.timer
systemctl list-timers | grep stonks
```

| Timer | When (Pacific) | What it runs |
|---|---|---|
| `stonks-weekly-pick` | Tuesday 05:15 | In the second install: the recommendation, saved to its `data/tuesday_pick.json`. No orders |
| `stonks-weekly-buy` | Tuesday 06:36 | `stonks rebuy --pick`: the saved recommendation, sized by the ledger |
| `stonks-weekly-sell` | Monday 12:50 | `stonks sell --everything`: every position the rule can sell |

The pick runs before the open for two reasons: Monday's close is then the last
bar it sees, and it is over by about 05:30, an hour before the take-profit
job's first run at 06:32. If it fails, systemd retries every five minutes, up
to three times. The unit caps its memory and makes it the first process the
kernel would kill, so a bad run costs a recommendation, not the website. The
buy refuses a recommendation more than twelve hours old, so a pick that never
succeeded means a quiet Tuesday: the money waits a week. Neither half runs on
a day the exchange is closed.

Start each by hand once, in this order, and read what it prints. The first
places nothing; the second is a dry run; with nothing on the ledger yet the
third says `Nothing to re-buy`.

```bash
systemctl start stonks-weekly-pick.service; journalctl -u stonks-weekly-pick -n 30
systemctl start stonks-weekly-sell.service; journalctl -u stonks-weekly-sell -n 40
systemctl start stonks-weekly-buy.service; journalctl -u stonks-weekly-buy -n 30
```

### 7c. Go live

Two override files, one for each half that trades:

```bash
install -d /etc/systemd/system/stonks-weekly-sell.service.d /etc/systemd/system/stonks-weekly-buy.service.d
install -m 644 /root/gitclones/stonks/deploy/systemd/live/weekly-sell-override.conf \
  /etc/systemd/system/stonks-weekly-sell.service.d/override.conf
install -m 644 /root/gitclones/stonks/deploy/systemd/live/weekly-buy-override.conf \
  /etc/systemd/system/stonks-weekly-buy.service.d/override.conf
systemctl daemon-reload
systemctl show -p ExecStart --value stonks-weekly-sell.service stonks-weekly-buy.service    # both must show --execute --yes
```

To stop the cycle, disable its timers together:

```bash
systemctl disable --now stonks-weekly-pick.timer stonks-weekly-sell.timer stonks-weekly-buy.timer
```

Disabling only the sell leaves the buy re-spending the take-profit sales of
every week, so always stop both halves.

### What the cycle leaves alone

- Positions under $1.00, ones with an open sell order and ones bought the same
  day, for the reasons in step 4's table. A few cents of such positions can
  stay behind for good.
- A rejected order. The take-profit run stops at its first failure; the
  Monday run reports it and goes on to the next position, so one rejected
  order does not leave the rest of the account unsold. Either way only what
  filled is re-bought.
- A sale made in the app. It is not on the ledger and is never re-bought.
- A Monday on a half-day schedule: 12:50 is after the close, so nothing is
  sold that week.

## Day to day

```bash
journalctl -u stonks-sell -f                  # watch runs arrive
journalctl -u stonks-sell --since today       # what today's runs did
systemctl start stonks-sell.service           # run now: live or dry run, per the override
systemctl list-timers | grep stonks           # when the next one is
```

Order status from a shell needs the credentials loaded:

```bash
set -a; . /etc/stonks/robinhood.env; set +a
/root/gitclones/stonks/.venv/bin/stonks orders
```

To update the code:

```bash
cd /root/gitclones/stonks
git fetch origin && git checkout <branch> && git pull --ff-only
.venv/bin/pip install --no-deps -e .
.venv/bin/stonks sell                         # a dry run by hand; this form never sells
```

The unit files in `/etc/systemd/system` are copies. If `deploy/systemd/`
changed, repeat the `install` line from step 5 and run `systemctl daemon-reload`.

To stop the job for good:

```bash
systemctl disable --now stonks-sell.timer
```

## Sign-ins are the one thing it cannot do alone

The job reuses the saved session and signs in again only when that expires. A
fresh sign-in makes Robinhood ask for approval in its app, so the job waits and
systemd ends it after 10 minutes. That run sells nothing, and the next one asks
again. When it happens, either approve the prompt within those 10 minutes, or
sign in by hand with the two commands from step 3. How long a session lasts is
not known; the one measured so far was still valid after 24 hours.

## When something goes wrong

`journalctl -u stonks-sell -n 50` shows the failing run. A failed run never
sells anything further: the first failed order stops the rest.

| The journal says | Meaning | What to do |
|---|---|---|
| `Robinhood credentials are missing` | The env file is empty, or a value did not load | Redo step 2 |
| `Authentication may be expired`, then `Verification required` | The session expired and Robinhood is waiting for an approval | Approve the prompt, or redo step 3 |
| `Robinhood login failed` | Wrong credentials, or an approval never arrived | Redo steps 2 and 3 |
| `Robinhood returned no positions`, `no quotes` or `no market hours` | A read failed: an outage, a rate limit, or an API change | Nothing was sold; run it again |
| `Robinhood returned no order confirmation` | An order got no readable answer, often a rate limit. It may or may not have been placed | Check the app's activity before retrying |
| `N of M orders returned an error` | The first of those orders was rejected; the rest were not attempted | Check the app's activity before retrying |
| `Nothing to re-buy: the ledger holds $0.00` (weekly buy) | No sale is recorded since the last re-buy | Nothing to do; a cycle begins with a Monday sale |
| `the recommendation in ... is N hours old` (weekly buy) | The pick did not finish that morning | Read `journalctl -u stonks-weekly-pick`; the money waits for the next Tuesday, or run the pick and then the buy by hand |

Each run makes roughly one request per position plus a few, so a large account
runs into rate limits sooner.

If `rsync` over SSH fails with a protocol error, the server is printing
something at login (for example `neofetch` in `.bashrc`). Use `tar` over SSH, or
move that command below the interactive check in the file.

## Where things live

| Path | Role |
|---|---|
| `/root/gitclones/stonks` | The code and its `.venv` |
| `/etc/stonks/robinhood.env` | Username and password, readable by root only |
| `~/.tokens/robinhood.pickle` | The session the job reuses |
| `/etc/systemd/system/stonks-sell.service` | How one run starts: as root, with the credentials, a 10-minute limit, and `/usr` and `/etc` read-only |
| `/etc/systemd/system/stonks-sell.timer` | When: every 15 minutes, 06:32 to 12:47 Pacific |
| `/etc/systemd/system/stonks-sell.service.d/override.conf` | Live mode. No file means dry run |
| `/root/gitclones/stonks/data/last_robinhood_sells.json` | Order IDs of the latest sell submission |
| `/root/gitclones/stonks/data/sell_ledger.json` | The weekly cycle's ledger: what each sell run brought in and what each re-buy spent |
| `/root/gitclones/stonks-opt` | The second install: the optimizer with `torch`, and its price cache |
| `/root/gitclones/stonks-opt/data/tuesday_pick.json` | Tuesday's saved recommendation |
| `/etc/systemd/system/stonks-weekly-{sell,pick,buy}.{service,timer}` | The weekly cycle: Monday 12:50, Tuesday 05:15 and Tuesday 06:36 Pacific |
| `/etc/systemd/system/stonks-weekly-{sell,buy}.service.d/override.conf` | Live mode for the two halves that trade. No file means dry run |
