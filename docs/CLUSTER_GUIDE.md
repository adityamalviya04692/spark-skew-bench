# Running the cluster experiment on Azure Databricks
### A complete walkthrough, assuming no prior Azure knowledge

**Time:** ~1 hour of your attention, spread over 1–2 days (one step needs Microsoft's approval, which can take minutes or hours).
**Cost:** roughly **$10 (₹850)**, paid out of Azure's $200 free credit. Net cash out of pocket: **₹0**, if you follow the shutdown steps.

---

## Read this first — one thing that will surprise you

**You cannot do this on an Azure Free Trial subscription.** I checked carefully, because this is where most people get stuck and give up.

A free trial subscription has a hard limit of about **4 CPU cores** and — this is the part that catches people — **is not allowed to request more**. If you try, Azure returns:

> *"Azure subscription 1 is not eligible for quota increases. Consider upgrading your subscription."*

Databricks needs more than 4 cores for even a small cluster, so the trial simply cannot run one.

**The fix is easy and does not cost you money:** you upgrade the subscription to "Pay-As-You-Go". **Your unused $200 credit carries over.** You are still spending credit, not cash. The upgrade exists because Microsoft won't let trial accounts request more capacity — that's all.

The trade-off you're accepting: after upgrading, the automatic spending cap is removed. That's why Step 9 (shutting things down) is not optional. I'll flag it again when we get there.

---

## What you're actually doing, in plain English

You have already proven, on your laptop, that Spark's automatic skew fix (AQE) did nothing while manual salting worked. The paper predicts one more thing it could not test:

> On a real cluster, data moving between machines has to cross the **network**, which is slower than moving it around inside one machine. Salting duplicates some rows. On a laptop those duplicates are cheap. On a cluster they should be **expensive** — so the ideal amount of salt should be **smaller**.

The cluster run either confirms that or refutes it. **Both outcomes are publishable.** You are not hoping for a particular answer; you are removing an "unknown" from your paper.

---

# STEP 1 — Create an Azure account

**What you need:** an email address, a phone number, and a credit or debit card.

> **The card is for identity verification.** Azure places a temporary **$1 hold** which is reversed. It is not a charge. You cannot be billed during the trial because the spending limit is on.

1. Open **https://azure.microsoft.com/free**
2. Click **Start free**.
3. Sign in with a Microsoft account, or click **Create one!** to make one.
4. Fill in your details. Country: **India**.
5. Enter your phone number, click **Text me**, type in the code.
6. Enter your card details.
7. Accept the agreement → **Sign up**.

**You should now see:** a page saying you have **$200 credit, valid 30 days**.

✅ **Checkpoint:** Go to **https://portal.azure.com**. You should see the Azure dashboard. If you do, Step 1 is done.

---

# STEP 2 — Upgrade to Pay-As-You-Go

This is the step that unblocks everything. It does not charge you.

1. In the Azure portal, click the **search bar at the very top**.
2. Type `Subscriptions` and click the **Subscriptions** result.
3. Click your subscription (likely named **"Azure subscription 1"** or **"Free Trial"**).
4. On the left menu, look for **Upgrade subscription** (sometimes just **Upgrade**). Click it.
5. If asked for an account name, type anything — `Personal` is fine.
6. Confirm the upgrade.

**What you should see:** the subscription type changes to **Pay-As-You-Go**, and your remaining credit is still shown.

> ⚠️ **What just changed:** the spending limit is now off. Azure can now bill your card if you use more than $200 of resources. Step 9 tells you how to make sure that never happens. Nothing you do in this guide will approach $200.

✅ **Checkpoint:** on the Subscriptions page, your subscription says **Pay-As-You-Go**.

---

# STEP 3 — Ask Azure for enough CPU cores (quota)

Azure gives new subscriptions almost no capacity by default. You have to ask. It's a form, it's free, and it's usually approved quickly.

**We want 20 cores in Central India** — that's 5 machines × 4 cores. (Central India is the cheapest Indian region for these machines; I checked.)

1. Search bar → type `Quotas` → click **Quotas**.
2. Click **Compute** in the left menu.
3. At the top there are dropdown filters. Set:
   - **Subscription:** your subscription
   - **Region:** **Central India**
4. In the search box on that page, type: `DDSv5`
5. Find the row **Standard DDSv5 Family vCPUs**. The **Current limit** will probably say **0**.
6. Tick the checkbox next to that row.
7. Click **New Quota Request** at the top → **Enter a new limit**.
8. Type **40** (asking for a bit more than the 20 you need costs nothing and saves a second request).
9. Click **Submit**.

**What happens next:** usually approved within a few minutes, sometimes a few hours. You'll get an email. The page will show **Approved**.

> **If it gets rejected**, or if you'd rather not wait: repeat the same steps but search for `DSv2` instead and request **40** on **Standard DSv2 Family vCPUs**. That's an older machine family that's more reliably available on new accounts. If you use this fallback, note it — in Step 6 you'll pick `Standard_DS3_v2` instead of `Standard_D4ds_v5`.

✅ **Checkpoint:** the quota row shows a limit of **40**, not 0. **Do not continue until this says 40.** Everything after this will fail otherwise.

---

# STEP 4 — Create the Databricks workspace

A "workspace" is your Databricks environment — where notebooks and clusters live.

1. Search bar → type `Azure Databricks` → click it.
2. Click **+ Create** (or **Create Azure Databricks service**).
3. Fill in the **Basics** tab:

   | Field | What to put |
   |---|---|
   | Subscription | your subscription |
   | Resource group | click **Create new**, name it **`skewbench-rg`** |
   | Workspace name | **`skewbench-ws`** |
   | Region | **Central India** (must match your quota region) |
   | **Pricing Tier** | **Trial (Premium — 14-Days Free DBUs)** |

   > 💡 **The Pricing Tier field is the one that matters.** Choosing **Trial** makes Databricks' own software charge **$0** for 14 days. You'll only pay for the underlying machines — about **$2.44/hour**, roughly **₹200/hour**. Miss this and you pay roughly three times more.

   > 📝 **Remember `skewbench-rg`.** In Step 9 you delete that one thing and everything disappears with it.

4. Click **Review + create** → **Create**.
5. Wait 3–5 minutes.
6. When it says *"Your deployment is complete"*, click **Go to resource**, then click the big **Launch Workspace** button.

A new browser tab opens — that's Databricks. This is the environment you already know from work.

✅ **Checkpoint:** you're looking at the Databricks workspace with a left sidebar (Workspace, Catalog, Compute, Jobs…).

---

# STEP 5 — Put the code and a storage location in place

### 5a. Put the code on GitHub

Databricks pulls code from GitHub. If your repo isn't public yet, make it public — **just check it contains no client-identifiable material first.**

On your Mac, inside the `spark-skew-bench` folder:

```bash
git init
git add -A
git commit -m "skewbench: benchmark and cost model for Spark skewed-join mitigation"
gh repo create spark-skew-bench --public --source=. --push
```

(If `gh` isn't installed, create the repo on github.com and follow its push instructions.)

Copy the repo URL — e.g. `https://github.com/YOURNAME/spark-skew-bench`.

### 5b. Pull the code into Databricks

1. In the Databricks left sidebar click **Workspace**.
2. Click your username, then the **Create** button (top right) → **Git folder**.
3. Paste your GitHub URL. The other fields fill in automatically.
4. Click **Create Git folder**.

> No password or token needed — Databricks clones public repos anonymously.

✅ **Checkpoint:** you can see your project files (`src`, `config`, `cluster`…) in the Databricks Workspace browser.

### 5c. Create a storage location (a "volume")

This is where data and logs go.

> ⚠️ **Do not use DBFS.** Older tutorials tell you to use paths starting `dbfs:/`. That storage is deprecated, and **new workspaces are created without it**. Those instructions will fail. We use a Unity Catalog **volume** instead.

1. Left sidebar → **Catalog**.
2. Expand your workspace catalog — on a new workspace it is named after the
   workspace itself, e.g. **`skewbench_ws`** — then click the **`default`** schema.
3. Click **Create** → **Create volume**.
4. Name it **`skewbench`**. Type: **Managed**. Click **Create**.

Your storage path is now: **`/Volumes/skewbench_ws/default/skewbench`**

> 📌 **Write this path down.** It is the one value the notebooks cannot guess.
> If your catalog is named something else, the path changes accordingly, and
> you set it once in `cluster/notebooks/_bootstrap.py` on the `VOLUME` line.
> Right-click the volume in Catalog → **Copy path** to get it exactly.

✅ **Checkpoint:** `skewbench` appears under `skewbench_ws` → `default` → Volumes.

---

# STEP 6 — Create the cluster

This is the most detail-sensitive step. Five settings matter; the defaults are wrong for all five.

1. Left sidebar → **Compute** → **Create compute**.

2. Set each field exactly:

   | Setting | Value | Why it matters |
   |---|---|---|
   | **Compute name** | `skewbench-cluster` | — |
   | **Policy** | Unrestricted | Lets you set the rest |
   | **Machine learning** | OFF | Not needed |
   | **Databricks Runtime** | **`16.4 LTS (Scala 2.12, Spark 3.5.2)`** | ⚠️ **The default is a Spark 4.x version.** Your laptop ran Spark 3.5.3. Using Spark 4 would make the two runs incomparable. You must change this by hand. |
   | **Use Photon Acceleration** | ⚠️ **UNCHECK IT** | Photon is a different, faster engine. It's **on by default**. It would invalidate the comparison *and* roughly double the cost. |
   | **Worker type** | `Standard_D4ds_v5` | 4 cores each. *(Use `Standard_DS3_v2` if you took the fallback in Step 3.)* |
   | **Workers** | **4** — turn OFF autoscaling | A fixed size keeps the experiment controlled |
   | **Driver type** | Same as worker | — |
   | **Terminate after** | **20** minutes of inactivity | Your safety net against a surprise bill |

3. Now expand **Advanced options** at the bottom.

4. Click the **Logging** tab. **This is the step you cannot go back and add later.**
   - **Destination:** choose **Volumes**
   - **Path:** `/Volumes/skewbench_ws/default/skewbench/logs`

   > 🔴 **Why this is critical:** every measurement in your study is computed from Spark's event logs. Databricks only saves them if you turn this on **before the cluster starts**. If you forget, the run completes, produces timings, and you get **no metrics at all** — and you cannot recover them. You'd have to run the whole thing again.

5. Click the **Spark** tab (still under Advanced options). In the **Spark config** box, paste this one line:

   ```
   spark.sql.adaptive.enabled false
   ```

   > The benchmark switches AQE on and off per test case. This just makes sure the cluster doesn't start with it forced on.

6. Click **Create compute**. Starting takes 5–10 minutes.

✅ **Checkpoint:** the cluster shows a **green dot** and says **Running**.

> ❌ **If you see "quota exceeded" or the node type is greyed out:** go back to Step 3. Either the quota isn't approved yet, or it was approved for a different region or a different machine family. Region and family must both match.

---

# STEP 7 — Preflight (90 seconds)

> Everything from here on runs from the four notebooks in
> `cluster/notebooks/`. You do not paste code into Databricks any more, and you
> do not edit any paths. Earlier drafts of this guide told you to clone
> `run_cluster.py` and hand-edit `YOUR_EMAIL_HERE` — ignore that; the notebooks
> replace it. They find the repo themselves by walking up from wherever they sit
> until they see `src/skewbench/runner.py`, because a mistyped workspace path was
> the single most common way this setup failed.

1. Left sidebar → **Workspace** → `spark-skew-bench` → `cluster` → `notebooks`
2. Open **`00_preflight`**
3. Top right, next to **Connect**, choose **`skewbench-cluster`** and wait for the green dot
4. **Run all**

Six checks print. The one that decides whether the run is worth starting:

```
PASS  Databricks detected   True
```

`False` means the platform was not detected, so the harness will try to set
`spark.eventLog.*` itself, fight Databricks' own logging, and collect **no
metrics at all**. The run would complete, look fine, and contain no science.
Stop and fix it rather than proceeding.

The other five checks: `datagen_spark` imports (proves the Git pull took), the
code fingerprint prints (record it — results with different fingerprints cannot
be pooled), the three configs exist, the volume is writable (a read-only volume
otherwise fails at the *end* of a three-hour run), and executors are online.

> ❌ **`ModuleNotFoundError: skewbench`** → you are running the notebook from
> outside the Git folder. Open it from **Workspace**, not from a copy.
> ❌ **`Could not find the repo root`** → same cause, same fix.
> ❌ **Volume not writable** → open `_bootstrap` and check the `VOLUME` line
> matches your volume exactly. It ships set to
> `/Volumes/skewbench_ws/default/skewbench`.

---

# STEP 8 — Smoke test (~10 minutes, saves you hours)

Open **`01_smoke`** → **Run all**. Three cells at 6M rows — one tenth scale,
identical machinery.

### Checkpoint (a): metrics present

```
cells without metrics: none
```

Any cell listed there means the event-log parser found nothing for it. **Do not
start the full run.** Timings would still be written, but the straggler index,
the shuffle-read skew and the "did AQE actually fire" evidence would all be
missing — a run in that state looks complete and is not.

### Checkpoint (b): the projection

The last cell extrapolates the smoke timings to the real grids:

```
median cell wall time       48.3 s  (6M rows)
projected cell wall time   483.0 s  (60M rows)

config/cluster_salt.yaml
   cells  44   workloads 4 (4 new)
   measurement 5.9 h   data generation 1.3 h
config/cluster_aqe.yaml
   cells  10   workloads 2 (0 new)
   measurement 1.3 h   data generation 0.0 h

PROJECTED TOTAL             8.5 hours  (floor, not a forecast)
```

The figure is a floor rather than a forecast, because shuffle cost grows
slightly faster than linearly once spill begins. If it exceeds about four hours,
the script prints the cut list in priority order:

1. `repetitions: 5 → 3` in both configs (saves ~33%)
2. drop `k: 64` from `salt_selective` (saves ~9% of the salt grid)
3. `n_fact: 60000000 → 30000000` (saves ~50%; weakens the network-cost argument
   but does not invalidate it)

Two things are never cut: the `baseline` arm and the
`aqe_coalesce_enabled: false` profile. Both are controls, and without them the
run answers nothing.

---

# STEP 9 — The two real runs

Run **`02_run_salt`** to completion, then **`03_run_aqe`**. Not in parallel.

### Why two files and not one

The old single `cluster.yaml` expanded to **156 cells** — 4 workloads × 3 Spark
profiles × 13 arm-`k` combinations, or 936 measured joins at 60M rows. That is
well over a day of cluster time, and it answered two different questions out of
one file. It has been deleted. In its place:

| Notebook | Config | Cells | Question |
|---|---|---|---|
| `02_run_salt` | `config/cluster_salt.yaml` | 44 | Does `k*` fall as 1/√P, and is it smaller on a cluster than on one node? AQE off throughout. |
| `03_run_aqe` | `config/cluster_aqe.yaml` | 10 | Does the single-node AQE null survive where the skew rule's preconditions are comfortably met? |

Each file carries its **own `baseline` arm**. That is the important part. A
wall-clock measured in one Spark session cannot be compared against one from
another session — JIT state, cache warmth and executor placement all differ, and
comparing across sessions is what once inflated this study's noise floor from 2%
to 29%. Keeping a baseline inside each file means every ratio you report was
measured against a baseline from the same session. Running the two notebooks
concurrently against the same volume would reintroduce exactly the problem the
split was designed to remove.

### Why the AQE grid is the honest test

The obvious objection to the single-node null is that AQE never fired because
the hot partition was 11.41 MB against a 16 MB advisory size — that the study
tested a fixture, not a mechanism. At cluster scale that objection closes by
construction:

* θ = 1.2 over 2000 engines puts **22.23%** of rows on the hot key
* 60M × 0.2223 = **13,338,446** hot rows
* at 100–180 bytes per row the hot partition is **1,334–2,401 MB**

against a 256 MB threshold, a 5× median factor (median partition ≈ 30–54 MB) and
a 64 MB advisory size. All three of `OptimizeSkewedJoin`'s conjunctive
preconditions clear by roughly an order of magnitude each. A null under those
conditions is a statement about the rule, not about the test rig.

### While they run

> 💡 **If either dies partway** — laptop sleeps, connection drops, cluster
> auto-terminates — just run the cell again. Results are appended and `fsync`'d
> after every cell and completed cells are skipped by `run_id`. You lose at most
> one cell. This was built in after a 32-cell run was lost to exactly this.

You can close the browser; the run continues. Reopen the notebook to watch
`[1/44]`, `[2/44]` scroll past.

### What to read first when they finish

`02` ends with a `[control]` block comparing `baseline` against
`salt_selective(k=1)`. Those two are the *same query with the same settings* —
identical plans. Whatever gap appears between them is pure measurement noise.
**Any effect smaller than that gap is not a result.** Read this number before
any other.

`03` prints, per AQE cell, whether `skew` appears in the physical plan. A
speedup without it is partition coalescing, not skew handling, and must not be
written up as the latter.

---

# STEP 10 — Collect, verify, and only then shut down

⏳ **Leave the cluster running for 5 minutes after `03` finishes.** Databricks
flushes event logs roughly every 5 minutes; terminating immediately loses the
last batch of measurements.

Open **`04_collect`** → **Run all**. It prints a verification block and then
bundles everything.

Four lines decide whether the run is usable:

1. **`rows` equals 44 and 10.** Fewer means cells were skipped.
2. **`with_metrics` equals `rows`.** Anything less and part of the task-level
   evidence does not exist for those cells.
3. **`sessions` has exactly one entry per file.** More than one means the file
   was assembled from separate Spark sessions and its wall-clock comparisons are
   not valid.
4. **`fingerprints` has exactly one entry.** More than one means the code changed
   mid-run, so the rows are not pooled results of a single program.

It then writes `cluster_bundle.zip` to the volume, containing both JSONLs, the
control and summary files, the smoke results, and **the event logs**. The logs
matter as much as the JSONL: with them the run can be re-parsed for a metric
nobody thought to record, and without them a new question costs another three
hours of cluster time.

Download it: **Catalog → `skewbench_ws` → `default` → Volumes → `skewbench` →
`results` → right-click `cluster_bundle.zip` → Download.**

---

# STEP 11 — Shut everything down 🔴

**Do not skip this.** The spending limit was removed in Step 2. A forgotten
cluster costs roughly **₹200/hour, about ₹4,800/day**.

### Immediately after collecting

1. Left sidebar → **Compute**
2. `skewbench-cluster` → **⋮** → **Terminate**
3. Confirm the green dot is gone

Terminate, do not delete — keep the cluster definition in case a re-run is
needed.

### When the paper is final — delete everything

1. Azure portal → search **Resource groups**
2. Click **`skewbench-rg`** → **Delete resource group**
3. Type `skewbench-rg` to confirm

This removes the workspace, the machines and the storage in one action.
**Download the bundle first.**

### Check what you spent

Azure portal → **Cost Management** → **Cost analysis**. Expect roughly **$10**.

**Set a budget alarm now:** Cost Management → **Budgets** → **+ Add** → amount
**$25** → alert at **80%**.

---

# What can go wrong, and what to do

| Symptom | Cause | Fix |
|---|---|---|
| *"not eligible for quota increases"* | Still on Free Trial | Step 2 — upgrade first |
| Cluster won't start, *"quota exceeded"* | Quota not approved, or wrong region/family | Step 3 — region **and** machine family must both match |
| Node type greyed out | Machine family not enabled on a new subscription | Use `Standard_DS3_v2` + `DSv2` quota instead |
| `ModuleNotFoundError: skewbench` | Notebook opened outside the Git folder | Open it from **Workspace** → `spark-skew-bench` → `cluster/notebooks` |
| `Could not find the repo root` | Same as above | Same as above |
| `Permission denied` on `/Volumes/...` | `VOLUME` in `_bootstrap` doesn't match your volume | Catalog → your volume → Copy path → paste into `_bootstrap` |
| Anything with `dbfs:/` fails | DBFS is deprecated; new workspaces don't have it | Use `/Volumes/skewbench_ws/default/skewbench/...` |
| `Databricks detected: False` | Git pull didn't take, or old code | Pull again, re-run `00_preflight` |
| Run finished but no metrics | Log delivery wasn't enabled before cluster start | Not recoverable — enable it (Step 6.4) and re-run |
| Cluster keeps stopping | Auto-termination firing | Expected. Restart it; the run resumes where it left off |
| `git push` rejected, *"fetch first"* | GitHub repo was created with a README | `git pull --rebase origin main` then `git push -u origin main` |

---

# Honest notes for the paper

Three things to state plainly rather than hope nobody checks:

1. **Spark version differs slightly.** The single-node study ran **3.5.3**; the
   Databricks LTS runtime carries **3.5.2**. A patch-level difference, almost
   certainly irrelevant — but disclose it rather than claim an exact match.
2. **Photon must be off, and the paper should say it was.** It is a different
   execution engine; leaving it on would make the comparison meaningless.
3. **The cluster is 4 workers × 4 cores.** Small. It has real network shuffle,
   which is the entire point, but it is not a production-scale result and must
   not be described as one.

---

# What happens to the results

Send back `cluster_bundle.zip`, and the analysis will:

1. Attach the task-level measurements from the event logs
2. Run `analysis/compare_scales.py`, which puts single-node against cluster and
   states whether the 1/√P prediction **held**, **failed**, or **remains
   unresolved**
3. Write up whichever of the three actually happened

**All three are publishable.** If the prediction holds, the paper's main
limitation becomes a validated result. If it fails, that is more interesting
than what was predicted. If the replication cost is *still* too small to measure
even across a network, that is a strong practical finding on its own: selective
salting is essentially free to over-provision, and the folklore warning against
large `k` applies only to the uniform variant.

The one outcome that would be bad is claiming an answer the data does not
support. That is why the analysis refuses to report a `k*` it cannot identify.
