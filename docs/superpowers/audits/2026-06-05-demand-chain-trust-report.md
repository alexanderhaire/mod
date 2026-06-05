# Demand-Chain Trust Report — (Open Orders + Sales) → BOM → Raw Materials

**To:** Owner
**From:** Lead reliability engineer
**Date:** 2026-06-05
**Scope:** The purchasing engine that turns open customer orders (and historical sales) into a raw-material buy list, via BOM explosion. Every finding below was independently re-verified by a second agent; I report the corrected verdict/severity, not the first-pass one.

---

## 1. Bottom line

**No — you cannot trust today's buy list to decide what to buy.** The chain has at least four independent defects that each produce a *wrong quantity* or *wrong/missing item*, and they stack. The dominant one alone over-states raw demand by **14x–23x** for every product that has more than one stored recipe: on the live Triangle order 29138 (TRITOPS250 × 4,000 gal) the engine says **buy 192,119 LBS of Boric Acid** when the correct number is **~14,000 LBS** — a single order asking for ~74% of a *full year* of real Boric Acid usage. On top of that, in-house "REC-" receiving codes generate **~$115K of pure phantom buys** for items that already sit in stock under their base code, the engine recommends buying items you have **never purchased from any vendor and physically cannot order** (NPK3011, 641,063 units), and a real heavily-sold product (PETRO1300, 7,000 gal on order 29085) **vanishes to zero raw demand with no warning at all**. The arithmetic is only "safe" today on a handful of items, and only because high inventory happens to clip the inflated numbers to zero — the moment stock draws down, the over-buys fire. The engine should be treated as a triage aid, not a buy authority, until the must-fix list below is done. Notably, the problem is **NOT units/density confusion** — the UoM handling is sound; the magnitude errors are all recipe-row summing.

---

## 2. What actually works (confirmed)

These links held up under verification and can be trusted as-is:

- **UoM / density encoding is correct (UOM-1, UOM-2, BOM-5, UOM-5).** The `GALxx.xx` tokens are GP UoM-schedule *names* whose suffix is the product's lbs-per-gallon density; finished goods are denominated in gallons, and BOM `QUANTITY_I` is already stored as **LBS-of-raw per 1 gallon of finished good**, so a clean single-recipe item explodes dimensionally correct with no conversion needed. Proven on FLO150016250: BOM = 12.640 LBS/gal of NPKNPHURCDI, and MO ground truth (C2501-0074: 12,640 LBS → 1,000 gal) = 12.640 LBS/gal exactly. **Do not add a density divisor** — that would *introduce* an error.
- **No live UoM scale error on the current buy list (UOM-5).** Zero open SOP lines are in a non-base UoM today; the only large-factor units present (TON, EACH5) are each in their item's own base unit and are not raw-explodable.
- **Step-1 quantity sums are unit-safe (S1-05).** No item appears on open demand with mixed UoMs, so per-item `total_qty` aggregation is not mixing units.
- **On-hand location selection is correct (REC-6a).** The netting reads the real `MAIN` location row and never the blank-`LOCNCODE` aggregate; verified that the blank row equals the sum of real-location rows for every item (no double-count from this path).
- **Pack-size variants are modeled as independent items correctly (REC-6b).** SOARBLM02 vs SOARBLM250 carry their own per-pack BOM quantities (NO3CA 101.69 vs 67.79); do **not** force-normalize them to a "00" parent.
- **Recursion reaches true raw leaves (BOM-4).** The explosion descends through WIP/intermediates to RAWMAT-classed leaves; containers/discontinued leaves that slip through are correctly dropped downstream by the class filter.
- **Synthetic-BOM path is rarely reached and is not corrupting today's buy list (2b-1).** For the live/test FGs it contributes nothing; only 2 reachable items, both packaging.
- **The "control" reconciliations check out.** Single-recipe **NPKNPHURCDI reconciles at ~1.22x** (implied vs actual) — proof the chain is arithmetically correct *when* the BOM has clean single rows. **SO4MN implied = 0 is correct** (dormant item, superseded by NO3MN), not a chain failure.

---

## 3. What's broken & why it has "never worked" — ranked by severity

I have **consolidated** the findings: the same root bug was reported up to six times. Each numbered item below is one distinct code defect, with all cross-confirming evidence folded in. Items 1–6 produce a **wrong buy quantity or wrong/missing item** and lead the list.

### 🔴 #1 — Recursive BOM SUMs every stored recipe version → 14x–23x over-buy (HIGH)
*(BOM-1, BOM-2, F1, Recon-F1, Recon-F2, UOM-3 — all confirm this single defect)*

- **Symptom:** For any finished good with more than one stored recipe, the per-unit raw requirement is multiplied by the number of stored recipe rows. Live: order 29138 (TRITOPS250 × 4,000 gal) → **SO4BORIC = 192,119 LBS** emitted vs **~14,000 LBS** correct (**13.0x**); also CHEMONETH 58,689 LBS and NPKUREA 39,769 LBS, similarly inflated.
- **Root cause:** `fetch_recursive_bom_for_item` (`inventory_queries.py` ~L266–299) selects `FROM BM010115 WHERE PPN_I=?` with **no `BOMCAT_I` / `BOMNAME_I` filter**, then `GROUP BY ComponentItem` + `SUM(QUANTITY_I)`. `BM010115` stores the one active manufacturing BOM (`BOMCAT_I=1`, blank `BOMNAME_I`) **plus** every archived/named shop-floor batch recipe (`BOMCAT_I=4`, names like `'BAGS 4 AND 5'`, `'USING 20-0-0'`, `'NB 107'`). For TRITOPS00/SO4BORIC there are **13 rows** (1 canonical at 3.512 + 12 archived at 3.05–3.95), summing to **48.0298 = 13 × ~3.69**. The inflated `Design_Qty` flows untouched: `explode_demand_to_raw_materials` (`kanban_reorder.py` ~L488–499) does `required = Design_Qty × order_qty` → gross → net, with no reconciliation. **30% of parents (440 / 1,461) carry duplicate rows.**
- **It compounds, it is not a constant.** Mass balance proves it independently of any recipe ground truth: summed raw input mass per *one* parent gallon runs **2x–104x** the gallon's own weight (GOLDMZF02 = 1,162 LBS inputs / 11.2 LBS/gal = **103.75x**; SOARBLM02 = 48x; TRITOPS = 13.1x). Multi-level FGs that re-explode multiple package-variant sub-parents amplify it further.
- **Concrete fix:** Add `AND BOMCAT_I = 1 AND RTRIM(BOMNAME_I) = ''` to **both** the anchor and recursive members of the CTE (and to the parent-exclusion subquery). Verified: this restores SO4BORIC to **3.5119 LBS/gal** (one canonical row). Apply the same to `fetch_mfg_bom_grouped_by_component` (it has the identical `SUM` bug, L~231–245). **Caveats from verification:** (a) do **not** "de-duplicate via DISTINCT" — the rows carry *different* quantities, you must select one category, not dedup identical rows; (b) do **not** try to pick "latest revision by effective date" — all `EFFECTIVEINDATE_I/OUTDATE_I` are `1900-01-01` and unusable; (c) the active `BOMCAT_I=1` row **equals true physical consumption** (3.512 LBS/gal matches the MO ledger and IV30300 chunk size), so the active-row fix is correct and sufficient — an earlier worry that it would flip into a "2x under-buy" was **wrong** (that figure double-counted a vendor PO receipt as a second issue).

### 🔴 #2 — In-house "REC-/dilution" raws are dropped and replaced by inflated proxy inputs (HIGH)
*(BOM-3, Recon-F3, REC-6 side-note — one defect)*

- **Symptom:** A real *purchasable* raw material that also has a one-step in-house dilution recipe (REC-X + water) is silently dropped from the leaf output and replaced by its `REC-` input, inflated by the same row-summing. NO3MN (Manganese Nitrate, real 12-mo usage 396,278 LBS) never appears as a leaf; instead it recurses to `REC-NO3MN` summed over 3 archive rows (~2.94x). Affects **~28 RAW-class items** (NO3FE, NO3CU, EDTAMN, CL2CALIQ, …). For multi-level FGs it compounds further: GOLDTP00 → **27.79x**, SOARCIT00 → **17.76x** on NO3MN.
- **Root cause:** Leaf filter `WHERE ComponentItem NOT IN (SELECT DISTINCT PPN_I FROM BM010115)` (`inventory_queries.py` L296). The parent-set subquery includes `BOMCAT_I=4` archive records, so any raw that was ever the subject of a concentration-correction batch is treated as a non-leaf and recursed away. `_accumulate` strips the `REC-` prefix back to the base code, so the **item is correct** — but the **quantity is 3x–12x inflated**.
- **Concrete fix (corrected — the finding's stated fix makes it WORSE):** Editing *only* the exclusion subquery leaves the CTE still recursing NO3MN → REC-NO3MN, producing **both** rows (worse than today). The fix that works is **`AND b.BOMCAT_I = 1` on the recursive CTE JOIN itself**, and/or treat any `ITMCLSCD` starting with `RAWMAT` as a hard leaf even when it appears as a parent (the orchestrator's `_classify` already does this for the drill path — apply the same gate inside the CTE).

### 🔴 #3 — Canned/pack-size parents re-explode the full bulk recipe N times (HIGH)
*(F2, Recon-F4 — one defect, stacks on #1)*

- **Symptom:** A "02" case SKU lists multiple size-variant sub-assemblies that each re-explode to the full bulk recipe, multiplying raws a further **2x–5x** (some correctly 1.0x, AGBAPMN02 up to 23.7x). Live: **PMX1404-0106 recommends buying 21,018 LBS vs a correct ~3,853 LBS (5.46x)** — zero on-hand, zero open PO, so the over-statement flows straight to the purchase quantity. Also FLO15001602 → NPKNPHURCDI at **5x**, FLO15001630 at **2x** (sold through 2026-05, so active).
- **Root cause:** Package variants (00 bulk / 250 tote / 30·55 drum / 02 case) are modeled as nested `BM010115` parents; the "02" BOM lists several alternate-size intermediates, each of which recurses to the full bulk recipe, and the CTE `SUM`s them all with no de-duplication of the same liquid (`fetch_recursive_bom_for_item` SUM-over-CTE with no protection against a leaf reachable via both a direct row and a nested sub-assembly).
- **Concrete fix:** Collapse same-liquid size-variant intermediates to one, or model package variants as a single bulk parent + packaging (the CEIAKARBUP02 / GROMZN02 convention already present in the data). Tie validation to GOLDMZF02 / PMX1404-0106 (expect ~3.85k LBS, not 21k). **Note:** the finding's original proof item (SO4FEDRY19 on order 29128) is overstocked and nets to 0, so validate on PMX1404-0106 instead.

### 🔴 #4 — "REC-" split identity → ~$115K of phantom buys for items you already have in stock (HIGH)
*(REC-3 / "REC-1" split-identity)*

- **Symptom:** `REC-` receiving codes appear as standalone buy rows with on-hand = 0, even though the real stock sits under the base code. Live buy list: **REC-NPKPHOS85 net_need = 225,340** (≈$110K at $0.488/LB) and **REC-SO4FELIQ net_need = 43,900** (≈$4.8K) — while `MAIN` holds **76,244 LBS Phosphoric Acid** and **48,276 LBS Iron Sulphate** under the base codes, never credited, and the base codes never appear on the list. Wrong item *and* wrong quantity.
- **Root cause:** Split identity. Usage history (IV30300) and the kanban-refill layer (`compute_kanban_rates`, `kanban_reorder.py` L129–194) key on the literal `REC-` code; on-hand keys on the base code. `_accumulate` strips `REC-` only on the SOP-explosion path (L356), not on the kanban stream, and `_filter_raw_materials` keeps `REC-` rows because they carry RAWMAT* classes. The existing dilution-proxy band-aid fails here (NPKPHOS85 isn't a BOM parent; SO4FELIQ has only one non-water component).
- **Concrete fix:** Canonicalize `REC-` → base code at the earliest point (inside `compute_kanban_rates` and before the outer merge), then **unify** on-hand + open-PO + usage across `REC-` and base into one buy row under the orderable base code. **Do not simply exclude `REC-`** — that would discard the open POs and usage filed under it.

### 🔴 #5 — `_classify` buys non-purchasable in-house items whole off the kanban layer (HIGH)
*(REC-3 "RAWMAT→vendor")*

- **Symptom:** The buy list recommends items that have **never been bought from any vendor** (0 lifetime PO receipts, 0 open PO lines) — you literally cannot place these POs. Live: **NPK3011 net_need = 641,063** (class RAWMATNTB, "KNO3 Solution 3-0-11", never purchased), plus NIT9392700# (56,000), PREMIX93927 (24,490), and ~6 more — **~718K units of phantom vendor demand.**
- **Root cause:** `_classify` (`kanban_reorder.py` L334–345) returns `'vendor'` for any item whose class starts with `RAWMAT` **before** checking BOM/receipt history, so misclassed in-house intermediates are bought whole instead of exploded. In the live snapshot these reach the list via the **kanban refill layer**, not the BOM-explosion path (the two paths were conflated in the original finding — the on-hand *is* netted correctly; the item itself is just non-orderable).
- **Concrete fix (corrected):** (a) Suppress items with **0 lifetime PO receipts** from the *vendor* buy list (or route in-house intermediates to a production queue). (b) Do **not** route NPK3011 to the synthetic-BOM path as the finding suggested — NPK3011 has zero BM010115 rows; synthesis would replace a phantom buy with 32 components of inflated false demand. Use an allow-list of genuinely-purchased RAWMAT items plus a "must have a real PO receipt" gate.

### 🔴 #6 — Dilution-proxy credits the same physical stock twice → live under-buy (HIGH)
*(REC-5 — severity raised by verifier from MEDIUM to HIGH; confirmed=false only because it is NOT latent as originally claimed)*

- **Symptom:** When both a base raw and its `REC-` row need buying, the on-hand is consumed on the base row **and** re-credited in full as `proxy_credit` on the `REC-` row — the same physical stock counted twice, understating the buy. This is **firing today, not latent**: **REC-NO3CU is under-bought by exactly 17,406 LBS** (its base NO3CU is simultaneously on the buy list as net-short *and* has 17,406 LBS credited to REC-NO3CU), and **REC-NO3FE under-bought by ≥37,423 LBS**.
- **Root cause:** Proxy credit (`net_by_location` ~L628) is computed per-row without reserving stock already consumed by the base row. The proxy was a band-aid for the split identity (#4) rather than a unification.
- **Concrete fix:** Unify `REC-`/base codes (same fix as #4 makes this disappear). If kept, the correct credit is `max(0, base_on_hand − base_gross) × factor`, deducting base-consumed stock before crediting.

### 🟠 #7 — PETRO1300 black hole: a 7,000-gal order produces zero demand and zero warning (HIGH)
*(F3, 2b-3 — one defect)*

- **Symptom:** Any finished good on a live open order with **no `BM010115` parent BOM** explodes to **zero** raw demand and is logged **nowhere** — not in the buy list, not in any diagnostic report. Live: **PETRO1300, 7,000 GAL on order 29085** (Petro Canada; 694 historical sales lines / 4.5M units sold) → 0 raw_needs, 0 reports. Also MAC828YM1 (2.5 TON liquid fertilizer, order 29141).
- **Root cause:** The explosion loops over `bom_cache[parent]`; if `fetch_recursive_bom_for_item` returns `[]`, the FG contributes nothing, the drill loop body never executes (`kanban_reorder.py` L491), and the synthetic fallback is never invoked at the top-FG level. There is **no missing-top-level-BOM diagnostic** (only a missing-leaf-BOM one). MISCCHG is a benign member of this set (it is a NONINV service line that correctly should not explode).
- **Concrete fix:** Add a missing-top-level-BOM report parallel to `missing_bom_report`, class-aware to suppress NONINV/service lines. For PETRO1300-class items either author a `BM010115` BOM or flag "demand not exploded — manual raw planning required."

### 🟠 #8 — MISCCHG: 7 real products sold under one non-inventory SKU → live under-buy + phantom CRITICAL (HIGH)
*(S1-02 — confirmed=false only because the original "all latent" claim was refuted; the under-buy is ACTIVE today)*

- **Symptom:** A non-inventory misc-charge SKU (MISCCHG, ITEMTYPE=4, NONINV) is accepted as a finished-good demand line. Order 29113 carries **7 distinct real manufactured products** (JRM Boost, Cal Amino, Double K, Root Push, Sea Feed, Triad 30, Kit Assembly Fee) under the one MISCCHG code. The engine collapses them into one phantom **CRITICAL qty-61** line with no BOM that explodes to **zero raw materials**. **4 of the 7 lines (qty 40) are real products with real BOMs and genuine past-due demand that are NOT separately on order** — a live wrong-quantity under-buy of every raw in those 4 BOMs (CHEGLUCO, NO3ZN, GRPSEAEX, GRPALGAE, GRPHUMICLIQ, …).
- **Root cause:** No ITEMTYPE / item-class filter on the demand query (`production_queries.py` L99–115). Aggregation by item_number then relabels the merged phantom line with whichever description sorts to top priority, further hiding that 7 products are involved.
- **Concrete fix:** Filter the demand query to `ITEMTYPE IN (1)` (or exclude `ITEMTYPE=4/5` + class NONINV). Recover masked demand by mapping MISCCHG `ITEMDESC` → real `ITEMNMBR`, or stop selling manufactured products under MISCCHG.

### 🟠 #9 — On-hand never subtracts allocations (ATYALLOC) → magnitude under-buys (HIGH severity, partial verdict)
*(REC-4 — confirmed=false: the code defect is real, but its specific "SO4BORIC dropped today" evidence was wrong)*

- **Symptom:** `fetch_on_hand_by_item` returns gross `QTYONHND` and never subtracts `ATYALLOC` (stock already committed to MOs), so available stock is overstated and PO-free items are under-bought. Verified real under-buys on items where on-hand binds: **SO4FEGREENLIQ +63%, NPKUREA +55%.**
- **Root cause:** `inventory_queries.py` L171–176 sums gross `QTYONHND` with no `ATYALLOC` term, contradicting the app's own few-shot SQL which uses `QTYONHND − ATYALLOC`.
- **Correction to the original finding:** It claimed SO4BORIC/NPKNPHURCDI/SO4FEDRY19 are "dropped from today's list only because allocations are ignored." **False** — those three leave the list because of large open POs (86k/50k/54k), and none flip when ATYALLOC is subtracted. The true effect is *magnitude* under-buys on PO-free, on-hand-bound items, not an item dropping today.
- **Concrete fix:** Change to `SUM(QTYONHND − ATYALLOC)` (confirm the GP misspelling `ATYALLOC`).

### 🟡 #10 — Synthetic-BOM reconstructor pollutes a packaging item with raw chemistry (MEDIUM)
*(2b-5 / "2b-4" — confirmed=false; verifier split the rating: latent by default, but the algorithm is genuinely wrong and CAN fire live)*

- **Symptom:** The single reachable surviving-raw synthetic recipe, `ZZLIDSVENTED` (a packaging lid, n_mos=1), is polluted with **48.25 LBS of humic acid (GRPHUMICLIQ) per lid** from one co-occurring batched MO. With future demand enabled, PHYLO02 (180 units) → 72 lids → **3,474 LBS of spurious humic-acid demand** that survives the raw filter (10x over-statement of GRPHUMICLIQ).
- **Root cause:** `reconstruct_synthetic_bom` treats every item on a producing MO as a component (`synthetic_bom.py` L97–114); a batched MO touched both lids and a humic-acid product, and with n_mos=1 there is no averaging. The underlying ratio math (`per_unit = total_comp / total_end`, L129) is also dimensionally fragile — it pools fill-MOs (ratio ≈0) with blend-MOs and is skewed by partial/unclosed MOs.
- **Why MEDIUM not HIGH:** By default (`include_future_demand=False`) PHYLO02 sits in the future bucket and never fires; even with future-ON, GRPHUMICLIQ on-hand (47,749 LBS) absorbs the demand and net_need = 0 today. **But see §6 — there is an unresolved conflict about whether it nets out.**
- **Concrete fix:** Exclude CONTAINERS/packaging-class items from being treated as synthesizable end-items; hard-gate `MIN_CONFIDENT_MOS` (drop n_mos < 3 recipes, don't buy from them); restrict synthetic components to mixing MOs (`X%` prefix / `WO010032` join) and exclude packaging classes; require a synthesized component to also appear in co-located IV30300 consumption.

### 🟡 #11 — Posted sales never feed the BOM explosion (MEDIUM, partial)
*(S1-01 — confirmed=false; verifier DOWN-weighted this: high→medium, "RM-drain is a defensible design choice")*

- **Symptom:** The order-driven explosion is **100% open-orders-only** (SOP10100/SOP10200). Posted sales (SOP30200/30300) are used only in an informational "Top Sellers" tab and never enter explosion quantities.
- **Why this is partly OK (the original "high" was overstated):** Historical demand *is* captured — **63% of buy-list net_need (1.22M of 1.94M units) comes from the kanban IV30300 raw-material-drain layer**, which reflects the RM consumption that past (sales-driven) production caused. Exploding historical FG sales through the BOM *and* keeping the kanban RM-drain would **double-count**. So using RM-drain instead of sales-explosion is a defensible design, not a wrong buy.
- **Concrete fix / decision:** Either add a forecast-demand source from SOP30200/30300 and feed it into the explosion alongside open orders (and remove the kanban RM-drain to avoid double-count), **or** document explicitly that the buy list is open-orders + kanban-refill, and stop implying in the UI that posted sales drive the buy.

### 🟡 #12 — Missing demand-query exclusions: void / hold / drop-ship / stale (MEDIUM, latent)
*(S1-03 — confirmed; zero offending rows today)*

- **Symptom:** The demand query has no guard against voided, on-hold (SOP10104), drop-ship, or stale orders, and no time-fence. **Zero offending rows in today's snapshot**, but each is a routine SOP state that would inject 100% false RM demand when it fires (a drop-ship FG ships vendor→customer and must never trigger a raw buy; a never-closed old order would explode forever).
- **Correction:** The originally-listed "fully-cancelled (QTYCANCE>0)" guard is **redundant** — a fully-cancelled line already has QTYREMAI=0 and is excluded; a partial cancel's remaining qty you correctly *want*.
- **Concrete fix:** `AND h.VOIDSTTS=0`; exclude `SOP10104` process holds; `AND d.DROPSHIP <> 1`; add a `REQSHIPDATE` lower bound (or explicit stale-order review).

### 🟡 #13 — No demand time-fence: all open qty summed into "buy now" regardless of horizon (LOW→MEDIUM trigger, latent)
*(S1-06 / "S1-05 aggregation" — confirmed; the cross-bucket folding is latent, but the no-time-fence behavior is active)*

- **Symptom:** `fetch_open_demand_prioritized` sums `total_qty` across **all** urgency buckets (past-due + due-today + future) per item, but labels the item with only the single highest-priority line's bucket. Future demand is **not** fenced out — today's run already folds +3-day future items (TRITOPS250=4,000, FLO11100=3,000, …) into the buy-now total. Harmless only because the farthest req date is +3 days, well inside lead time. The moment an item gets near *and* far orders straddling a bucket boundary with material qty outside RM lead time, it drives an early/oversized buy with a mislabeled urgency.
- **Concrete fix:** Bucket-aware aggregation or a lead-time time-fence; surface longer-horizon demand separately from the immediate buy.

### 🟢 #14 — Urgency label flips CRITICAL↔HIGH under a one-day clock shift (MEDIUM, dormant — does NOT change buy quantities)
*(S1-04 — confirmed)*

- **Symptom:** Urgency bucketing uses Python `date.today()`; if the app host's local date ever diverges from the SQL Server date, the CRITICAL/HIGH label flips by a day. **Today Python date == DB date (both 2026-06-05), so zero miscalculation.** The flip changes only the display label and triage sort order — **the buy list items and quantities are byte-identical** under a one-day shift (verified). It misleads *which order you work first*, not what you buy.
- **Concrete fix:** Anchor "today" to `CAST(GETDATE() AS date)` (the DB clock) instead of Python local date. This also closes the latent future/due-tomorrow inclusion-boundary risk.

### Findings the verifier REFUTED (down-weighted / dropped)

- **2b-2 ("98% of synthetic recipes lose 100% of chemical demand")** — **REFUTED** (broken→low). The alarming "all chemical raw-material demand is silently lost" headline is **false**: 170/177 zero-raw items are routed through real `BM010115` BOMs, not synthesis; the synthetic path produces essentially no live wrong buy. The *mechanism* (producing-MO scan sees only packaging) is real but its live impact was grossly overstated. Folded into #10 only.
- **BOM-4 / BOM-5 conclusions** — confirmed=false on *wording* only; verdicts stay **works**. Container/discontinued leaves are filtered correctly; UoM is internally consistent. No action.

---

## 4. Reconciliation scorecard — actual raw usage vs BOM-implied (12-mo)

Implied = Σ(FG units sold × app recursive-BOM design_qty, REC-normalized). Actual = |Σ IV30300 TRXQTY<0|. **Gap is driven almost entirely by the recipe-version-summing bug (#1/#2).**

| Raw material | Actual (LBS) | BOM-implied (LBS) | Gap (over-state) | Verdict |
|---|---:|---:|---:|---|
| **NO3MN** (Manganese Nitrate) | 396,278 | 9,272,542 | **23.4x** | 🔴 broken (#1+#2) |
| **CHEMONETH** (MEA) | 65,562 | 1,386,617 | **21.2x** | 🔴 broken (#1) |
| **SO4BORIC** (Boric Acid) | 260,913 | 4,533,675 | **17.4x** | 🔴 broken (#1) |
| **SO4FEDRY19** (Iron Sulfate) | 133,048 | 1,911,436 | **14.4x** | 🔴 broken (#1)¹ |
| **SO4MN32** (Manganese Sulfate) | 86,357 | 1,304,979 | **15.1x** | 🔴 broken (#1) |
| **SO4BORON** | 52,303 | 622,545 | **11.9x** | 🔴 broken (#1) |
| **NPKNPHURCDI** (control) | 192,738 | ~235,086 | **~1.22x** | 🟢 reconciles² |
| **SO4MN** (superseded) | ~0 | 0 | **0 ≈ 0** | 🟢 correct (dormant) |

¹ SO4FEDRY19 still carries **~2.5x** residual *after* the dedup fix — partly nested size-variants (#3) and possibly REC/dilution scope. Flagged in §6.
² NPKNPHURCDI reconciles in aggregate **only because its dominant seller FLO150016250 (single clean recipe) masks inflated siblings** (FLO15001602 = 5x, FLO15001630 = 2x). The family aggregate is safe; individual SKUs are not — see §6.

**Single-order proof of live impact (order 29138, TRITOPS250 × 4,000 gal):** engine emits SO4BORIC = **192,119 LBS** vs correct **~14,000 LBS** — the inflated demand on this one order ≈ **74% of a full year** of actual Boric Acid usage (260,913 LBS).

---

## 5. Must-fix list (ordered — do these BEFORE building any automation on the chain)

1. **Filter the BOM walk to the active recipe.** Add `AND BOMCAT_I = 1 AND RTRIM(BOMNAME_I) = ''` to **both** members of the recursive CTE in `fetch_recursive_bom_for_item`, the parent-exclusion subquery, **and** `fetch_mfg_bom_grouped_by_component`. Single biggest fix — kills the 14x–23x over-buy (#1) and most of #2/#3. *Validate:* TRITOPS00→SO4BORIC = 3.5119 LBS/gal; order 29138 → ~14,000 LBS.
2. **Treat RAWMAT* items as hard leaves inside the CTE** (don't drill REC-/dilution raws), via the `BOMCAT_I=1` JOIN fix above plus a class gate. Fixes #2. *Validate:* NO3MN appears as a leaf at ~13.12/unit, not REC-NO3MN at 38.63.
3. **Collapse same-liquid size-variant intermediates to one** (or model pack variants as bulk + packaging). Fixes #3. *Validate:* PMX1404-0106 → ~3,853 LBS, not 21,018.
4. **Unify `REC-` and base codes** end-to-end (on-hand + open-PO + usage) under the orderable base code; remove the dilution-proxy band-aid. Fixes #4 and #6. *Validate:* REC-NPKPHOS85 / REC-SO4FELIQ disappear from the buy list; REC-NO3CU under-buy of 17,406 closes.
5. **Gate the vendor buy list on a real PO-receipt history.** Suppress 0-lifetime-PO items (NPK3011, premixes); route in-house intermediates to production. Fixes #5.
6. **Add a missing-top-level-BOM report** for FGs on open orders, class-aware. Fixes #7 (PETRO1300). *Validate:* PETRO1300/29085 surfaces as "no BOM — manual planning."
7. **Filter the demand query to inventory FGs.** `ITEMTYPE IN (1)`; map MISCCHG `ITEMDESC` → real `ITEMNMBR`. Fixes #8.
8. **Subtract allocations:** `SUM(QTYONHND − ATYALLOC)` in `fetch_on_hand_by_item`. Fixes #9.
9. **Harden the demand query:** `VOIDSTTS=0`, exclude SOP10104 holds, `DROPSHIP<>1`, add a ReqShipDate lower bound; anchor "today" to `CAST(GETDATE() AS date)`; add a lead-time time-fence. Fixes #12/#13/#14.
10. **Repair / gate the synthetic-BOM path** (exclude packaging end-items, hard-gate n_mos, restrict to mixing MOs). Fixes #10.
11. **Add a mass-balance regression test:** Σ leaf RAW LBS per parent gallon must be within ~10% of the parent's lbs/gal from `UOMSCHDL`. Cleanly catches any future version-summing regression.

**Do NOT do these (verified wrong fix directions):**
- Do **not** add a density divisor — UoM is already correct.
- Do **not** use raw `MOP1016`/`SUM(QTYRECVD)` as the recipe source — it double-counts PO receipts and WIP-mirror rows (this is the source of the bogus "7.024 LBS/gal").
- Do **not** "de-duplicate via DISTINCT" or pick "latest by effective date" — rows have different quantities and all effective dates are 1900-01-01; you must *select the active category*, not dedup.
- Do **not** route NPK3011 (no BOM) into the synthetic path — it produces 32 components of false demand.

---

## 6. Couldn't fully settle (needs manual confirmation)

1. **GRPHUMICLIQ — does the synthetic-lid pollution reach the buy list?** Two verified findings conflict. One concludes the 3,474 LBS spurious humic-acid demand **nets to 0** (absorbed by 47,749 LBS on-hand). The other concludes the synthetic path contributes **~90% of GRPHUMICLIQ's live buy quantity**. Both ran the engine; the difference is likely run-mode (`include_future_demand` on/off) and on-hand draw-down state. **Confirm by running `build_integrated_reorder_list` in both modes and inspecting GRPHUMICLIQ net_need** before relying on either number.
2. **SO4FEDRY19 residual ~2.5x after the dedup fix.** The `BOMCAT_I=1` filter brings the other materials to ~0.9–1.2x but SO4FEDRY19 stays ~2.5x — attributable to nested size-variants (#3), REC/dilution scope, or synthetic contributions. **Re-validate after fixes #1 and #3 land** to confirm it lands near 1.0x.
3. **"Report said TRITOPS250 due today."** The DB fact is solid — TRITOPS250 (order 29138) has `REQSHIPDATE = 2026-06-08` (future), not due today. But the claim that the *daily report artifact* listed it "due today" rests on a document the verifier never saw. **Confirm against the actual report you were holding** if the staleness matters.
4. **Latent synthetic under-buy via stale MO lookback.** USCMANPLX00's all-time synthetic recipe contains real RAWMAT components (SO4FELIQ, SO4MG, CHECITRIC) that the 12-month lookback would zero out; today this is cushioned because those items already appear as direct leaves, but true net lost-demand is **uncertain, not provably zero**. Confirm if any manufactured intermediate with stale MO history ever becomes a live leaf without a formal BOM.
