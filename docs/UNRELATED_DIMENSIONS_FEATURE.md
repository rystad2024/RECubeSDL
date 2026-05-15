# Unrelated Dimensions Handling - Feature Documentation

**Author:** Usman Yasin
**Branch:** `feat/unrelated-dimensions-handling`
**Date:** October 2025 (commits) + ongoing uncommitted work
**Purpose:** This document is intended for an agent/developer who needs to reimplement this behavior in a newer version of the codebase that uses **query folding** technique.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Commit-by-Commit Breakdown](#3-commit-by-commit-breakdown)
4. [Current Working Directory Changes (Uncommitted)](#4-current-working-directory-changes-uncommitted)
5. [Key Files and Their Roles](#5-key-files-and-their-roles)
6. [Algorithm / Control Flow](#6-algorithm--control-flow)
7. [SQL Generation Examples](#7-sql-generation-examples)
8. [Important Helper Functions](#8-important-helper-functions)
9. [Side Fixes Bundled in This Branch](#9-side-fixes-bundled-in-this-branch)
10. [Differences for the New (Query Folding) Implementation](#10-differences-for-the-new-query-folding-implementation)

---

## 1. Problem Statement

In Cube's Tesseract SQL planner, when a query references **dimensions from multiple cubes that have no direct relationship** (no foreign key / join path), the join condition between them is modeled as `1 = 1` (a constant true condition). This results in a **Cartesian product (CROSS JOIN)**, which:

- Produces an exponentially large intermediate result set
- Causes incorrect aggregation (measures get multiplied)
- Leads to performance degradation and potential OOM errors

**Example scenario:**
A query asks for `CubeA.revenue` (measure), `CubeA.country` (dimension), and `CubeB.product_category` (dimension), where `CubeA` and `CubeB` have no defined join relationship.

Without this feature, the planner would generate:
```sql
SELECT ...
FROM CubeA
LEFT JOIN CubeB ON 1 = 1   -- Cartesian product!
```

---

## 2. Solution Overview

The solution has **three layers**:

### Layer 1: Detection (`sql_call.rs`)
Detect when a join condition is the constant expression `1 = 1`, meaning the cubes are unrelated.

### Layer 2: Subquery Substitution (`builder.rs` - `process_logical_join`)
Instead of joining the unrelated cube directly (causing a cross join), replace it with a **`SELECT DISTINCT` subquery** containing only the dimensions from that cube that are actually referenced in the query's SELECT list. This avoids the Cartesian product.

### Layer 3: Filter Relocation (`builder.rs` - filter management)
Any WHERE filters that reference dimensions from the unrelated cube are:
- **Moved into** the DISTINCT subquery (so they filter early)
- **Removed from** the outer query (to avoid referencing columns that no longer exist at the outer level)

### Layer 4 (Uncommitted): Keys Pattern Elimination (`builder.rs` - `process_aggregate_multiplied_subquery`)
For multiplied measure queries, **eliminate the keys subquery self-join pattern entirely**. Instead of wrapping dimensions in a separate `keys` subquery and then joining it back to the cube, build the FROM clause directly with all dimensions and measures together, using GROUP BY instead of DISTINCT.

---

## 3. Commit-by-Commit Breakdown

### Commit `088e6088c` — "feat: unrelated dimensions draft implementation as cross joins"

**Files changed:**
- `rust/cubesqlplanner/cubesqlplanner/src/planner/sql_evaluator/sql_call.rs`
- `rust/cubesqlplanner/cubesqlplanner/src/physical_plan_builder/builder.rs`

**What it does:**

#### `sql_call.rs` — Added `is_constant_one_equals_one()` method

```rust
pub fn is_constant_one_equals_one(&self, base_tools: Rc<dyn BaseTools>) -> Result<bool, CubeError>
```

This method on `SqlCall`:
1. Checks if the SQL expression has **no member symbol dependencies** (i.e., it doesn't reference any cube columns)
2. Evaluates the SQL string
3. Strips whitespace and checks if the result is literally `"1=1"`
4. Returns `true` if this is a constant `1 = 1` join condition

#### `builder.rs` — Modified `process_logical_join()`

Added a new parameter `schema: &LogicalSchema` to `process_logical_join()` so it knows which dimensions are actually in the query's SELECT list.

**Logic added inside the `CubeJoinItem` match arm:**

```
IF on_sql.is_constant_one_equals_one():
    1. Find which dimensions from this cube are in the query schema (SELECT list)
    2. Create a SELECT DISTINCT subquery with only those dimensions
    3. Set up render_references so the outer query references the subquery alias
    4. LEFT JOIN the subquery instead of the raw cube
ELSE:
    (original behavior — LEFT JOIN the cube directly)
```

#### `builder.rs` — Added `create_distinct_dimensions_subquery()`

```rust
fn create_distinct_dimensions_subquery(
    &self,
    cube: &Rc<BaseCube>,
    cube_alias: &str,
    dimensions: &Vec<Rc<MemberSymbol>>,
    context: &PhysicalPlanBuilderContext,
) -> Result<Rc<Select>, CubeError>
```

Creates:
```sql
SELECT DISTINCT dim1, dim2, ...
FROM unrelated_cube AS alias
```

---

### Commit `cbba7bc8a` — "fix: removed unrelated dimension and filters in case not referenced in select"

**Files changed:**
- `rust/cubesqlplanner/cubesqlplanner/src/physical_plan_builder/builder.rs`

**What it does:**

#### 1. Skip join entirely if no dimensions referenced

If a `1 = 1` cube has **zero dimensions** in the SELECT list, the join is **skipped entirely**. No subquery is generated, no join is added. This is a major optimization — if you're only filtering on an unrelated cube but not selecting its dimensions, there's no need to join it at all.

#### 2. Filter relocation into subqueries

**New flow in `process_logical_join()`:**
- The function signature now accepts `filter: Option<Filter>` and returns `(Rc<From>, Vec<FilterItem>)` (a tuple with the FROM clause AND a list of "applied" filter items)
- For each `1 = 1` join, it calls `extract_cube_filter_items()` to find filters belonging to that cube
- Those filter items are passed into `create_distinct_dimensions_subquery()` which applies them as a WHERE clause inside the subquery
- The applied filter items are tracked and returned to the caller

**New flow in `build_simple_query()`:**
- After `process_logical_join()` returns, calls `remove_filter_items()` to strip the already-applied filters from the outer query's WHERE clause

#### 3. New helper: `extract_cube_filter_items()`

```rust
fn extract_cube_filter_items(&self, filter_items: &[FilterItem], cube_name: &str) -> Vec<FilterItem>
```

Iterates filter items and returns those whose member evaluators belong to the specified cube.

#### 4. New helper: `remove_filter_items()`

```rust
fn remove_filter_items(&self, filter: Option<Filter>, items_to_remove: &[FilterItem]) -> Option<Filter>
```

Removes specified filter items from a Filter. Returns `None` if all items are removed.

#### 5. Updated `create_distinct_dimensions_subquery()` signature

Now accepts `filter_items: Vec<FilterItem>` and applies them inside the subquery:

```rust
if !filter_items.is_empty() {
    select_builder.set_filter(Some(Filter { items: filter_items }));
}
```

---

### Commit `3fa032fdf` — "fix: duplicate measures - ambiguous column error"

**File changed:**
- `rust/cubesqlplanner/cubesqlplanner/src/planner/query_properties.rs`

**What it does:**

Deduplicates measures in `QueryProperties` to prevent the same measure from appearing multiple times (which caused ambiguous column errors in SQL). Uses `unique_by(|m| m.full_name())` on:
- `regular_measures`
- `multiplied_measures`
- `multi_stage_measures`

This is a companion fix needed when unrelated dimensions cause measures to be resolved multiple times through different paths.

---

## 4. Current Working Directory Changes (Uncommitted)

### 4.1 `builder.rs` — Keys Pattern Elimination (MAJOR CHANGE)

**`process_aggregate_multiplied_subquery()`** has been **completely rewritten**.

#### Before (old pattern):
```
1. Build keys_query (SELECT DISTINCT dimensions FROM cube)
2. Create JoinBuilder starting from keys_query
3. LEFT JOIN the cube back to keys_query on primary key dimensions
4. This creates a self-join: keys → cube
```

The old pattern generated SQL like:
```sql
SELECT dims..., agg(measures...)
FROM (
    SELECT DISTINCT pk_dim1, pk_dim2 FROM cube   -- keys subquery
) AS keys
LEFT JOIN cube AS cube_alias
    ON keys.pk_dim1 = cube_alias.pk_dim1
    AND keys.pk_dim2 = cube_alias.pk_dim2
GROUP BY dims...
```

#### After (new pattern — no self-join):
```
1. Build FROM directly using process_logical_join() with combined schema
   (dimensions + measures together)
2. Use GROUP BY instead of the keys-subquery + DISTINCT pattern
3. Apply filters directly, removing those handled in 1=1 subqueries
```

The new pattern generates SQL like:
```sql
SELECT dims..., agg(measures...)
FROM cube AS cube_alias
LEFT JOIN (SELECT DISTINCT unrelated_dim FROM unrelated_cube WHERE ...) AS unrelated_alias ON 1=1
GROUP BY dims...
```

**Key changes:**
- For `AggregateMultipliedSubquerySouce::Cube`: Builds FROM directly from `keys_subquery.source` with a combined `LogicalSchema` that includes both dimensions and measures
- For `AggregateMultipliedSubquerySouce::MeasureSubquery`: Same approach — uses `keys_subquery.source` instead of building a separate measure subquery, includes query dimensions
- Both paths use `process_logical_join()` which handles the `1=1` detection and subquery substitution
- Filter management: calls `remove_filter_items()` on the remaining filters after `process_logical_join()` handles some
- `process_measure_subquery()` and `process_keys_sub_query()` are now marked `#[allow(dead_code)]` (still present but unused by the new path)

#### Updated `process_keys_sub_query()` (still exists, also improved):
- Now passes a proper `keys_schema` (with actual dimensions) instead of an empty schema to `process_logical_join()`
- Now passes `keys_subquery.filter.all_filters()` instead of `None`
- Applies `remove_filter_items()` to avoid double-filtering

### 4.2 `base_query.rs` and `node_export.rs` — Performance Instrumentation

Extensive `eprintln!("[PERF] ...")` logging added throughout the pipeline:
- `BaseQuery::try_new` → tracks `QueryTools::try_new` and `QueryProperties::try_new`
- `build_sql_and_params_impl` → tracks `QueryPlanner::plan`, `try_pre_aggregations`, `physical_plan_builder.build`, `to_sql`, `build_sql_and_params`
- `build_sql_and_params` (native Neon entry) → tracks argument parsing, `from_native`, `try_new`, `build_sql_and_params`

> **Note for new implementation:** These are debug/profiling logs and should NOT be carried over. They are temporary instrumentation.

### 4.3 `query_result_transform.rs`, `transport.rs`, `query_message_parser.rs` — HashMap → IndexMap

**Purpose:** Ensure deterministic column ordering in query results.

**Changes:**
- `MembersMap` type changed from `HashMap<String, String>` to `IndexMap<String, String>`
- `JsRawData` type changed from `Vec<HashMap<...>>` to `Vec<IndexMap<...>>`
- `columns_pos` changed from `HashMap<String, usize>` to `IndexMap<String, usize>`
- `get_compact_row()` and `get_vanilla_row()` parameter types updated
- `Cargo.toml` added `indexmap = { version = "2.0", features = ["serde"] }`

> **Note for new implementation:** This is a separate concern (column ordering) and may or may not be needed depending on the new version's result handling.

### 4.4 `PreAggregations.ts` — Null guard

Added a null check for `join` in `preAggregationCubes()`:
```typescript
if (!join) {
    return [];
}
```
This prevents crashes when the Tesseract engine doesn't produce a join object.

---

## 5. Key Files and Their Roles

| File | Role |
|------|------|
| `rust/cubesqlplanner/cubesqlplanner/src/physical_plan_builder/builder.rs` | **Core file.** Translates logical plans into physical SQL plans. Contains all the unrelated dimensions logic. |
| `rust/cubesqlplanner/cubesqlplanner/src/planner/sql_evaluator/sql_call.rs` | Provides `is_constant_one_equals_one()` detection on join conditions. |
| `rust/cubesqlplanner/cubesqlplanner/src/planner/query_properties.rs` | Measure deduplication fix. |
| `rust/cubesqlplanner/cubesqlplanner/src/planner/base_query.rs` | Entry point for SQL generation pipeline. |
| `packages/cubejs-backend-native/src/node_export.rs` | Neon (Node.js ↔ Rust) bridge for `buildSqlAndParams`. |

---

## 6. Algorithm / Control Flow

### Main Query Path (`build_simple_query`)

```
build_simple_query(logical_plan)
│
├── Get filter from logical_plan.filter.all_filters()
│
├── process_logical_join(join, schema, filter)
│   │
│   ├── For each CubeJoinItem in join.joins:
│   │   │
│   │   ├── IF on_sql.is_constant_one_equals_one():
│   │   │   │
│   │   │   ├── Find cube_dimensions in schema (SELECT list)
│   │   │   ├── Extract cube_filter_items from filter
│   │   │   ├── Track as applied_filter_items
│   │   │   │
│   │   │   ├── IF cube_dimensions is NOT empty:
│   │   │   │   ├── create_distinct_dimensions_subquery(cube, dims, filters)
│   │   │   │   ├── Set up render_references for subquery alias
│   │   │   │   └── LEFT JOIN subquery ON 1=1
│   │   │   │
│   │   │   └── IF cube_dimensions IS empty:
│   │   │       └── Skip join entirely (no-op)
│   │   │
│   │   └── ELSE (normal join):
│   │       └── LEFT JOIN cube ON condition (original behavior)
│   │
│   └── Return (From, applied_filter_items)
│
├── remove_filter_items(outer_filter, applied_filter_items)
│   └── Strip filters already handled in subqueries
│
└── Build SELECT with remaining filters, projections, GROUP BY, etc.
```

### Multiplied Measures Path (`process_aggregate_multiplied_subquery`) — NEW

```
process_aggregate_multiplied_subquery(agg_subquery)
│
├── Build combined schema (keys dimensions + measures)
│
├── process_logical_join(source, combined_schema, filter)
│   └── (same 1=1 detection and subquery substitution as above)
│
├── Build SELECT with:
│   ├── Dimensions as projections + GROUP BY
│   ├── Measures as aggregated projections
│   └── Remaining filters (after remove_filter_items)
│
└── Return Select (no keys wrapper, no self-join)
```

---

## 7. SQL Generation Examples

### Example 1: Simple query with unrelated dimension

**Query:** `SELECT CubeA.country, CubeB.category, SUM(CubeA.revenue)`

**Before (Cartesian product):**
```sql
SELECT a.country, b.category, SUM(a.revenue)
FROM cube_a AS a
LEFT JOIN cube_b AS b ON 1 = 1
GROUP BY 1, 2
```

**After (with this feature):**
```sql
SELECT a.country, cube_b_subq.category, SUM(a.revenue)
FROM cube_a AS a
LEFT JOIN (
    SELECT DISTINCT b.category
    FROM cube_b AS b
) AS cube_b_subq ON 1 = 1
GROUP BY 1, 2
```

### Example 2: With filter on unrelated dimension

**Query:** `SELECT CubeA.country, SUM(CubeA.revenue) WHERE CubeB.category = 'Electronics'`

**After (filter pushed into subquery, dimension not in SELECT so join skipped):**
```sql
-- The CubeB join is SKIPPED entirely because:
-- 1. CubeB.category is NOT in the SELECT list
-- 2. The filter on CubeB.category is marked as "applied" and removed from outer WHERE
-- (The filter was logically moved into a subquery that was then eliminated)
SELECT a.country, SUM(a.revenue)
FROM cube_a AS a
GROUP BY 1
```

### Example 3: Multiplied measures (NEW — uncommitted)

**Before (keys pattern):**
```sql
SELECT keys.dim1, agg(cube_alias.measure1)
FROM (
    SELECT DISTINCT dim1 FROM cube
) AS keys
LEFT JOIN cube AS cube_alias ON keys.dim1 = cube_alias.dim1
GROUP BY 1
```

**After (direct, no self-join):**
```sql
SELECT cube_alias.dim1, agg(cube_alias.measure1)
FROM cube AS cube_alias
LEFT JOIN (SELECT DISTINCT unrelated_dim FROM unrelated_cube) AS unrelated_alias ON 1 = 1
GROUP BY 1
```

---

## 8. Important Helper Functions

### `is_constant_one_equals_one()` — `sql_call.rs`
- **Input:** `base_tools: Rc<dyn BaseTools>`
- **Returns:** `Result<bool, CubeError>`
- **Logic:** No member dependencies + evaluated SQL trims to `"1=1"`

### `create_distinct_dimensions_subquery()` — `builder.rs`
- **Input:** cube, alias, dimensions, context, filter_items
- **Returns:** `Rc<Select>` — a `SELECT DISTINCT dim1, dim2 FROM cube WHERE filters`

### `extract_cube_filter_items()` — `builder.rs`
- **Input:** filter_items slice, cube_name
- **Returns:** `Vec<FilterItem>` — filters whose members belong to the cube

### `remove_filter_items()` — `builder.rs`
- **Input:** filter, items_to_remove
- **Returns:** `Option<Filter>` — filter with specified items removed (by member name matching in uncommitted version)
- **Note:** The uncommitted version uses `HashSet<String>` of member full names for O(1) lookup instead of the original `contains()` comparison

---

## 10. Differences for the New (Query Folding) Implementation

The new version uses **query folding**, which means the SQL generation approach is fundamentally different. Here are the key concepts that need to be adapted:

### What Must Be Preserved (Behavioral Requirements)

1. **Detection of unrelated cubes:** When two cubes have no join path, the join condition is `1 = 1`. This must be detected.

2. **DISTINCT subquery substitution:** Instead of cross-joining the raw table, wrap the unrelated cube's dimensions in a `SELECT DISTINCT` subquery.

3. **Filter relocation:** Filters on unrelated dimensions must be pushed into the DISTINCT subquery, not applied at the outer level.

4. **Join elimination:** If no dimensions from an unrelated cube appear in the SELECT, skip the join entirely and also remove any filters on that cube (since they were intended for the subquery that was eliminated).

5. **Measure deduplication:** Ensure measures aren't duplicated when resolved through multiple paths.

### What Will Be Different (Query Folding Context)

1. **No `PhysicalPlanBuilder`:** Query folding builds SQL differently — the concept of `process_logical_join` may not exist. The equivalent logic needs to be placed wherever join SQL is emitted.

2. **No `LogicalSchema` to check SELECT membership:** In query folding, you'll need an equivalent way to know which dimensions from an unrelated cube are actually needed in the final output.

3. **No `keys` subquery pattern:** The current version already eliminates the keys pattern (uncommitted). In query folding, multiplied measures are likely handled differently — but the core principle remains: don't self-join when you can GROUP BY directly.

4. **Filter objects may differ:** The `FilterItem` / `Filter` types and their `all_member_evaluators()` method are specific to this codebase. The new version needs equivalent filter introspection.

5. **The `1 = 1` detection** may need to happen at a different layer — possibly during logical plan construction rather than physical plan building, depending on query folding's architecture.

### Recommended Approach for Query Folding

1. **During fold/merge phase:** When folding queries from multiple cubes, detect if the cubes are unrelated (no join path → `1 = 1` condition).

2. **Emit a CTE or inline subquery** for each unrelated cube's dimensions:
   ```sql
   WITH unrelated_dims AS (
       SELECT DISTINCT dim1, dim2
       FROM unrelated_cube
       WHERE <filters on this cube>
   )
   SELECT main.*, unrelated_dims.*
   FROM main_folded_query
   CROSS JOIN unrelated_dims
   ```

3. **Strip filters** from the main query that were pushed into the CTE.

4. **Handle the empty-dimensions case** by omitting the CTE/join entirely.

---

## Appendix: File Diff Summary

| File | Commit | Change Type |
|------|--------|-------------|
| `sql_call.rs` | `088e6088c` | Added `is_constant_one_equals_one()` |
| `builder.rs` | `088e6088c` | Added 1=1 detection + DISTINCT subquery in `process_logical_join` |
| `builder.rs` | `cbba7bc8a` | Added filter relocation + join elimination for empty dims |
| `query_properties.rs` | `3fa032fdf` | Measure deduplication |
| `builder.rs` | uncommitted | Keys pattern elimination, perf logging, improved filter removal |
| `base_query.rs` | uncommitted | Perf logging only |
| `node_export.rs` | uncommitted | Perf logging only |
| `PreAggregations.ts` | uncommitted | Null guard for join |
| `query_result_transform.rs` | uncommitted | HashMap → IndexMap (column ordering) |
| `transport.rs` | uncommitted | HashMap → IndexMap (type aliases) |
| `query_message_parser.rs` | uncommitted | HashMap → IndexMap (column positions) |
| `Cargo.toml` | uncommitted | Added indexmap dependency |
