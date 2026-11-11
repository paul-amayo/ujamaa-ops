#!/usr/bin/env python3
"""azalai_h1_tools.py (11 Nov 2026) — lookup tools for the H1 pilot v1. Both arms get the SAME five generic tools over their own
registry (the ICRA 2026 architecture: the LLM decomposes the instruction, the graph resolves it); the only difference between the
arms is the set of fields a record carries:

  enumerated  bag: id, row, n_in_row, x, z                                   tree: id
  twin        bag: + height_dec_m, dh_apr_m, growth, status                  tree: + fruit_count

The tools are lookups (rows, a plant by id, a plant by row + order, filter / sort / count, nearest by position); none of them
encodes an instruction category. No free text goes through an embedding: every target is resolved to plant ids here (HiGH
supplies node addresses only, 11 Nov). Design of record: Azalai_RSS/numbers/pilots_20261111/H1_v1_design.md."""
import json, math

FIELDS = {'enumerated': {'bag': ['id', 'row', 'n_in_row', 'x', 'z'], 'tree': ['id']},
          'twin': {'bag': ['id', 'row', 'n_in_row', 'x', 'z', 'height_dec_m', 'dh_apr_m', 'growth', 'status'], 'tree': ['id', 'fruit_count']}}
OPS = ('=', '!=', '<', '<=', '>', '>=', 'in')
SHOW = 10          # records shown to the model per result; the whole set stays behind the result_id


class ToolError(Exception):
    pass


def _num(v):
    if isinstance(v, bool): raise ToolError('a number is needed, got a boolean')
    if isinstance(v, (int, float)): return v
    try: return int(str(v).strip())
    except Exception:
        try: return float(str(v).strip())
        except Exception: raise ToolError(f'a number is needed, got {v!r}')


class Registry:
    """One arm's registry with its tools. results[result_id] = the ordered ids of a find / nearest result."""

    def __init__(self, R, arm):
        assert arm in FIELDS
        self.arm = arm; self.fields = FIELDS[arm]; self.n_rows = R['rows']
        self.bags = sorted([b for b in R['bags'] if isinstance(b['id'], int)], key=lambda b: (b['row'], b['n_in_row']))   # the December census: the plants that have an address
        # bags first seen in April are change records (no order in a row, no December node); they exist in the twin arm only
        self.added = [b for b in R['bags'] if not isinstance(b['id'], int)] if arm == 'twin' else []
        fn = {k: v for k, v in R['fruit']['fruitnerf']['ours'].items() if isinstance(v, (int, float))}    # the citrus trees of v0 (FruitNeRF trees 1-3)
        self.trees = [dict(id=k, fruit_count=v) for k, v in sorted(fn.items())]
        self.byid = {str(b['id']): ('bag', b) for b in self.bags + self.added}; self.byid.update({t['id']: ('tree', t) for t in self.trees})
        self.results = {}; self._q = 0

    # ---- helpers
    def view(self, kind, rec, extra=None):
        d = {f: rec.get(f) for f in self.fields[kind]}
        if extra: d.update(extra)
        return d

    def _row(self, row):
        r = _num(row)
        if r != int(r) or not 1 <= int(r) <= self.n_rows: raise ToolError(f'no row {row}; rows are 1..{self.n_rows}')
        return [b for b in self.bags if b['row'] == int(r)]

    def _store(self, kind, recs, matched=None, extra=None):
        self._q += 1; rid = f'q{self._q}'; self.results[rid] = [r['id'] for r in recs]
        out = dict(result_id=rid, count=len(recs))
        if matched is not None and matched != len(recs): out['matched_before_limit'] = matched
        out['plants'] = [self.view(kind, r, (extra or {}).get(r['id'])) for r in recs[:SHOW]]
        if len(recs) > SHOW: out['note'] = f'showing the first {SHOW} of {len(recs)}; the result_id stands for all {len(recs)}, in this order'
        return out

    @staticmethod
    def _match(v, op, val):
        if val is None:
            if op == '=': return v is None
            if op == '!=': return v is not None
            raise ToolError('null can only be compared with = or !=')
        if v is None: return False
        if op == 'in':
            if not isinstance(val, list): raise ToolError("op 'in' needs a list value")
            return any(Registry._match(v, '=', x) for x in val)
        if isinstance(v, str):
            if op not in ('=', '!='): raise ToolError('a text field can only be compared with =, != or in')
            eq = v.strip().lower() == str(val).strip().lower(); return eq if op == '=' else not eq
        val = _num(val)
        return {'=': v == val, '!=': v != val, '<': v < val, '<=': v <= val, '>': v > val, '>=': v >= val}[op]

    # ---- the five tools
    def list_rows(self):
        rows = []
        for r in range(1, self.n_rows + 1):
            L = self._row(r); rows.append(dict(row=r, n_bags=len(L), x=round(sum(b['x'] for b in L) / len(L), 2), z_first=L[0]['z'], z_last=L[-1]['z']))
        return dict(n_rows=self.n_rows, n_bags=len(self.bags), n_trees=len(self.trees), rows=rows)

    def get_plant(self, id):
        k = str(id).strip()
        if k not in self.byid: raise ToolError(f'no plant with id {id}')
        kind, rec = self.byid[k]; return self.view(kind, rec)

    def get_plant_at(self, row, n):
        L = self._row(row); n = _num(n)
        if n != int(n) or n == 0 or abs(int(n)) > len(L): raise ToolError(f'row {int(_num(row))} has bags 1..{len(L)}; no bag {n}')
        n = int(n); return self.view('bag', L[n - 1] if n > 0 else L[n])

    def find_plants(self, kind='bag', where=None, sort_by=None, descending=False, limit=None):
        if kind not in self.fields: raise ToolError(f"kind must be 'bag' or 'tree', got {kind!r}")
        F = self.fields[kind]; where = where or []
        if isinstance(where, dict): where = [where]
        if not isinstance(where, list): raise ToolError('where must be a list of {field, op, value}')
        for c in where:
            if not isinstance(c, dict) or 'field' not in c or 'op' not in c or 'value' not in c: raise ToolError('each where condition needs field, op and value')
            if c['field'] not in F: raise ToolError(f"unknown field {c['field']!r} for a {kind}; a {kind} has only {F}")
            if c['op'] not in OPS: raise ToolError(f"unknown op {c['op']!r}; ops are {list(OPS)}")
        if sort_by is not None and sort_by not in F: raise ToolError(f"unknown field {sort_by!r} for a {kind}; a {kind} has only {F}")
        pool = list(self.trees) if kind == 'tree' else list(self.bags)
        if kind == 'bag' and any(c['field'] == 'status' for c in where): pool += self.added      # change records answer only a question about status
        recs = [r for r in pool if all(self._match(r.get(c['field']), c['op'], c['value']) for c in where)]
        if sort_by is not None:
            recs = sorted([r for r in recs if r.get(sort_by) is not None], key=lambda r: r[sort_by], reverse=bool(descending))   # a plant with no value in the sort field is left out
        matched = len(recs)
        if limit is not None:
            k = _num(limit)
            if k != int(k) or k < 1: raise ToolError('limit must be a positive integer')
            recs = recs[:int(k)]
        return self._store(kind, recs, matched)

    def nearest_plants(self, x, z, row=None, limit=1):
        x, z = float(_num(x)), float(_num(z)); pool = self._row(row) if row is not None else self.bags
        k = _num(limit)
        if k != int(k) or k < 1: raise ToolError('limit must be a positive integer')
        d = {b['id']: math.hypot(b['x'] - x, b['z'] - z) for b in pool}
        recs = sorted(pool, key=lambda b: d[b['id']])[:int(k)]
        return self._store('bag', recs, None, {i: dict(distance_m=round(v, 2)) for i, v in d.items()})

    TOOLS = ('list_rows', 'get_plant', 'get_plant_at', 'find_plants', 'nearest_plants')

    def call(self, name, args):
        if name not in self.TOOLS: return dict(error=f'unknown tool {name!r}; tools are {list(self.TOOLS)}')
        if not isinstance(args, dict): return dict(error='args must be a JSON object')
        try: return getattr(self, name)(**args)
        except ToolError as e: return dict(error=str(e))
        except TypeError as e: return dict(error=f'bad arguments for {name}: {e}')

    def expand(self, targets):
        """final targets -> ids: a result_id stands for its whole result, in order."""
        out = []
        for t in (targets if isinstance(targets, list) else [targets]):
            if isinstance(t, str) and t.strip() in self.results: out.extend(self.results[t.strip()])
            elif t is not None: out.append(t)
        return out


# ---- the text the model sees (identical in the two arms except the field lists and the description of what a record holds)
DESCRIBE = {
 'enumerated': ("The registry is an enumerated scene graph. Site 1, a blueberry nursery: rows 1..{rows} (numbered by increasing x) of grow bags, one plant per bag. "
                "A bag has: id (a unique integer, the census index; it says nothing about where the bag stands), row, n_in_row (its order along the row; 1 is the first bag), "
                "x and z (position in metres; z increases along the row, x across the rows, larger x is the right-hand side). "
                "Site 2, a citrus orchard: trees, each with an id (tree_01, ...). Nothing else is known about a bag or a tree."),
 'twin': ("The registry is a hierarchical digital twin's registry. Site 1, a blueberry nursery: rows 1..{rows} (numbered by increasing x) of grow bags, one plant per bag. "
          "A bag has: id (a unique integer, the census index; it says nothing about where the bag stands), row, n_in_row (its order along the row; 1 is the first bag), "
          "x and z (position in metres; z increases along the row, x across the rows, larger x is the right-hand side), "
          "height_dec_m (December plant height, metres), dh_apr_m (April height minus December height, metres), growth (the April call: 'grown', 'stable', 'shrunk', 'split' or null), "
          "status ('found again' or 'association miss': present in April; 'removed': gone in April; 'added': new in April, not there in December; 'unclear'; 'december false positive'). "
          "Site 2, a citrus orchard: trees, each with an id (tree_01, ...) and fruit_count (the number of fruit on the tree). Nothing else is known about a bag or a tree.")}

TOOLTEXT = """TOOLS
1. list_rows  args {{}}  -> the rows of the nursery: for each row its number, n_bags, mean x, and the z of its first and last bag; also n_rows, n_bags, n_trees.
2. get_plant  args {{"id": <id>}}  -> the record of the bag or tree with that id.
3. get_plant_at  args {{"row": <row>, "n": <n>}}  -> the record of the n-th bag along a row (n = 1 is the first; n = -1 is the last, -2 the one before it).
4. find_plants  args {{"kind": "bag" or "tree", "where": [{{"field": <field>, "op": <op>, "value": <value>}}, ...], "sort_by": <field>, "descending": true/false, "limit": <k>}}  -> the plants that meet every condition, as a result_id, their count and the first {show} records. Every argument is optional (kind defaults to "bag"). ops: = != < <= > >= in ("in" takes a list; value null with = or != tests for a missing value). Without sort_by, bags come in row and n_in_row order; with sort_by, plants with no value in that field are left out. Fields of a bag: {bag_fields}. Fields of a tree: {tree_fields}.{added_note}
5. nearest_plants  args {{"x": <x>, "z": <z>, "row": <row, optional>, "limit": <k, default 1>}}  -> the bags nearest to a position (in one row if row is given), nearest first, as a result_id with distance_m."""

ADDED_NOTE = {'enumerated': '', 'twin': " Bags with status 'added' were first seen in April: they have no n_in_row and are returned only when a condition is on status."}

PROTOCOL = """You ground a farm manager's instruction for a field robot against the farm's registry. You do not hold the registry: you break the instruction into lookups, call the tools, and the registry resolves them. The robot can only drive to plants and inspect them.

{describe}

Addresses: an instruction that gives a row and a number ("plant 4 in row 4"; "row 5, bag 30") names the n-th bag along that row; a bag number with no row ("bag 412") is an id.

{tools}

HOW TO REPLY
Reply with exactly one JSON object per turn and nothing else.
To call a tool: {{"tool": "<name>", "args": {{...}}}}. The result comes back in the next message. You have at most {max_calls} tool calls.
When the instruction is resolved: {{"final": {{"targets": [...], "answer": <number or null>, "reject": <true/false>, "reason": "<at most twenty words>"}}}}
- targets: the bag ids (or tree ids) in the order they should be visited. A result_id (for example "q2") in targets stands for every plant of that result, in its order; use it instead of copying a list.
- answer: if the instruction asks for a number, the number; otherwise null.
- reject: true, with empty targets, if the instruction cannot be carried out from this registry: no such row, plant or thing; an action the robot cannot do; another site; a property or a history the registry does not hold; a future state; more than one robot; a landmark; a previous session.
Use the tools for every lookup; never guess an id, an order or a count.

EXAMPLES
INSTRUCTION: Go to the ninth bag of row 11.
{{"tool": "get_plant_at", "args": {{"row": 11, "n": 9}}}}
TOOL RESULT: {ex1}
{{"final": {{"targets": [{ex1_id}], "answer": null, "reject": false, "reason": "ninth bag along row 11"}}}}

INSTRUCTION: Inspect the first ten bags of row 11.
{{"tool": "find_plants", "args": {{"kind": "bag", "where": [{{"field": "row", "op": "=", "value": 11}}, {{"field": "n_in_row", "op": "<=", "value": 10}}], "sort_by": "n_in_row"}}}}
TOOL RESULT: {{"result_id": "q1", "count": 10, "plants": [...]}}
{{"final": {{"targets": ["q1"], "answer": null, "reject": false, "reason": "bags 1 to 10 along row 11"}}}}"""


def system_prompt(R, arm, max_calls=8):
    reg = Registry(R, arm); ex1 = reg.get_plant_at(11, 9)
    tools = TOOLTEXT.format(show=SHOW, bag_fields=', '.join(FIELDS[arm]['bag']), tree_fields=', '.join(FIELDS[arm]['tree']), added_note=ADDED_NOTE[arm])
    return PROTOCOL.format(describe=DESCRIBE[arm].format(rows=R['rows']), tools=tools, max_calls=max_calls, ex1=json.dumps(ex1), ex1_id=ex1['id'])


if __name__ == '__main__':   # print both system prompts (for the design file):  python3 azalai_h1_tools.py registry_v0.json
    import sys
    R = json.load(open(sys.argv[1]))
    for arm in ('twin', 'enumerated'): print(f'===== {arm} =====\n{system_prompt(R, arm)}\n')
