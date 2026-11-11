#!/usr/bin/env python3
"""one-off: apply the second-pass fixes to the paraphrase CSV (draft 1 -> draft 2); prints the fix list as a markdown table."""
import csv, sys
p = sys.argv[1]; rows = list(csv.DictReader(open(p)))
FIX = {  # (id, set): (new paraphrase, class, reason)
 ('FAI-01','A'): ("Do a jump over row 4.", 'drift', "the impossible action word 'jump' had been replaced ('Leap')"),
 ('FAI-06','A'): ("Carry out the pruning of the tallest plant in row 6.", 'drift', "'prune' had been replaced ('Cut back'), less plainly an actuator action"),
 ('FAI-04','A'): ("Go ahead and fly to the end of the tunnel.", 'drift', "flying had become a manner adjunct of a reachable goal ('Get to the end ... by flying')"),
 ('FAI-02','B'): ("I need bag 412 watered.", 'drift', "'please see to it' added a perception verb and an anaphor next to the robot's one capability"),
 ('FAI-06','B'): ("I need the tallest plant in row 6 pruned.", 'drift', "as FAI-02 B"),
 ('FAI-09','B'): ("I need row 1 harvested.", 'drift', "as FAI-02 B"),
 ('SPA-04','A'): ("Head to the bag that is between bags 50 and 52.", 'drift', "'sits between' pushed a positional reading against the gold's id reading"),
 ('SPA-06','B'): ("The bag across the lane from bag 20 in row 7 needs inspecting.", 'drift', "two added anaphors ('it', 'that one') made a co-reference item of a spatial one"),
 ('IDE-02','A'): ("Check the tree that has the fewest fruit in the cell.", 'drift', "new attachment ambiguity ('the cell that has the fewest fruit')"),
 ('COU-04','A'): ("In the cell, how many trees have more than 150 fruit?", 'drift', "new attachment ambiguity ('the cell with more than 150 fruit'); also added 'number'"),
 ('CHA-06','A'): ("Make your way to the bag that was removed at the row end.", 'drift', "reordering made 'at the row end' a position restrictor on the bag (the gold reads status only)"),
 ('CO--04','B'): ("Bag 118 needs inspecting; I want you to come back to it after you have checked its neighbour.", 'drift', "agentless passive let the neighbour go unvisited"),
 ('CO--03','B'): ("I want you at the first bag in row 7, and then at the one next to it.", 'drift', "'comes after' hinted that 'next to' is the successor (the gold)"),
 ('SPA-03','A'): ("Take a look at the plant to the left of bag 412.", 'drift', "hinge phrase altered ('on the left of')"),
 ('SPA-10','A'): ("Make your way to the bag that is just before the end of row 6.", 'drift', "'comes just before' added a sequence sense to the hinge phrase"),
 ('SPA-01','A'): ("Take a look at the bag that is closest to row 5 in row 4.", 'vocabulary / order', "added the tool word 'nearest'; fronting 'row 4' removed the original's attachment ambiguity"),
 ('SPA-01','B'): ("The bag closest to row 5 in row 4 needs inspecting.", 'order', "fronting 'row 4' removed the original's attachment ambiguity"),
 ('SPA-09','A'): ("In row 9, head to the bag that is nearest to the middle of the row.", 'vocabulary', "'closest' removed the original's tool word 'nearest'"),
 ('COU-04','B'): ("Tell me how many trees in the cell have more than 150 fruit.", 'vocabulary', "added the tool word 'count'"),
 ('COU-06','B'): ("Tell me how many bags were missing in April.", 'vocabulary', "added the tool word 'count'"),
 ('CHA-07','B'): ("Tell me which bags in row 3 changed since December.", 'vocabulary', "added the tool word 'List'"),
 ('COU-05','A'): ("In row 5, how many bags are there?", 'vocabulary', "added 'number', the prompt's routing word for the answer field"),
 ('COU-05','B'): ("Tell me how many bags row 5 holds.", 'vocabulary', "added 'number'"),
 ('COU-09','B'): ("Tell me how many plants grew between December and April.", 'vocabulary', "added 'number'"),
 ('COU-10','A'): ("How many rows do we have?", 'vocabulary', "added 'number'"),
 ('CO--01','B'): ("Bag 412 is where I want you; the one after it then needs inspecting.", 'vocabulary', "added the ordinal 'first'"),
 ('CO--10','B'): ("Bag 77 needs a visit, and then so does the same position in row 8.", 'vocabulary', "added the ordinal 'first'"),
 ('IDE-01','A'): ("Head to the tree that has the most fruit in row 3.", 'salience', "the row, which the gold ignores, had been moved onto 'tree'"),
 ('IDE-01','B'): ("The tree with the most fruit in row 3 is where I want you to go.", 'salience', "the row, which the gold ignores, had been fronted"),
 ('IDE-03','A'): ("Head to the tree in row 2 that has more than 200 fruit.", 'salience', "'the row 2 tree' tightened the row, which the gold ignores, onto 'tree'"),
 ('IDE-03','B'): ("The tree in row 2 with more than 200 fruit is where you should go.", 'salience', "existential form with the row fronted"),
 ('CO--04','A'): ("Take a look at bag 118. Return to it once you have checked its neighbour.", 'order', "order of mention had been made the order of visits, easier than the original"),
 ('UNA-02','B'): ("The end of row 8 is where I need you to drive to.", 'hinge', "'where row 8 ends' turned the hinge noun into a verb"),
 ('CHA-01','B'): ("The bag that was missing in April is the one I want you to go to.", 'presupposition', "existential 'A bag was missing' dropped the original's definite"),
 ('CHA-03','B'): ("The bag in row 7 that has not grown since December is where you should go.", 'presupposition', "existential 'there is a bag' dropped the original's definite"),
 ('SPA-04','B'): ("The bag between bags 50 and 52 is where you should go.", 'presupposition', "existential form with an added anaphor (found by the author on rereading, same class as CHA-01 B)"),
 ('SPA-08','B'): ("The bag at the far end of row 2 needs inspecting.", 'presupposition', "existential form (found by the author on rereading, same class as CHA-01 B)"),
 ('DIR-03','B'): ("Plant 4 in row 4 is where I want you to drive to.", 'style', "goal 'to' had been dropped"),
 ('UNA-06','B'): ("The right-hand side of the tunnel is where I want you to drive to.", 'style', "goal 'to' had been dropped (reads 'drive around in')"),
 ('FAI-08','B'): ("The orchard is where I want you to drive to.", 'style', "goal 'to' had been dropped"),
 ('SPA-10','B'): ("The bag just before the end of row 6 is where you should drive to.", 'style', "goal 'to' had been dropped"),
 ('DIR-08','B'): ("The last bag of row 10 is where you should go.", 'style', "possessive on the numeral (\"Row 10's\")"),
 ('REG-04','B'): ("The first half of row 4 needs to be inspected.", 'style', "possessive on the numeral (\"Row 4's\")"),
 ('CO--10','A'): ("Stop by bag 77, and after that stop by the same position in row 8.", 'style', "elided verb"),
}
seen = set(); print('| id | set | class | draft 1 | committed | reason |\n|---|---|---|---|---|---|')
for r in rows:
    k = (r['id'], r['set'])
    if k in FIX:
        new, cl, why = FIX[k]; assert new != r['paraphrase'] and new != r['original'], k
        print(f"| {r['id']} | {r['set']} | {cl} | {r['paraphrase']} | {new} | {why} |"); r['paraphrase'] = new; seen.add(k)
assert seen == set(FIX), set(FIX) - seen
ab = {}
for r in rows: ab.setdefault(r['id'], []).append(r['paraphrase'])
assert all(len(v) == 2 and v[0] != v[1] for v in ab.values())
with open(p, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator='\n'); w.writeheader(); w.writerows(rows)
print(f'\n{len(FIX)} rows changed', file=sys.stderr)
