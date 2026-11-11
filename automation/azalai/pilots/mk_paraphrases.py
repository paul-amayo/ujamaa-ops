#!/usr/bin/env python3
"""one-off: write the paraphrase CSV (draft 1) from the table below; the CSV is the source of record afterwards."""
import csv, json, sys
NUM = '/Users/paulamayo/Documents/UCT/SIL/Publications/Azalai_RSS/numbers'
bank = {r['id']: r for r in csv.DictReader(open(f'{NUM}/instruction_bank_v0.csv'))}
G = json.load(open(f'{NUM}/instruction_bank_v0_gold.json'))
P = {
 # id: (set A: a command again, other verb / clause order;  set B: another sentence form)
 'DIR-01': ("Head over to bag 412.", "I need you at bag 412."),
 'DIR-02': ("In row 7, take a look at bags 12, 13 and 14.", "Bags 12, 13 and 14 in row 7 need to be inspected."),
 'DIR-03': ("Make your way to plant 4 in row 4.", "Plant 4 in row 4 is where I want you to drive."),
 'DIR-04': ("Head to the fifth bag in row 2.", "In row 2, the fifth bag is the one I want you to go to."),
 'DIR-05': ("Check the first bag of row 9.", "The first bag in row 9 needs inspecting."),
 'DIR-06': ("Head to bag 118 in row 3.", "I want you at bag 118 in row 3."),
 'DIR-07': ("Stop by bags 40 and 41.", "Bags 40 and 41 are the ones I want visited."),
 'DIR-08': ("Head for the last bag in row 10.", "Row 10's last bag is where you should go."),
 'DIR-09': ("Have a look at plant 20 in row 6.", "Plant 20 in row 6 needs an inspection."),
 'DIR-10': ("Make your way to row 5, bag 30.", "Your destination is row 5, bag 30; drive there."),
 'UNA-01': ("Move to the beginning of row 3.", "The start of row 3 is where I want you."),
 'UNA-02': ("Make your way to the end of row 8.", "I need you to drive to where row 8 ends."),
 'UNA-05': ("Head to the middle of row 6.", "The middle of row 6 is where you should go."),
 'UNA-06': ("Make your way to the right-hand side of the tunnel.", "The right-hand side of the tunnel is where I want you to drive."),
 'UNA-07': ("Head to the far end of the nursery.", "I want you at the far end of the nursery."),
 'SPA-01': ("In row 4, check the bag that is nearest to row 5.", "Of the bags in row 4, the one closest to row 5 needs inspecting."),
 'SPA-03': ("Take a look at the plant on the left of bag 412.", "The plant to the left of bag 412 needs to be inspected."),
 'SPA-04': ("Head to the bag that sits between bags 50 and 52.", "There is a bag between bags 50 and 52; that is where you should go."),
 'SPA-05': ("Move to the bag that is behind bag 300.", "The bag behind bag 300 is the one I want you to go to."),
 'SPA-06': ("Check the bag that is across the lane from bag 20 in row 7.", "Bag 20 in row 7 has a bag across the lane from it; that one needs inspecting."),
 'SPA-08': ("Have a look at the bag at the far end of row 2.", "At the far end of row 2 there is a bag that I want inspected."),
 'SPA-09': ("In row 9, head to the bag that is closest to the middle of the row.", "The bag I want you at is the one in row 9 nearest to the middle of the row."),
 'SPA-10': ("Make your way to the bag that comes just before the end of row 6.", "The bag just before the end of row 6 is where you should drive."),
 'REG-01': ("Look over the plants on the right side of the tunnel.", "The plants on the right side of the tunnel need inspecting."),
 'REG-03': ("Work your way through rows 1 to 3.", "Rows 1 to 3 are the ones I want you to go through."),
 'REG-04': ("Check the first half of row 4.", "Row 4's first half needs to be inspected."),
 'REG-05': ("Inspect the back of the nursery.", "The back of the nursery needs checking."),
 'REG-06': ("Look at every single bag in row 10.", "Each bag in row 10 needs to be inspected."),
 'REG-07': ("Carry out a survey of the left block.", "The left block needs surveying."),
 'REG-08': ("Head for the rows near the entrance.", "The rows near the entrance are where I want you to go."),
 'REG-09': ("Inspect the middle rows.", "The middle rows need checking."),
 'REG-10': ("Take a look at the corner of the nursery.", "The corner of the nursery needs inspecting."),
 'FAI-01': ("Leap over row 4.", "I want you to jump over row 4."),
 'FAI-02': ("Give bag 412 some water.", "Bag 412 needs watering; please see to it."),
 'FAI-03': ("Head over to row 14.", "Row 14 is where I need you."),
 'FAI-04': ("Get to the end of the tunnel by flying there.", "I need you to fly to the end of the tunnel."),
 'FAI-05': ("Head to the tomato in row 2.", "The tomato in row 2 is where I want you to go."),
 'FAI-06': ("Cut back the tallest plant in row 6.", "The tallest plant in row 6 needs pruning; please see to it."),
 'FAI-07': ("Head over to bag 5000.", "I need you at bag 5000."),
 'FAI-08': ("Make your way to the orchard.", "The orchard is where I want you to drive."),
 'FAI-09': ("Carry out the harvest of row 1.", "Row 1 needs harvesting; please see to it."),
 'FAI-10': ("Head to whichever plant is going to be the tallest next year.", "Next year one plant will be the tallest; that plant is where I want you to go."),
 'IDE-01': ("Head to the tree in row 3 that has the most fruit.", "In row 3, the tree carrying the most fruit is where I want you to go."),
 'IDE-02': ("Check the tree in the cell that has the fewest fruit.", "The tree with the fewest fruit in the cell needs inspecting."),
 'IDE-03': ("Head to the row 2 tree that has more than 200 fruit.", "In row 2 there is a tree with more than 200 fruit; that is where you should go."),
 'IDE-07': ("Head to the tree that has no fruit.", "The tree with no fruit is where I want you to go."),
 'IDE-10': ("Head to the grow bag in row 7 that holds the tallest plant.", "In row 7, the grow bag with the tallest plant is where you should go."),
 'COU-04': ("What is the number of trees in the cell with more than 150 fruit?", "Count the trees in the cell that have more than 150 fruit."),
 'COU-05': ("What is the number of bags in row 5?", "Tell me the number of bags that row 5 holds."),
 'COU-06': ("In April, how many bags were missing?", "Give me the count of bags that were missing in April."),
 'COU-09': ("Between December and April, how many plants grew?", "Give me the number of plants that grew between December and April."),
 'COU-10': ("What is the number of rows?", "Tell me how many rows there are."),
 'CHA-01': ("Head to the bag that was missing in April.", "A bag was missing in April; that is the one I want you to go to."),
 'CHA-02': ("Check the bags that are new since December.", "The bags that are new since December need to be inspected."),
 'CHA-03': ("Head to the row 7 bag that has not grown since December.", "In row 7 there is a bag that has not grown since December; that is where you should go."),
 'CHA-04': ("Head to whichever plant shrank the most.", "The plant that shrank the most is the one I want you to go to."),
 'CHA-06': ("Head to the bag at the row end that was removed.", "The bag that was removed at the row end is where I want you to go."),
 'CHA-07': ("In row 3, which bags changed since December?", "List the bags in row 3 that changed since December."),
 'CHA-08': ("Head to the row 2 plant that grew the most.", "In row 2, the plant that grew the most is where you should go."),
 'CHA-09': ("Take a look at the bags that were unclear in the April check.", "The bags that were unclear in the April check need to be inspected."),
 'CHA-10': ("Head to the bag that was missing in April and is back now.", "The bag that was missing in April and is back now is where I want you to go."),
 # supplementary block (outside the primary 61): the eight co-reference instructions that the gold builder has a resolver for
 'CO--01': ("Head to bag 412; once there, check the one after it.", "Bag 412 is your first stop; the one after it then needs inspecting."),
 'CO--02': ("Check bag 30 in row 2 as well as the bag opposite it.", "Bag 30 in row 2 and the bag opposite it both need inspecting."),
 'CO--03': ("Head to the first bag in row 7, and after that to the one next to it.", "The first bag in row 7 is where I want you first; the one next to it comes after."),
 'CO--04': ("Take a look at bag 118. Once you have checked its neighbour, return to it.", "Bag 118 needs inspecting; after its neighbour has been checked, I want you to come back to it."),
 'CO--05': ("Head to row 5 and check its tallest plant.", "Row 5 is where I want you; its tallest plant needs inspecting."),
 'CO--07': ("Inspect bag 200 together with the two either side of it.", "Bag 200 and the two either side of it need checking."),
 'CO--08': ("Check the second bag in row 4, and after that its partner in the next row.", "The second bag in row 4 needs inspecting, followed by its partner in the next row."),
 'CO--10': ("Stop by bag 77, and after that the same position in row 8.", "Bag 77 is the first one I want visited, followed by the same position in row 8."),
}
res = [i for i, g in G.items() if 'unresolvable' not in g]; cor = ['CO--01', 'CO--02', 'CO--03', 'CO--04', 'CO--05', 'CO--07', 'CO--08', 'CO--10']
assert len(res) == 61 and set(P) == set(res) | set(cor), (set(res) | set(cor)) ^ set(P)
with open(sys.argv[1], 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n'); w.writerow(['id', 'category', 'site', 'block', 'set', 'original', 'paraphrase', 'target_rule'])
    for i in [r for r in bank if r in P]:
        for s, p in zip('AB', P[i]):
            assert p != bank[i]['instruction']
            w.writerow([i, bank[i]['category'], bank[i]['site'], 'primary' if i in res else 'supplementary co-reference', s, bank[i]['instruction'], p, bank[i]['target_rule']])
print('rows', 2 * len(P))
