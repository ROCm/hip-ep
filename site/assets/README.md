# Generated assets

Maintainer notes. Not published — see the `exclude` list in `_config.yml`.

## The backdrop network

`_includes/backdrop.html` contains a field of nodes and edges used as the
landing page's background texture. The coordinates in that file are output,
not hand-drawn, and should be regenerated rather than edited in place: a
person nudging points keeps the spacing even without meaning to, and an evenly
spaced network reads as wallpaper instead of as a graph.

Run this from anywhere and paste the three `<g>` blocks over the ones in
`_includes/backdrop.html`. The seed is fixed, so re-running it unchanged
reproduces the drawing that is checked in.

```python
import random

W, H = 1600, 900
random.seed(20261021)

# Rejection sampling with a minimum separation: irregular without clumping.
MIN_D = 118
pts, tries = [], 0
while len(pts) < 40 and tries < 60000:
    tries += 1
    x = random.uniform(40, W - 40)
    y = random.uniform(36, H - 30)
    if all((x - px) ** 2 + (y - py) ** 2 > MIN_D * MIN_D for px, py in pts):
        pts.append((x, y))

# Each node to its two nearest neighbours, deduped. Varying valence; not a
# lattice.
edges = set()
for i, p in enumerate(pts):
    near = sorted(range(len(pts)),
                  key=lambda j: (pts[j][0] - p[0]) ** 2 + (pts[j][1] - p[1]) ** 2)
    for j in near[1:3]:
        edges.add((min(i, j), max(i, j)))

# Lit nodes picked along a diagonal, not at random: random selection clustered
# them in one corner.
lit = sorted(range(len(pts)), key=lambda i: pts[i][0] + pts[i][1] * 0.6)[::7][:6]

f = lambda v: round(v, 1)

print('  <g class="backdrop__edges">')
print('    <path d="' + ' '.join(
    'M%s %s L%s %s' % (f(pts[a][0]), f(pts[a][1]), f(pts[b][0]), f(pts[b][1]))
    for a, b in sorted(edges)) + '" />')
print('  </g>')
print('  <g class="backdrop__nodes">')
for i, (x, y) in enumerate(pts):
    if i in lit:
        continue
    print('    <circle cx="%s" cy="%s" r="%s" />' % (f(x), f(y), 4.5 if i % 5 == 0 else 3.2))
print('  </g>')
print('  <g class="backdrop__nodes backdrop__nodes--lit">')
for i in lit:
    print('    <circle cx="%s" cy="%s" r="5" />' % (f(pts[i][0]), f(pts[i][1])))
print('  </g>')
```

Changing `MIN_D` or the node count changes density. Keep the viewBox at
`1600 900`: `.backdrop__svg` cover-fits it, and a different aspect ratio
changes how much of the field is cropped at every viewport width rather than
just how much is in it.

Colour lives entirely in CSS (`--net-edge`, `--net-node`, `--net-node-lit`),
so the drawing follows the theme toggle and the script never emits a hex
value.
