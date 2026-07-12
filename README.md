# PHYSARUM — a living interface

The interface is an organism. 150,000 slime-mold agents live in a WebGPU compute
shader; the page's nav labels are rasterized into a pheromone field the agents
can smell, so the colony grows transport veins toward the words on screen.
Hover a label and it reroutes to feed on it. Your cursor is also food.

A generative Web Audio patch (detuned drones, breathing lowpass, pentatonic
plucks through a feedback delay) plays forever without repeating. Cursor motion
opens the filter; feeding the organism strikes tones.

**Zero dependencies, zero build step.** Three files: `index.html`, `main.js`
(sim: agents → diffuse → render, raw WGSL), `audio.js`.

## Run

Any static server:

```sh
python3 -m http.server 8123
# open http://localhost:8123
```

Requires WebGPU (Chrome/Edge desktop, Safari 26+).

## Tune it

All behavior constants live in `writeParams()` in `main.js`:
sensor angle/distance, turn speed, agent speed, attractant weight, decay.
Agent count and sim resolution are at the top of the file.
