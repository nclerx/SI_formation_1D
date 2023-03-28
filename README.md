# SI_formation_1D
Numerically simulates heat flow and associated superimposed ice (SI) formation in snow and ice.

- Simulation is numerically in a layered 1D domain.
- Model can take presence of irreducible water inside the snowpack into account.

Shortcomings: 
- currently the modelling domain is fixed, which means growth of SI does not alter snow thickness or thickness of the ice slab
- No melt processes are simulated
- No percolation of water is simulated, conditions inside the snowpack (irreducible water content, presence of slush) are fixed over time.