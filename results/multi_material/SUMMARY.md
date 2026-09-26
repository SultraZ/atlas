# Where does domain decomposition pay off? Multi-material benchmarks

On the five single-material benchmarks (cantilever, plate with hole, L-bracket, notched plate,
multi-cutout bracket) ATLAS-XPINN never beat the single-network ATLAS energy solver. The benchmarks
below test the case where it should: a material interface, across which the exact displacement has
a kink that one smooth network cannot represent but a subdomain boundary can.

Every candidate that was tried is listed, including the failures. Each had its protocol and decision
rule written down before its first run (PROTOCOL.md in each folder): ATLAS-XPINN had to beat both a
4x64 single network and a parameter-matched 4x91 one on displacement AND von Mises at seed 42, then
on the 5-seed means. Same budget (2000 Adam + 2000 L-BFGS), same quadrature points, same default
ATLAS-XPINN recipe everywhere; nothing tuned per benchmark.

| benchmark | interface | contrast | outcome | ATLAS-XPINN vs best single network (5 seeds) |
|---|---|---|---|---|
| [Steel insert in an aluminium plate](steel_insert/STEEL_INSERT.md) | closed circle, stiff inside | 3 | **pass** | displacement 3x, von Mises **10x** lower; all 5 seeds |
| [Plate with a soft circular inclusion](bimaterial/BIMATERIAL.md) | closed circle, soft inside | 15 (and 3) | **pass** | displacement 3.3x, von Mises 2.0x lower; all 5 seeds |
| [Epoxy-filled elliptical slot](filled_slot/FILLED_SLOT.md) | closed ellipse, soft inside | 60 | pass (weak) | displacement 3.8x lower in all seeds; von Mises 12% lower on average, a statistical tie |
| [Steel-aluminium transition joint](failed_candidates/transition_joint/TRANSITION_JOINT.md) | wavy line across the part | 3 | fail | better stress (4.8% vs 6.4%), worse displacement (2.0% vs 0.7%); slower convergence |
| [Sandwich panel](failed_candidates/sandwich/SANDWICH.md) | two long lines, thin faces in bending | 70 | fail | neither solver converged (compliance -5%) |
| [Fibre composite](failed_candidates/fibre_composite/FIBRE_COMPOSITE.md) | eight circles, stiff inside | 21 | fail (2 attempts) | neither solver converged (39% / 92% displacement error) |

## The pattern
ATLAS-XPINN wins when the interface is CLOSED and INSIDE the part: one subnetwork then carries a
simple field (uniform strain in a circular inclusion) and the other a plate-with-hole-like field that
a single network also handles, so the kink at the interface is the dominant error of the single
network, and decomposition removes it. The single network's error is structural, not a capacity
limit: the parameter-matched network barely improves on the interface (e.g. 21.6% -> 18.7%) while
ATLAS-XPINN reaches 1-2%.

The gain in the global von Mises error scales with how much of the stress error sits at the
interface: 10x for the stiff insert (the interface error dominates), 2x for the soft inclusion,
none for the soft-filled slot (the stress concentration sits in the steel, where decomposition along
the interface does not help).

It loses when the interface crosses the whole part (both subdomains carry half the deformation and
the augmented-Lagrangian coupling costs iterations), when thin stiff layers bend (the energy
functional is ill-conditioned for both solvers), and with many high-contrast interfaces (one set of
interface weights cannot serve both sides of a 20:1 contrast).

## Defect found and fixed along the way
The interface terms were first scaled by the problem's base material; with a soft base material
(the composite's epoxy) the traction term was 60x the strain energy and the solution collapsed. They
are now scaled by the stiffest material present. This changed nothing on any earlier benchmark
(checked bit-for-bit); the composite's first run is kept and reported.
