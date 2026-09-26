"""Benchmarks beyond the three in the paper (cantilever, plate with hole and L-bracket live in
atlas/energy_solver.py and atlas/energy_xpinn.py).

    multi-material (results/multi_material/)
        bimaterial         plate with a soft circular inclusion              passed
        steel_insert       steel insert in an aluminium plate                passed
        filled_slot        epoxy-filled elliptical slot in a steel plate     passed (weak)
        sandwich           sandwich panel                                    failed candidate
        fibre_composite    glass-fibre / epoxy composite                     failed candidate
        transition_joint   steel-aluminium joint with a wavy bond line       failed candidate
    single-material, hard (results/hard_single_material/)
        notched_plate      U-notched plate
        multicutout        L-bracket with three holes and a slot
"""
