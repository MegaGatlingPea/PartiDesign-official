"""Short-range safeguards for attractive zero-LJ 1-4 exceptions during relaxation.

Amber polar H can have zero Lennard-Jones epsilon. Its attractive 1-4 Coulomb
exception is then unbounded at zero separation. Add a repulsive r^-12 guard
whose isolated minimum with that Coulomb term is 0.10 nm. At 0.20 nm its
energy is only 1/(12*2^11) of the attractive Coulomb magnitude. Original
parameters, exclusions, atom identities and normal contact geometry stay intact.
"""
from __future__ import annotations


def guard_zero_lj_exceptions(system, topology):
    """Bound attractive, unshielded exceptions involving hydrogen; return count."""
    from openmm import CustomBondForce, NonbondedForce, unit
    atoms = list(topology.atoms())
    guard = CustomBondForce("a/r^12")
    guard.setName("PartiDesign short-range exception safeguard")
    guard.addPerBondParameter("a")
    minimum_nm = 0.10
    count = 0
    for force in system.getForces():
        if not isinstance(force, NonbondedForce):
            continue
        for index in range(force.getNumExceptions()):
            i, j, product, _sigma, epsilon = force.getExceptionParameters(index)
            charge_product = product.value_in_unit(unit.elementary_charge**2)
            if charge_product >= 0 or epsilon.value_in_unit(unit.kilojoule_per_mole) > 0:
                continue
            if not any(atoms[row].element is not None and atoms[row].element.symbol == "H"
                       for row in (i, j)):
                continue
            if all(system.getParticleMass(row).value_in_unit(unit.dalton) == 0 for row in (i, j)):
                continue
            # OpenMM Coulomb constant in kJ/mol, nm and elementary charge.
            coefficient = 138.935456 * abs(charge_product) * minimum_nm**11 / 12.0
            guard.addBond(i, j, [coefficient])
            count += 1
    if count:
        system.addForce(guard)
    return count
