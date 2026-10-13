import Init.Data.Rat

namespace SixBirds.Examples

-- At a_void=a_wall=1, for the actual two-population generator settings
-- g4pi=rho_mean=1, f_void=delta_rho=1/2, kappa_scale=52/3.
def witnessRhoVoid : Rat := 1/2
def witnessRhoWall : Rat := 3/2
def witnessKappaVoid : Rat := (52/3) * (1 - witnessRhoVoid)
def witnessLocalAccelVoid : Rat := -witnessRhoVoid/3
def witnessLocalAccelWall : Rat := -witnessRhoWall/3
def witnessVolume : Rat := (1:Rat)^3 + (1:Rat)^3
def witnessDotVolume : Rat := 3 * (1*1*3 + 1*1*1)
def witnessDDotVolume : Rat := 3 * (2*3*3+witnessLocalAccelVoid) + 3 * (2*1*1+witnessLocalAccelWall)
def witnessDomainAcceleration : Rat :=
  witnessDDotVolume / (3*witnessVolume) -
    2*witnessDotVolume*witnessDotVolume / (9*witnessVolume*witnessVolume)

-- These are exact initial-state equations. No claim about global ODE solutions
-- or a general-relativistic realization is included in this declaration.
theorem twoPatch_exact_acceleration_witness :
    (3:Rat)^2 = (2/3)*witnessRhoVoid+witnessKappaVoid ∧
    (1:Rat)^2 = (2/3)*witnessRhoWall ∧
    witnessLocalAccelVoid < 0 ∧ witnessLocalAccelWall < 0 ∧
    witnessDomainAcceleration = 5/3 ∧ witnessDomainAcceleration > 0 := by
  decide +kernel

end SixBirds.Examples
