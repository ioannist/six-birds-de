import SixBirds.Lemmas

namespace SixBirds.Examples

/-- A lawful projection completion with an invisible evolved coordinate. -/
def firstLens : Lens (Bool × Bool) Bool := ⟨Prod.fst⟩
def firstCompletion : Completion (Bool × Bool) Bool := ⟨fun b => (b, false)⟩
def invisibleEvolution (x : Bool × Bool) : Bool × Bool := (x.1, true)

theorem first_section : IsSection firstLens firstCompletion := by intro b; rfl

/-- Noncommutation does NOT imply failure of observable descent. -/
theorem descent_with_nonzero_mismatch :
    MacroDescends firstLens invisibleEvolution ∧
    NonzeroRouteMismatchAt invisibleEvolution (Packaging firstLens firstCompletion) (false, false) := by
  constructor
  · exact ⟨id, fun _ => rfl⟩
  · change (false, false) ≠ (false, true)
    decide

/-- Two-point sum lens: equal sums are equal means (denominator two is fixed). -/
def sumLens : Lens (Nat × Nat) Nat := ⟨fun x => x.1 + x.2⟩
def sumCompletion : Completion (Nat × Nat) Nat := ⟨fun y => (y, 0)⟩
def quadraticEvolution (x : Nat × Nat) : Nat × Nat := (x.1 * x.1, x.2 * x.2)

/-- An actual fiber obstruction for quadratic evolution, independent of completion. -/
theorem quadratic_no_macro_law : ¬ MacroDescends sumLens quadraticEvolution := by
  exact no_macroDescends_of_fiber_witness sumLens quadraticEvolution (0, 2) (1, 1)
    (by decide) (by decide)

end SixBirds.Examples
