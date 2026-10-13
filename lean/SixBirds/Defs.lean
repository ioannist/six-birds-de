universe u v

/-- A lens wraps a map from microstates to macrostates. -/
structure Lens (X : Type u) (Y : Type v) where
  f : X → Y

/-- A completion wraps a map from macrostates to microstates. -/
structure Completion (X : Type u) (Y : Type v) where
  U : Y → X

/-- Packaging operator induced by a lens and completion. -/
def Packaging {X : Type u} {Y : Type v} (lens : Lens X Y) (comp : Completion X Y) : X → X :=
  fun x => comp.U (lens.f x)

/-- Exact vanishing of the state-space route mismatch at a point. -/
def ZeroRouteMismatchAt {X : Type u} (T : X → X) (E : X → X) (x : X) : Prop :=
  E (T x) = T (E x)

/-- Exact vanishing of state-space route mismatch at every point. -/
def ZeroRouteMismatch {X : Type u} (T : X → X) (E : X → X) : Prop :=
  ∀ x, ZeroRouteMismatchAt T E x

/-- A genuine obstruction to commutation at a point. -/
def NonzeroRouteMismatchAt {X : Type u} (T E : X → X) (x : X) : Prop :=
  E (T x) ≠ T (E x)

/-- Historical name: this means ZERO mismatch, not noncommutation. -/
abbrev RouteMismatchAt := @ZeroRouteMismatchAt

/-- Historical name: this means ZERO mismatch, not noncommutation. -/
abbrev RouteMismatch := @ZeroRouteMismatch

/-- Idempotence defect at a point: E(E x) = E x. -/
def IdempotenceDefectAt {X : Type u} (E : X → X) (x : X) : Prop :=
  E (E x) = E x

/-- Idempotence: defect is zero at all points. -/
def Idempotent {X : Type u} (E : X → X) : Prop :=
  ∀ x, IdempotenceDefectAt E x

/-- The completion is a right inverse of the lens. -/
def IsSection {X : Type u} {Y : Type v} (lens : Lens X Y) (comp : Completion X Y) : Prop :=
  ∀ y, lens.f (comp.U y) = y

/-- The evolved observable depends only on the retained observable. -/
def FiberCompatible {X : Type u} {Y : Type v} (lens : Lens X Y) (T : X → X) : Prop :=
  ∀ x x', lens.f x = lens.f x' → lens.f (T x) = lens.f (T x')

/-- An exact deterministic macro law, rather than an assumed commuting completion. -/
def MacroDescends {X : Type u} {Y : Type v} (lens : Lens X Y) (T : X → X) : Prop :=
  ∃ S : Y → Y, ∀ x, lens.f (T x) = S (lens.f x)

/-- A distance-like readout with only the equality-to-zero law required here.
No separation property is assumed: zero readout need not imply equality. -/
structure ReflexiveReadout (X : Type u) (D : Type v) [OfNat D 0] where
  distance : X → X → D
  self_zero : ∀ x, distance x x = 0
