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

/-- Route mismatch at a point: commuting of T and E. -/
def RouteMismatchAt {X : Type u} (T : X → X) (E : X → X) (x : X) : Prop :=
  E (T x) = T (E x)

/-- Route mismatch: commuting at all points. -/
def RouteMismatch {X : Type u} (T : X → X) (E : X → X) : Prop :=
  ∀ x, RouteMismatchAt T E x

/-- Idempotence defect at a point: E(E x) = E x. -/
def IdempotenceDefectAt {X : Type u} (E : X → X) (x : X) : Prop :=
  E (E x) = E x

/-- Idempotence: defect is zero at all points. -/
def Idempotent {X : Type u} (E : X → X) : Prop :=
  ∀ x, IdempotenceDefectAt E x
