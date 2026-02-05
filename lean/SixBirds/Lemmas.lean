import SixBirds.Defs

universe u v

/-- If E ∘ T = T ∘ E, then RouteMismatch holds. -/
theorem routeMismatch_of_commute {X : Type u} (T E : X → X)
    (h : E ∘ T = T ∘ E) : RouteMismatch T E := by
  intro x
  have h' := congrArg (fun f => f x) h
  simpa [RouteMismatchAt, Function.comp] using h'

/-- Idempotent implies the pointwise idempotence defect is zero. -/
theorem idempotenceDefectAt_of_idempotent {X : Type u} (E : X → X)
    (h : Idempotent E) (x : X) : IdempotenceDefectAt E x := by
  exact h x

/-- Packaging commutes with T given a homomorphism and intertwined completion. -/
theorem routeMismatch_packaging_of_homomorphism_section {X : Type u} {Y : Type v}
    (lens : Lens X Y) (comp : Completion X Y) (T : X → X) (S : Y → Y)
    (h_hom : lens.f ∘ T = S ∘ lens.f)
    (h_int : T ∘ comp.U = comp.U ∘ S)
    (_h_sec : lens.f ∘ comp.U = id) :
    RouteMismatch T (Packaging lens comp) := by
  have h_comm : (Packaging lens comp) ∘ T = T ∘ (Packaging lens comp) := by
    funext x
    have h_hom_x := congrArg (fun f => f x) h_hom
    have h_int_x := congrArg (fun f => f (lens.f x)) h_int
    have h_int_x' : comp.U (S (lens.f x)) = T (comp.U (lens.f x)) := by
      simpa [Function.comp] using h_int_x.symm
    calc
      comp.U (lens.f (T x)) = comp.U (S (lens.f x)) := by
        simpa [Function.comp] using congrArg comp.U h_hom_x
      _ = T (comp.U (lens.f x)) := h_int_x'
  exact routeMismatch_of_commute T (Packaging lens comp) h_comm
