import SixBirds.Defs

universe u v

/-- Commutation means exact zero mismatch. -/
theorem zeroRouteMismatch_of_commute {X : Type u} (T E : X → X)
    (h : E ∘ T = T ∘ E) : ZeroRouteMismatch T E := by
  intro x
  have h' := congrArg (fun f => f x) h
  simpa [ZeroRouteMismatchAt, Function.comp] using h'

/-- Historical export; its conclusion is exact ZERO mismatch. -/
theorem routeMismatch_of_commute {X : Type u} (T E : X → X)
    (h : E ∘ T = T ∘ E) : RouteMismatch T E :=
  zeroRouteMismatch_of_commute T E h

/-- Idempotent implies the pointwise idempotence defect is zero. -/
theorem idempotenceDefectAt_of_idempotent {X : Type u} (E : X → X)
    (h : Idempotent E) (x : X) : IdempotenceDefectAt E x := by
  exact h x

/-- Homomorphism and intertwining suffice; no section hypothesis is needed. -/
theorem zeroRouteMismatch_packaging_of_intertwining {X : Type u} {Y : Type v}
    (lens : Lens X Y) (comp : Completion X Y) (T : X → X) (S : Y → Y)
    (h_hom : lens.f ∘ T = S ∘ lens.f)
    (h_int : T ∘ comp.U = comp.U ∘ S) :
    ZeroRouteMismatch T (Packaging lens comp) := by
  intro x
  have h_hom_x := congrArg (fun f => f x) h_hom
  have h_int_x := congrArg (fun f => f (lens.f x)) h_int
  change comp.U (lens.f (T x)) = T (comp.U (lens.f x))
  calc
    comp.U (lens.f (T x)) = comp.U (S (lens.f x)) := by
      simpa [Function.comp] using congrArg comp.U h_hom_x
    _ = T (comp.U (lens.f x)) := by
      simpa [Function.comp] using h_int_x.symm

/-- Compatibility export: the section assumption is redundant for commutation. -/
theorem routeMismatch_packaging_of_homomorphism_section {X : Type u} {Y : Type v}
    (lens : Lens X Y) (comp : Completion X Y) (T : X → X) (S : Y → Y)
    (h_hom : lens.f ∘ T = S ∘ lens.f)
    (h_int : T ∘ comp.U = comp.U ∘ S)
    (_h_sec : lens.f ∘ comp.U = id) : RouteMismatch T (Packaging lens comp) :=
  zeroRouteMismatch_packaging_of_intertwining lens comp T S h_hom h_int

theorem packaging_idempotent_of_section {X : Type u} {Y : Type v}
    (lens : Lens X Y) (comp : Completion X Y) (h : IsSection lens comp) :
    Idempotent (Packaging lens comp) := by
  intro x
  change comp.U (lens.f (comp.U (lens.f x))) = comp.U (lens.f x)
  rw [h]

/-- Exact descent is equivalent to constancy on lens fibers when a section exists.
The section constructs the law S = f ∘ T ∘ U; it is not assumed as an input law. -/
theorem macroDescends_iff_fiberCompatible {X : Type u} {Y : Type v}
    (lens : Lens X Y) (comp : Completion X Y) (T : X → X)
    (h_sec : IsSection lens comp) :
    MacroDescends lens T ↔ FiberCompatible lens T := by
  constructor
  · rintro ⟨S, hS⟩ x x' h
    rw [hS x, hS x', h]
  · intro h
    refine ⟨fun y => lens.f (T (comp.U y)), ?_⟩
    intro x
    exact h x (comp.U (lens.f x)) (h_sec (lens.f x)).symm

/-- One pair in a common fiber with different evolved readouts forbids every law S. -/
theorem no_macroDescends_of_fiber_witness {X : Type u} {Y : Type v}
    (lens : Lens X Y) (T : X → X) (x x' : X)
    (h_same : lens.f x = lens.f x') (h_diff : lens.f (T x) ≠ lens.f (T x')) :
    ¬ MacroDescends lens T := by
  rintro ⟨S, hS⟩
  apply h_diff
  rw [hS x, hS x', h_same]

/-- Observable closure is the weaker condition E T = E T E, not E T = T E. -/
theorem fiberCompatible_iff_packaged_evolution {X : Type u} {Y : Type v}
    (lens : Lens X Y) (comp : Completion X Y) (T : X → X)
    (h_sec : IsSection lens comp) :
    FiberCompatible lens T ↔
      ∀ x, Packaging lens comp (T x) = Packaging lens comp (T (Packaging lens comp x)) := by
  constructor
  · intro h x
    exact congrArg comp.U (h x (Packaging lens comp x) (h_sec (lens.f x)).symm)
  · intro h x x' h_same
    have h_obs (z : X) : lens.f (T z) = lens.f (T (Packaging lens comp z)) := by
      have hz := congrArg lens.f (h z)
      change lens.f (comp.U (lens.f (T z))) =
        lens.f (comp.U (lens.f (T (Packaging lens comp z)))) at hz
      rw [h_sec _, h_sec _] at hz
      exact hz
    rw [h_obs x, h_obs x']
    change lens.f (T (comp.U (lens.f x))) = lens.f (T (comp.U (lens.f x')))
    rw [h_same]

theorem route_readout_zero_of_exact {X : Type u} {D : Type v} [OfNat D 0]
    (d : ReflexiveReadout X D) (T E : X → X) (x : X)
    (h : ZeroRouteMismatchAt T E x) : d.distance (E (T x)) (T (E x)) = 0 := by
  change E (T x) = T (E x) at h
  rw [h]
  exact d.self_zero _

theorem idempotence_readout_zero_of_exact {X : Type u} {D : Type v} [OfNat D 0]
    (d : ReflexiveReadout X D) (E : X → X) (x : X)
    (h : IdempotenceDefectAt E x) : d.distance (E (E x)) (E x) = 0 := by
  change E (E x) = E x at h
  rw [h]
  exact d.self_zero _

-- The manuscript cites these qualified names. Retain both those names and the
-- original root exports, with the zero-mismatch meaning made explicit above.
namespace SixBirds.Lemmas

theorem routeMismatch_of_commute {X : Type u} (T E : X → X)
    (h : E ∘ T = T ∘ E) : ZeroRouteMismatch T E :=
  _root_.zeroRouteMismatch_of_commute T E h

theorem idempotenceDefectAt_of_idempotent {X : Type u} (E : X → X)
    (h : Idempotent E) (x : X) : IdempotenceDefectAt E x :=
  _root_.idempotenceDefectAt_of_idempotent E h x

theorem routeMismatch_packaging_of_homomorphism_section {X : Type u} {Y : Type v}
    (lens : Lens X Y) (comp : Completion X Y) (T : X → X) (S : Y → Y)
    (h_hom : lens.f ∘ T = S ∘ lens.f)
    (h_int : T ∘ comp.U = comp.U ∘ S)
    (h_sec : lens.f ∘ comp.U = id) : ZeroRouteMismatch T (Packaging lens comp) :=
  _root_.routeMismatch_packaging_of_homomorphism_section lens comp T S h_hom h_int h_sec

end SixBirds.Lemmas
