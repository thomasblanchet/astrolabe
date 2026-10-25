import Spec.L2.Closure
import Spec.L5.Requirements

/-!
# Layer 5: the system is sound (`T-14`)
-/

namespace Spec

/-- `T-14`. A precondition that a sound decision proves from the record of an expression holds of
what the expression denotes, under every binding whose variables satisfy their declarations. -/
theorem T14.sound {V : Type} (decl : V → Record) (β : V → ColumnSet)
    (hdecl : ∀ v, describes (decl v) (β v)) (e : RemExpr V) (pre : ColumnSet → Prop)
    (decision : Record → Prop) (hsound : ∀ r X, decision r → describes r X → pre X)
    (hdecided : decision (know decl e)) : pre (e.denote β) :=
  hsound _ _ hdecided (T1.over_approximates decl β hdecl e)

/-- `T-14` for a per-column requirement: a row of `T-12` that holds of the record of an expression
makes the requirement hold of what the expression denotes. -/
theorem T14.requirement_sound {V : Type} (decl : V → Record) (β : V → ColumnSet)
    (hdecl : ∀ v, describes (decl v) (β v)) (e : RemExpr V) (ρ : PerColumn)
    (hrow : ρ.decides (know decl e)) : ρ.holds (e.denote β) :=
  T14.sound decl β hdecl e ρ.holds ρ.decides
    (fun r X hr hX => (T12.decide_iff ρ r).2 (Or.inl hr) X hX) hrow

end Spec
