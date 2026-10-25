import Spec.L1.Settles

/-!
# Layer 2: effects, records and remainder expressions (`E-1` to `E-6`, `D-16` to `D-20`)
-/

namespace Spec

/-- Section 6. An effect acting on one remainder. `E-4`, which combines two, is the `both` node of
a remainder expression. -/
inductive Effect
  | unchanged
  | emptied
  | filtered (C : Selector)
  | suffixed (s : Name)
  | prefixed (p : Name)
  | retyped (U : DType)

/-- Section 6. The action of an effect on a column set. -/
def Effect.apply : Effect → ColumnSet → ColumnSet
  | .unchanged, X => X
  | .emptied, _ => ∅
  | .filtered C, X => X ∩ C.denote
  | .suffixed s, X => {c | ∃ n t, (n, t) ∈ X ∧ c = (n ++ s, t)}
  | .prefixed p, X => {c | ∃ n t, (n, t) ∈ X ∧ c = (p ++ n, t)}
  | .retyped U, X => {c | ∃ n t, (n, t) ∈ X ∧ c = (n, U)}

/-- `D-16`. A record: an absence rule and a type set. -/
structure Record where
  /-- The names no column can have. -/
  absent : Absence
  /-- The data types a column can have. -/
  types : TypeSet

/-- `D-17`. A record describes a column set when no name of it is absent and every data type of
it is in the type set. -/
def describes (r : Record) (X : ColumnSet) : Prop :=
  ∀ n t, (n, t) ∈ X → n ∉ r.absent.denote ∧ t ∈ r.types.denote

/-- Section 7.1. The record rule of each effect on one remainder. -/
noncomputable def recordRule : Effect → Record → Record
  | .unchanged, r => r
  | .emptied, _ => ⟨.always, .empty⟩
  | .filtered C, r => ⟨.or r.absent (certainlyRejects C r.types), surviving C r.types⟩
  | .suffixed s, r => ⟨.or (.not (.endsWith s)) (.afterStripping s r.absent), r.types⟩
  | .prefixed p, r => ⟨.or (.not (.startsWith p)) (.beforeStripping p r.absent), r.types⟩
  | .retyped U, r => ⟨r.absent, .single U⟩

/-- Section 7.1. The record rule of `E-4`, which combines two remainders. -/
def bothRule (a b : Record) : Record :=
  ⟨.and a.absent b.absent, .union a.types b.types⟩

/-- `D-18`. A remainder expression over variables of type `V`. -/
inductive RemExpr (V : Type)
  | var (v : V)
  | app (ε : Effect) (e : RemExpr V)
  | both (e₁ e₂ : RemExpr V)

/-- `D-19`. The column set an expression denotes under a binding. -/
def RemExpr.denote {V : Type} (β : V → ColumnSet) : RemExpr V → ColumnSet
  | .var v => β v
  | .app ε e => ε.apply (e.denote β)
  | .both e₁ e₂ => e₁.denote β ∪ e₂.denote β

/-- `D-20`. The record computed for an expression, from the records its variables declare. -/
noncomputable def know {V : Type} (decl : V → Record) : RemExpr V → Record
  | .var v => decl v
  | .app ε e => recordRule ε (know decl e)
  | .both e₁ e₂ => bothRule (know decl e₁) (know decl e₂)

/-- `E-4`. The union of two column sets is a column set, that is functional, exactly when each
operand is and no name carries two different data types across them. -/
theorem E4.functional_union_iff (X Y : ColumnSet) :
    Functional (X ∪ Y) ↔
      Functional X ∧ Functional Y ∧ ∀ n t u, (n, t) ∈ X → (n, u) ∈ Y → t = u := by
  constructor
  · intro h
    exact ⟨h.mono Set.subset_union_left, h.mono Set.subset_union_right,
      fun n t u ht hu => h n t u (Or.inl ht) (Or.inr hu)⟩
  · rintro ⟨hX, hY, hXY⟩ n t u ht hu
    rcases ht with ht | ht <;> rcases hu with hu | hu
    · exact hX n t u ht hu
    · exact hXY n t u ht hu
    · exact (hXY n u t hu ht).symm
    · exact hY n t u ht hu

end Spec
