import Spec.L0.Objects

/-!
# Layer 1: the absence language (`D-9` to `D-11`)

An absence rule is a predicate on names. Where it holds, no column of that name can be present.
-/

namespace Spec

/-- `D-11`. A name expression: a set of names. `D-11` restricts it to a regular set; the predicate
`NameExpr.Regular` in layer 4 states that restriction, and layer 4 proves that every rule the
specification builds meets it. -/
structure NameExpr where
  /-- The names the expression accepts. -/
  lang : Set Name

/-- `D-11`. The names a name expression accepts. -/
def NameExpr.denote (m : NameExpr) : Set Name :=
  m.lang

/-- `D-9`. The absence language. -/
inductive Absence
  | never
  | always
  | oneOf (S : List Name)
  | startsWith (s : Name)
  | endsWith (s : Name)
  | contains (s : Name)
  | matching (m : NameExpr)
  | not (P : Absence)
  | afterStripping (s : Name) (P : Absence)
  | beforeStripping (s : Name) (P : Absence)
  | and (P Q : Absence)
  | or (P Q : Absence)

/-- `D-10`. The names an absence rule holds of. -/
def Absence.denote : Absence → Set Name
  | .never => ∅
  | .always => Set.univ
  | .oneOf S => {n | n ∈ S}
  | .startsWith s => {n | s <+: n}
  | .endsWith s => {n | s <:+ n}
  | .contains s => {n | s <:+: n}
  | .matching m => m.denote
  | .not P => (P.denote)ᶜ
  | .afterStripping s P => {w | ∃ n ∈ P.denote, w = n ++ s}
  | .beforeStripping s P => {w | ∃ n ∈ P.denote, w = s ++ n}
  | .and P Q => P.denote ∩ Q.denote
  | .or P Q => P.denote ∪ Q.denote

end Spec
